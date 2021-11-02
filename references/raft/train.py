import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torchvision.datasets import KittiFlow, FlyingChairs, FlyingThings3D, Sintel
from torchvision.models.video import RAFT

from presets import OpticalFlowPresetTrain, OpticalFlowPresetEval
from utils import MetricLogger, setup_ddp, sequence_loss, InputPadder


def get_train_dataset(dataset_name, small_data=False):
    d = {
        "kitti": KittiFlow,
        "chairs": FlyingChairs,
        "things": FlyingThings3D,
        "sintel": Sintel,
    }

    transforms = {
        "kitti": OpticalFlowPresetTrain(
            # resize and crop params
            crop_size=(288, 960),
            min_scale=-0.2,
            max_scale=0.4,
            stretch_prob=0,
            # flip params
            do_flip=False,
            # jitter params
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.3 / 3.14,
            asymmetric_jitter_prob=0,
        ),
        "chairs": OpticalFlowPresetTrain(crop_size=(368, 496), min_scale=0.1, max_scale=1.0, do_flip=True),
        "things": OpticalFlowPresetTrain(crop_size=(400, 720), min_scale=-0.4, max_scale=0.8, do_flip=True),
        "sintel": OpticalFlowPresetTrain(crop_size=(368, 768), min_scale=-0.2, max_scale=0.6, do_flip=True),
    }

    dataset_name = dataset_name.lower()

    if dataset_name not in d:
        raise ValueError(f"Unknown dataset {dataset_name}")

    klass = d[dataset_name]
    dataset = klass(transforms=transforms[dataset_name])

    if small_data:
        dataset._image_list = dataset._image_list[:200]
        dataset._flow_list = dataset._flow_list[:200]

    return dataset


@torch.no_grad()
def validate_sintel(model, args, iters=32, small=False):
    # FIXME: this isn't 100% accurate because some samples will be duplicated if
    # the dataset isn't divisible by the batch size.

    model.eval()

    for dstype in ["clean", "final"]:
        logger = MetricLogger(output_dir=args.output_dir)
        logger.add_meter("epe", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")
        logger.add_meter("1px", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")
        logger.add_meter("3px", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")
        logger.add_meter("5px", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")

        val_dataset = Sintel(split="training", dstype=dstype, transforms=PresetEval())
        if args.small_data:
            val_dataset._image_list = val_dataset._image_list[:200]
            val_dataset._flow_list = val_dataset._flow_list[:200]
        sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        header = f"Sintel val {dstype}"

        num_samples = 0

        def inner_loop(image1, image2, flow_gt):
            image1, image2 = image1.cuda(), image2.cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)

            flow = padder.unpad(flow_pr).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=1).sqrt()

            logger.meters["epe"].update(epe.mean().item(), n=epe.numel())
            for distance in (1, 3, 5):
                logger.meters[f"{distance}px"].update((epe < distance).float().mean().item(), n=epe.numel())

        for blob in logger.log(val_loader, header=header, sync=False, verbose=False):
            image1, image2, flow_gt = blob
            inner_loop(image1, image2, flow_gt)
            num_samples += image1.shape[0]

        num_samples = torch.tensor([num_samples], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(num_samples)
        num_samples = int(num_samples.item())
        print(f"Evaluated {num_samples} / {len(val_dataset)} samples in batch")

        if args.rank == 0:
            for i in range(num_samples, len(val_dataset)):
                image1, image2, flow_gt = val_dataset[i]
                inner_loop(image1[None, :, :, :], image2[None, :, :, :], flow_gt[None, :, :, :])

        logger.synchronize_between_processes()
        print(header, logger)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    setup_ddp(args)

    print(args)

    model = RAFT(args)  # TODO: pass better args
    model = model.to(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"), strict=False)

    torch.backends.cudnn.benchmark = True

    print("Parameter Count: %d" % count_parameters(model))

    if args.train_dataset != "chairs":
        model.module.freeze_bn()

    if args.train_dataset is None:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if args.val_dataset == ["sintel"]:
            validate_sintel(model, args)
        else:
            raise ValueError(f"can't validate on {args.val_dataset}")
        return

    model.train()

    train_dataset = get_train_dataset(args.train_dataset, small_data=args.small_data)

    # TODO: Should drop_last really be True? And shouhld it be set in the loader instead of the sampler?
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        pin_memory=True,  # TODO: find out why it was False in raft repo?
        num_workers=args.num_workers,
        # worker_init_fn=lambda x:print(f"I'm rank {args.rank} and my worker info for data loading is {torch.utils.data.get_worker_info()}", force=True)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    if args.num_steps is not None:
        extra_scheduler_args = dict(total_steps=args.num_steps + 100)
    else:
        steps_per_epoch = len(train_dataset) // (args.batch_size * args.world_size)  # +- 1 depending on drop_last?
        extra_scheduler_args = dict(epochs=args.num_epochs, steps_per_epoch=steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
        **extra_scheduler_args,
    )

    scaler = GradScaler(enabled=args.mixed_precision)  # TODO: currently untested
    logger = MetricLogger(output_dir=args.output_dir)
    logger.add_meter("epoch", tb_val="value", print=False)
    logger.add_meter("current_step", tb_val="value", print=False)
    logger.add_meter("lr", tb_val="value", print=False)
    logger.add_meter("wdecay", tb_val="value", print=False)
    logger.add_meter("last_lr", tb_val="value")
    logger.add_meter("flow_loss", tb_val="avg")
    logger.add_meter("1px", tb_val="avg")
    logger.add_meter("3px", tb_val="avg")
    logger.add_meter("5px", tb_val="avg")

    done = False
    current_epoch = current_step = 0
    while not done:
        sampler.set_epoch(current_epoch)  # set this, otherwise the data loading order would be the same for all epochs
        print(f"EPOCH {current_epoch}")

        for data_blob in logger.log(train_loader):

            # TODO: set p.grad = None instead? see https://twitter.com/karpathy/status/1299921324333170689/photo/1
            optimizer.zero_grad()

            if len(data_blob) == 4:
                image1, image2, flow, valid = data_blob
            else:
                image1, image2, flow = data_blob
                valid = ((flow[:, 0, :, :].abs() < 1000) & (flow[:, 1, :, :].abs() < 1000)).float()

            image1, image2, flow, valid = [x.cuda() for x in (image1, image2, flow, valid)]

            # print(f"I'm rank {args.rank} and I'm getting {image1[0][0][20][100]} as a given pixel value, {image1.shape = }", force=True)

            flow_predictions = model(image1, image2, iters=args.iters)

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics["epoch"] = current_epoch
            metrics["current_step"] = current_step
            metrics["last_lr"] = scheduler.get_last_lr()[0]
            metrics["lr"] = args.lr
            metrics["wdecay"] = args.wdecay
            logger.update(**metrics)

            current_step += 1

            if current_step == args.num_steps:
                done = True
                break

        logger.synchronize_between_processes()  # FIXME: actually useless since we don't print the global avg
        print(f"Epoch {current_epoch} done. ", logger)

        current_epoch += 1
        if current_epoch == args.num_epochs:
            done = True

        if args.rank == 0:
            torch.save(model.state_dict(), Path(args.output_dir) / f"{args.name}_{current_epoch}.pth")
            torch.save(model.state_dict(), Path(args.output_dir) / f"{args.name}.pth")

        if current_epoch % args.val_freq == 0 or done:
            model.eval()

            val_datasets = args.val_dataset or []
            for val_dataset in val_datasets:
                # if val_dataset == 'chairs':
                #     results.update(evaluate.validate_chairs(model.module))
                if val_dataset == "sintel":
                    validate_sintel(model, args)
                # elif val_dataset == 'kitti':
                #     results.update(evaluate.validate_kitti(model.module))

            model.train()
            if args.train_dataset != "chairs":
                model.module.freeze_bn()

    logger.close()


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--name", default="raft", help="name your experiment")
    parser.add_argument("--train-dataset", help="determines which dataset to use for training")
    parser.add_argument("--resume", help="restore checkpoint")
    parser.add_argument("--val-dataset", type=str, nargs="+")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--output-dir", default="checkpoints", type=str)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.00002)

    # TODO: make these mutually exclusive maybe?
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--num-steps", type=int, default=100000)

    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")

    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--wdecay", type=float, default=0.00005)
    parser.add_argument("--val-freq", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.8, help="exponential weighting")

    parser.add_argument("--small-data", action="store_true", help="use small data")

    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    # d = FlyingChairs(transforms=FlowAugmentor(crop_size=(368, 496), min_scale=0.1, max_scale=1.0, do_flip=True))
    # for glob in d:
    #     print(len(glob))
    #     print([x.dtype for x in glob])

    # from torchvision.datasets._optical_flow import KittiFlowDataset as K
    # dd = K()

    # d = KittiFlowDataset(transforms=FlowAugmentor(crop_size=(368, 496), min_scale=0.1, max_scale=1.0, do_flip=True))
    # d = KittiFlowDataset()
    # for glob in d:
    #     print(len(glob))

    # for glob, glob2 in zip(d, dd):
    #     print(len(glob))
    #     flow1, flow2 = glob[2], glob2[2]
    #     torch.testing.assert_close(flow1, flow2)
    #     print(flow2.min(), flow2.max(), flow2.dtype, flow2.shape)
    #     print(flow1.min(), flow1.max(), flow1.dtype, flow1.shape)

    main(args)
