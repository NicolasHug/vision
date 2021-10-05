import argparse
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torchvision.datasets import KittiFlowDataset, FlyingChairs, FlyingThings3D, Sintel
from torchvision.models.video import RAFT
from transforms import FlowAugmentor, SparseFlowAugmentor
from utils import MetricLogger, setup_ddp, sequence_loss, InputPadder


def get_train_dataset(dataset_name):
    d = {"kitti": KittiFlowDataset, "chairs": FlyingChairs, "things": FlyingThings3D, "sintel": Sintel}

    transforms = {
        "kitti": SparseFlowAugmentor(crop_size=(288, 960), min_scale=-0.2, max_scale=0.4, do_flip=False),
        "chairs": FlowAugmentor(crop_size=(368, 496), min_scale=0.1, max_scale=1.0, do_flip=True),
        "things": FlowAugmentor(crop_size=(400, 720), min_scale=-0.4, max_scale=0.8, do_flip=True),
        "sintel": FlowAugmentor(crop_size=(368, 768), min_scale=-0.2, max_scale=0.6, do_flip=True),
    }

    dataset_name = dataset_name.lower()

    if dataset_name not in d:
        raise ValueError(f"Unknown dataset {dataset_name}")

    klass = d[dataset_name]
    return klass(transforms=transforms[dataset_name])


@torch.no_grad()
def validate_sintel(model, args, iters=32):
    # FIXME: this isn't 100% accurate because some samples will be duplicated if
    # the dataset isn't divisible by the batch size.

    model.eval()
    for dstype in ["clean", "final"]:
        logger = MetricLogger(output_dir=args.output_dir)
        logger.add_meter("epe", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")
        logger.add_meter("1px", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")
        logger.add_meter("3px", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")
        logger.add_meter("5px", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")

        val_dataset = Sintel(split="training", dstype=dstype)
        sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        header = f"Sintel val {dstype}"
        for blob in logger.log(val_loader, header=header, sync=False):

            image1, image2, flow_gt, _ = blob
            image1, image2 = image1.cuda(), image2.cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=1).sqrt()

            logger.meters["epe"].update(epe.mean(), n=epe.numel())
            for distance in (1, 3, 5):
                logger.meters[f"{distance}px"].update((epe < distance).float().mean(), n=epe.numel())

        logger.synchronize_between_processes()
        print(header, logger)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    setup_ddp(args)

    # TODO: eventually remove
    torch.manual_seed(1234)
    np.random.seed(1234)

    print(args)

    model = RAFT(args)  # TODO: pass better args
    model = model.to(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"), strict=False)

    # TODO: maybe, maybe not:
    torch.backends.cudnn.benchmark = True

    print("Parameter Count: %d" % count_parameters(model))

    if args.train_dataset != "chairs":
        model.module.freeze_bn()

    if args.train_dataset is None:
        # just validate then
        if args.val_dataset == ["sintel"]:
            validate_sintel(model, args)
        else:
            raise ValueError(f"can't validate on {args.val_dataset}")
        return

    model.train()

    train_dataset = get_train_dataset(args.train_dataset)

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
    logger.add_meter("flow_loss", tb_val="value")
    logger.add_meter("1px", tb_val="value")
    logger.add_meter("3px", tb_val="value")
    logger.add_meter("5px", tb_val="value")

    VAL_FREQ = 2  # validate every X epochs

    done = False
    current_epoch = current_step = 0
    while not done:
        sampler.set_epoch(current_epoch)  # set this, otherwise the data loading order would be the same for all epochs
        print(f"EPOCH {current_epoch}")

        for data_blob in logger.log(train_loader):

            # TODO: set p.grad = None instead? see https://twitter.com/karpathy/status/1299921324333170689/photo/1
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.to(args.local_rank) for x in data_blob]

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

            if args.num_steps is not None and current_step == args.num_steps:
                done = True
                break

        logger.synchronize_between_processes()
        print(f"Epoch {current_epoch} done. ", logger, force=True)

        current_epoch += 1
        if args.num_epochs is not None and current_epoch == args.num_epochs:
            done = True

        if args.rank == 0:
            torch.save(model.state_dict(), Path(args.output_dir) / f"{args.name}_{current_epoch}.pth")
            torch.save(model.state_dict(), Path(args.output_dir) / f"{args.name}.pth")

        if current_epoch % VAL_FREQ == 0:
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
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.8, help="exponential weighting")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    main(args)
