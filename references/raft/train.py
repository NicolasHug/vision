import argparse
import warnings
from pathlib import Path

import torch
from presets import OpticalFlowPresetTrain, OpticalFlowPresetEval
from torchvision.datasets import KittiFlow, FlyingChairs, FlyingThings3D, Sintel, HD1K
from torchvision.models.video import raft
from utils import MetricLogger, setup_ddp, sequence_loss, InputPadder, reduce_across_processes, freeze_batch_norm


# TODO: remove
DATASET_ROOT = "/data/home/nicolashug/cluster/work/downloads"


def get_train_dataset(stage):
    if stage == "chairs":
        transforms = OpticalFlowPresetTrain(crop_size=(368, 496), min_scale=0.1, max_scale=1.0, do_flip=True)
        return FlyingChairs(root=DATASET_ROOT, split="train", transforms=transforms)
    elif stage == "things":
        transforms = OpticalFlowPresetTrain(crop_size=(400, 720), min_scale=-0.4, max_scale=0.8, do_flip=True)
        return FlyingThings3D(root=DATASET_ROOT, split="train", pass_name="both", transforms=transforms)
    elif stage == "sintel":
        transforms = OpticalFlowPresetTrain(crop_size=(368, 768), min_scale=-0.2, max_scale=0.6, do_flip=True)
        things_clean = FlyingThings3D(root=DATASET_ROOT, split="train", pass_name="clean", transforms=transforms)
        sintel = Sintel(root=DATASET_ROOT, split="train", pass_name="both", transforms=transforms)
        return 100 * sintel + things_clean
        # TODO: use distributed weighted sampler

    elif stage == "sintel_skh":  # S + K + H as from paper
        crop_size = (368, 768)
        transforms = OpticalFlowPresetTrain(crop_size=crop_size, min_scale=-0.2, max_scale=0.6, do_flip=True)

        things_clean = FlyingThings3D(root=DATASET_ROOT, split="train", pass_name="clean", transforms=transforms)
        sintel = Sintel(root=DATASET_ROOT, split="train", pass_name="both", transforms=transforms)

        kitti_transforms = OpticalFlowPresetTrain(crop_size=crop_size, min_scale=-0.3, max_scale=0.5, do_flip=True)
        kitti = KittiFlow(root=DATASET_ROOT, split="train", transforms=kitti_transforms)

        hd1k_transforms = OpticalFlowPresetTrain(crop_size=crop_size, min_scale=-0.5, max_scale=0.2, do_flip=True)
        hd1k = HD1K(root=DATASET_ROOT, split="train", transforms=hd1k_transforms)

        return 100 * sintel + 200 * kitti + 5 * hd1k + things_clean
    elif stage == "kitti":
        transforms = OpticalFlowPresetTrain(
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
        )
        return KittiFlow(root=DATASET_ROOT, split="train", transforms=transforms)
    elif stage == "hd1k":
        transforms = OpticalFlowPresetTrain(crop_size=(368, 768), min_scale=-0.5, max_scale=0.2, do_flip=True)
        hd1k = HD1K(root=DATASET_ROOT, split="train", transforms=transforms)
        return hd1k
    else:
        raise ValueError(f"Unknown stage {stage}")


@torch.no_grad()
def validate_sintel(model, args):
    model.eval()

    for pass_name in ("clean", "final"):
        logger = MetricLogger(output_dir=args.output_dir)
        logger.add_meter("epe", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")
        logger.add_meter("1px", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")
        logger.add_meter("3px", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")
        logger.add_meter("5px", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")

        val_dataset = Sintel(root=DATASET_ROOT, split="train", pass_name=pass_name, transforms=OpticalFlowPresetEval())
        sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        header = f"Sintel val {pass_name}"

        def inner_loop(image1, image2, flow_gt):
            image1, image2 = image1.cuda(), image2.cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_predictions = model(image1, image2, num_flow_updates=32)
            flow_pred = flow_predictions[-1]

            flow_pred = padder.unpad(flow_pred).cpu()

            epe = ((flow_pred - flow_gt) ** 2).sum(dim=1).sqrt()

            logger.meters["epe"].update(epe.mean().item(), n=epe.numel())
            for distance in (1, 3, 5):
                logger.meters[f"{distance}px"].update((epe < distance).float().mean().item(), n=epe.numel())

        num_processed_samples = 0
        for blob in logger.log(val_loader, header=header, sync=False, verbose=False):
            image1, image2, flow_gt = blob
            inner_loop(image1, image2, flow_gt)
            num_processed_samples += image1.shape[0]

        num_processed_samples = reduce_across_processes(num_processed_samples)
        print(f"Evaluated {num_processed_samples} / {len(val_dataset)} samples in batch")

        if args.rank == 0:
            for i in range(num_processed_samples, len(val_dataset)):
                image1, image2, flow_gt = val_dataset[i]
                inner_loop(image1[None, :, :, :], image2[None, :, :, :], flow_gt[None, :, :, :])

        logger.synchronize_between_processes()
        print(header, logger)


@torch.no_grad()
def validate_kitti(model, args):
    """Peform validation using the KITTI-2015 (train) split"""

    if args.batch_size != 1 and args.rank == 0:
        warnings.warn(
            f"Batch-size={args.batch_size} was passed. For technical reasons, evaluating on Kitti can only be done with a batch-size of 1."
        )

    model.eval()
    val_dataset = KittiFlow(root=DATASET_ROOT, split="train", transforms=OpticalFlowPresetEval())
    sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=sampler,
        batch_size=1,  # Kitti has different image sizes so we need to individually pad them, see comment in InputPadder
        pin_memory=True,
        num_workers=args.num_workers,
    )

    logger = MetricLogger(output_dir=args.output_dir)
    logger.add_meter("per_image_epe", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")
    logger.add_meter("f1", fmt="{global_avg:.4f} ({value:.4f})", tb_val="global_avg")
    header = "Kitti val"

    for blob in logger.log(val_loader, header=header, sync=False, verbose=False):
        image1, image2, flow_gt, valid_flow_mask = blob
        image1 = image1.cuda()
        image2 = image2.cuda()

        padder = InputPadder(image1.shape, mode="kitti")
        image1, image2 = padder.pad(image1, image2)

        predictions = model(image1, image2, num_flow_updates=24)
        flow = predictions[-1]
        flow = padder.unpad(flow).cpu()

        epe = ((flow - flow_gt) ** 2).sum(dim=1).sqrt()
        relative_epe = epe / (flow_gt ** 2).sum(dim=1).sqrt()

        epe, relative_epe = epe[valid_flow_mask], relative_epe[valid_flow_mask]
        bad_predictions = ((epe > 3) & (relative_epe > 0.05)).float()

        # note the n=1 for per_image_epe: we compute an average over averages. We first average within each image and
        # then average over the images. This is in contrast with the other epe computations of e.g. Sintel, where we
        # average only once over all the pixels of all images.
        logger.meters["per_image_epe"].update(epe.mean().item(), n=1)  # f1-epe in paper
        logger.meters["f1"].update(bad_predictions.mean().item(), n=bad_predictions.numel())  # f1-all in paper

    logger.synchronize_between_processes()
    print(header, logger)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate(model, args):
    val_datasets = args.val_dataset or []
    for val_dataset in val_datasets:
        try:
            {"sintel": validate_sintel, "kitti": validate_kitti}[val_dataset](model, args)
        except KeyError as ke:
            raise ValueError(f"Can't validate on {val_dataset}") from ke


def main(args):
    setup_ddp(args)

    print(args)

    model = raft()
    model = model.to(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"), strict=False)

    torch.backends.cudnn.benchmark = True

    print("Parameter Count: %d" % count_parameters(model))

    if args.train_dataset is None:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        validate(model, args)
        return

    model.train()
    if args.freeze_batch_norm:
        freeze_batch_norm(model.module)

    train_dataset = get_train_dataset(args.train_dataset)

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        # worker_init_fn=lambda x:print(f"I'm rank {args.rank} and my worker info for data loading is {torch.utils.data.get_worker_info()}", force=True)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adamw_eps)

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

            optimizer.zero_grad()

            image1, image2, flow, valid_flow_mask = (x.cuda() for x in data_blob)
            flow_predictions = model(image1, image2, num_flow_updates=args.num_flow_updates)

            loss, metrics = sequence_loss(flow_predictions, flow, valid_flow_mask, args.gamma)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()
            scheduler.step()

            metrics["epoch"] = current_epoch
            metrics["current_step"] = current_step
            metrics["last_lr"] = scheduler.get_last_lr()[0]
            metrics["lr"] = args.lr
            metrics["wdecay"] = args.weight_decay
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
            validate(model, args)
            model.train()
            if args.freeze_batch_norm:
                freeze_batch_norm(model.module)

    logger.close()


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help, description="Train or evaluate an optical-flow model.")
    parser.add_argument(
        "--name",
        default="raft",
        type=str,
        help="The name of the experiment - determines the name of the files where weights are saved.",
    )
    parser.add_argument(
        "--output-dir", default="checkpoints", type=str, help="Output dir where checkpoints will be stored."
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="A path to previously saved weights. Used to re-start training from, or evaluate a pre-saved model.",
    )

    parser.add_argument("--num-workers", type=int, default=12, help="Number of workers for the data loading part.")

    parser.add_argument(
        "--train-dataset",
        type=str,
        help="The dataset to use for training. If not passed, only validation is performed (and you probably want to pass --resume).",
    )
    parser.add_argument("--val-dataset", type=str, nargs="+", help="The dataset(s) to use for validation.")
    parser.add_argument("--val-freq", type=int, default=2, help="Validate every X epochs")
    parser.add_argument("--num-epochs", type=int, default=500, help="The maximum number of epochs")
    parser.add_argument("--num-steps", type=int, default=100000, help="The maximum number of steps")
    parser.add_argument("--batch-size", type=int, default=6)

    parser.add_argument("--lr", type=float, default=0.00002, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.00005, help="Weight decay for AdamW optimizer")
    parser.add_argument("--adamw-eps", type=float, default=1e-8, help="eps value for AdamW optimizer")

    parser.add_argument("--freeze-batch-norm", action="store_true", help="Set BatchNorm modules of the model in eval mode.")

    parser.add_argument(
        "--num_flow_updates",
        type=int,
        default=12,
        help="number of updates (or 'iters') in the 'update operator' of the model.",
    )

    parser.add_argument("--gamma", type=float, default=0.8, help="exponential weighting for loss. Must be < 1.")

    parser.add_argument("--dist-url", default="env://", help="URL used to set up distributed training")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(exist_ok=True)
    main(args)
