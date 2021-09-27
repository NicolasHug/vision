import argparse
import os
import os.path as osp
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import numpy as np

from torchvision.models.video import RAFT
from torchvision.models.video._raft.utils import KittiFlowDataset, FlyingChairs, FlyingThings3D, Sintel, InputPadder


MAX_FLOW = 400


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'flow_loss': flow_loss,
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


class Logger:

    def __init__(self, output_dir=None, freq=5):
        self.writer = SummaryWriter(log_dir=output_dir)
        self.freq = freq  # will log average metric values every "freq" calls to push()
        self._reset_metrics()

    def _reset_metrics(self):
        self.running_metrics = defaultdict(float)

    def push(self, metrics):
        # this is
        # U
        # G
        # L
        # Y
        # :)

        non_averaged_metrics = ('current_step', 'epoch', 'last_lr', 'lr', 'wdecay')
        current_step, epoch, last_lr = (metrics [key] for key in ('current_step', 'epoch', 'last_lr'))

        for key in metrics:
            if key in non_averaged_metrics:
                self.running_metrics[key] = metrics[key]
            else:
                self.running_metrics[key] += metrics[key]

        if current_step % self.freq == 0:
            s = f"[{epoch:3d}, {current_step:6d}, last_lr: {last_lr:10.7f}] "
            for metric, val in self.running_metrics.items():
                if metric in non_averaged_metrics:
                    self.writer.add_scalar(metric, val, current_step)
                else:
                    avg_val = val / self.freq
                    self.writer.add_scalar(metric, avg_val, current_step)
                    s += f"{metric}: {avg_val:10.4f}, "
            print(s)

            self._reset_metrics()

    def close(self):
        self.writer.close()


def _get_train_dataset(dataset_name):
    d = {
        'kitti': KittiFlowDataset,
        'chairs': FlyingChairs,
        'things': FlyingThings3D,
        'sintel': Sintel
    }
    aug_params = {
        'kitti': {'crop_size': (288, 960), 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False},
        'chairs': {'crop_size': (368, 496), 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True},
        'things': {'crop_size': (400, 720), 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True},
        'sintel': {'crop_size': (368, 768), 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True},
    }

    dataset_name = dataset_name.lower()

    if dataset_name not in d:
        raise ValueError(f"Unknown dataset {dataset_name}")

    klass = d[dataset_name]
    return klass(aug_params=aug_params[dataset_name])


@torch.no_grad()
def validate_sintel(model, args, iters=32):
    """Peform validation using the Sintel (train) split """
    model.eval()
    print("IN MY VERSION LOLOLOLOL")
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = Sintel(split='training', dstype=dstype)
        val_dataset = torch.utils.data.DataLoader(
            val_dataset,
            sampler=torch.utils.data.distributed.DistributedSampler(val_dataset),
            batch_size=args.batch_size, 
            pin_memory=True,  # TODO: find out why it was False in raft repo?
            num_workers=args.num_workers,
        )

        epe_list = []

        for i, blob in enumerate(val_dataset):
            print(f'blob {i}')
            image1, image2, flow_gt = blob[:3]
            image1, image2 = image1.cuda(), image2.cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel_orig(model, args, iters=32):
    """ Peform validation using the Sintel (train) split """
    print("IN ORIG VERSION LOLOLOL")
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = Sintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            print(val_id)
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    if 'gpu' in args:
        args.local_rank = args.gpu
    print(args)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size, init_method=args.dist_url if "dist_url" in args else "env://")
    
    model = RAFT(args)  # TODO: pass better args
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location='cpu'), strict=False)

    model = model.to(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # TODO: maybe, maybe not:
    torch.backends.cudnn.benchmark = True

    print("Parameter Count: %d" % count_parameters(model))

    # TODO: This looks important
    if args.train_dataset != 'chairs':
        model.module.freeze_bn()
    
    if args.train_dataset is None:
        # just validate then
        if args.val_dataset == ['sintel']:
            validate_sintel_orig(model, args)
            validate_sintel(model, args)
        else:
            raise ValueError(f"can't validate on {args.val_dataset}")
        return

    model.train()

    train_dataset = _get_train_dataset(args.train_dataset)

    # TODO: Should drop_last really be True? And shouhld it be set in the loader instead of the sampler?
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True, drop_last=True)
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
        optimizer=optimizer, max_lr=args.lr, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear',
        **extra_scheduler_args
    )


    scaler = GradScaler(enabled=args.mixed_precision)  # TODO: currently untested
    logger = Logger(output_dir=args.output_dir)

    VAL_FREQ = 20  # validate every X epochs

    done = False
    current_epoch = current_step = 0
    while not done:
        sampler.set_epoch(current_epoch)  # set this, otherwise the data loading order would be the same for all epochs
        print(f"EPOCH {current_epoch}")

        for i_batch, data_blob in enumerate(train_loader):

            # TODO: set p.grad = None instead? see https://twitter.com/karpathy/status/1299921324333170689/photo/1
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.to(args.local_rank) for x in data_blob]

            # print(f"I'm rank {args.rank} and I'm getting {image1[0][0][20][100]} as a given pixel value, {image1.shape = }", force=True)

            # if args.add_noise:
            #     stdv = np.random.uniform(0.0, 5.0)
            #     image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
            #     image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if args.rank == 0:
                metrics['epoch'] = current_epoch
                metrics['current_step'] = current_step
                metrics['last_lr'] = scheduler.get_last_lr()[0]
                metrics['lr'] = args.lr
                metrics['wdecay'] = args.wdecay
                logger.push(metrics)

            current_step += 1

            if args.num_steps is not None and current_step == args.num_steps:
                done = True
                break

        current_epoch += 1
        if args.num_epochs is not None and current_epoch == args.num_epochs:
            done = True

        torch.save(model.state_dict(), Path(args.output_dir) / f'{args.name}_{current_epoch + 1}.pth')
        torch.save(model.state_dict(), Path(args.output_dir) / f'{args.name}.pth')
        if current_epoch % VAL_FREQ == 0:
            model.eval()

            results = {}
            val_datasets = args.val_dataset or []
            for val_dataset in val_datasets:
                # if val_dataset == 'chairs':
                #     results.update(evaluate.validate_chairs(model.module))
                if val_dataset == 'sintel':
                    results.update(validate_sintel(model, args))
                # elif val_dataset == 'kitti':
                #     results.update(evaluate.validate_kitti(model.module))

            # logger.write_dict(results)
            model.train()
            if args.train_dataset != 'chairs':
                model.module.freeze_bn()

    logger.close()


def redefine_print(is_main):
    """disables printing when not in main process"""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--train-dataset', help="determines which dataset to use for training") 
    parser.add_argument('--resume', help="restore checkpoint")
    parser.add_argument('--val-dataset', type=str, nargs='+')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--output-dir', default='checkpoints', type=str)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.00002)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--num-epochs', type=int, default=None)
    group.add_argument('--num-steps', type=int, default=None)

    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()

    if all(key in os.environ for key in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE')):
        # if we're here, the script was called with torchhub. Otherwise
        # these args will be set already by the run_with_submitit script
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])

    redefine_print(args.rank == 0)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not osp.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    n_gpus = torch.cuda.device_count()
    print(f"{n_gpus} cuda devices available")
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    main(args)
