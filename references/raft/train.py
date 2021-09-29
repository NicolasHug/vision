import argparse
import os
import os.path as osp
from pathlib import Path
from collections import defaultdict
import time
from collections import deque
import datetime

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


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None, tb_val='avg'):
        if fmt is None:
            fmt = "{" + tb_val + ":.4f}"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.tb_val = tb_val

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    
    def get_tb_val(self):
        # tells tensorboard what it should register
        # For some values we want the running avg, for some we just want the last value
        return getattr(self, self.tb_val)

    @property
    def median(self):
        if self.deque:
            d = torch.tensor(list(self.deque))
            return d.median().item()
        else:
            return None

    @property
    def avg(self):
        if self.deque:
            d = torch.tensor(list(self.deque), dtype=torch.float32)
            return d.mean().item()
        else:
            return None

    @property
    def global_avg(self):
        if self.count != 0:
            return self.total / self.count
        else:
            return None

    @property
    def max(self):
        if self.deque:
            return max(self.deque)
        else:
            return None

    @property
    def value(self):
        if self.deque:
            return self.deque[-1]
        else:
            return None

    def __str__(self):
        if self.deque:
            return self.fmt.format(
                median=self.median,
                avg=self.avg,
                global_avg=self.global_avg,
                max=self.max,
                value=self.value)
        else:
            return str(None)


class MetricLogger(object):
    def __init__(self, freq=5, output_dir=None, delimiter="  "):
        # Note: passing freq in init instead of log() to keep the printing
        # frequency and the window_size equal. Might revisit.
        self.meters = defaultdict(lambda: SmoothedValue(window_size=freq))
        self.freq = freq
        self.delimiter = delimiter
        self.tb_writer = SummaryWriter(log_dir=output_dir)
        self.dont_print = set()
        self.current_step = 0

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        return self.delimiter.join([
                "{}: {}".format(name, str(meter))
                for name, meter in self.meters.items()
                if name not in self.dont_print

        ])

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, **kwargs):
        if not kwargs.pop('print', True):
            self.dont_print.add(name)
        self.meters[name] = SmoothedValue(window_size=self.freq, **kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def log(self, iterable, header='', sync=False):
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])

        MB = 1024.0 * 1024.0
        i = 0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % self.freq == 0:
                if sync:
                    self.synchronize_between_processes()
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
                
                for name, meter in self.meters.items():
                    self.tb_writer.add_scalar(f"{header} {name}", meter.get_tb_val(), self.current_step)
            i += 1
            self.current_step += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))

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
    # FIXME: this isn't 100% accurate because some samples will be duplicated if
    # the dataset isn't divisible by the batch size.

    model.eval()
    for dstype in ['clean', 'final']:
        logger = MetricLogger(output_dir=args.output_dir)
        logger.add_meter('flow_loss', fmt="{global_avg:.4f} ({value:.4f})")
        logger.add_meter('1px', fmt="{global_avg:.4f} ({value:.4f})")
        logger.add_meter('3px', fmt="{global_avg:.4f} ({value:.4f})")
        logger.add_meter('5px', fmt="{global_avg:.4f} ({value:.4f})")

        val_dataset = Sintel(split='training', dstype=dstype)
        sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=sampler,
            batch_size=args.batch_size, 
            pin_memory=True,
            num_workers=args.num_workers,
        )
        header = f'Sintel val {dstype}'
        for blob in logger.log(val_loader, header=header, sync=False):

            image1, image2, flow_gt, _ = blob
            image1, image2 = image1.cuda(), image2.cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
            
            logger.meters['epe'].update(epe.mean(), n=epe.numel())
            for distance in (1, 3, 5):
                logger.meters[f'px{distance}'].update((epe < distance).float().mean(), n=epe.numel())

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
        model.load_state_dict(torch.load(args.resume, map_location='cpu'), strict=False)

    # TODO: maybe, maybe not:
    torch.backends.cudnn.benchmark = True

    print("Parameter Count: %d" % count_parameters(model))

    # TODO: This looks important
    if args.train_dataset != 'chairs':
        model.module.freeze_bn()
    
    if args.train_dataset is None:
        # just validate then
        if args.val_dataset == ['sintel']:
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
    logger = MetricLogger(output_dir=args.output_dir)
    logger.add_meter('epoch', tb_val='value', print=False)
    logger.add_meter('current_step', tb_val='value', print=False)
    logger.add_meter('lr', tb_val='value', print=False)
    logger.add_meter('wdecay', tb_val='value', print=False)
    logger.add_meter('last_lr', tb_val='value')
    logger.add_meter('flow_loss', tb_val='value')
    logger.add_meter('1px', tb_val='value')
    logger.add_meter('3px', tb_val='value')
    logger.add_meter('5px', tb_val='value')

    VAL_FREQ = 5  # validate every X epochs

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

            metrics['epoch'] = current_epoch
            metrics['current_step'] = current_step
            metrics['last_lr'] = scheduler.get_last_lr()[0]
            metrics['lr'] = args.lr
            metrics['wdecay'] = args.wdecay
            logger.update(**metrics)

            current_step += 1

            if args.num_steps is not None and current_step == args.num_steps:
                done = True
                break

        logger.synchronize_between_processes()
        print(f'Epoch {current_epoch} done. ', logger, force=True)

        current_epoch += 1
        if args.num_epochs is not None and current_epoch == args.num_epochs:
            done = True

        if args.rank == 0:
            torch.save(model.state_dict(), Path(args.output_dir) / f'{args.name}_{current_epoch + 1}.pth')
            torch.save(model.state_dict(), Path(args.output_dir) / f'{args.name}.pth')

        if current_epoch % VAL_FREQ == 0:
            model.eval()

            val_datasets = args.val_dataset or []
            for val_dataset in val_datasets:
                # if val_dataset == 'chairs':
                #     results.update(evaluate.validate_chairs(model.module))
                if val_dataset == 'sintel':
                    validate_sintel(model, args)
                # elif val_dataset == 'kitti':
                #     results.update(evaluate.validate_kitti(model.module))

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


def setup_ddp(args):
    # Set the local_rank, rank, and world_size values as args fields
    # This is done differently depending on how we're running the script. We
    # currently support either torchrun or the custom run_with_submitit.py
    # If you're confused (like I was), this might help a bit
    # https://discuss.pytorch.org/t/what-is-the-difference-between-rank-and-local-rank/61940/2

    if all(key in os.environ for key in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE')):
        # if we're here, the script was called with torchhub. Otherwise
        # these args will be set already by the run_with_submitit script
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])

    elif 'gpu' in args:
        # if we're here, the script was called by run_with_submitit.py
        args.local_rank = args.gpu
    else:
        raise ValueError("Sorry, I can't set up the distributed training ¯\_(ツ)_/¯")

    redefine_print(args.rank == 0)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size, init_method=args.dist_url if "dist_url" in args else "env://")


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

    # TODO: make these mutually exclusive maybe?
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--num-steps', type=int, default=100000)

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

    if not osp.isdir(args.output_dir):
        # FIXME: not multiprocess-safe
        os.mkdir(args.output_dir)

    main(args)
