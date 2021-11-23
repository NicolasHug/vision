import datetime
import os
import time
from collections import defaultdict
from collections import deque

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None, tb_val="avg"):
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
        t = reduce_across_processes([self.count, self.total])
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
                median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
            )
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
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        return self.delimiter.join(
            ["{}: {}".format(name, str(meter)) for name, meter in self.meters.items() if name not in self.dont_print]
        )

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, **kwargs):
        if not kwargs.pop("print", True):
            self.dont_print.add(name)
        self.meters[name] = SmoothedValue(window_size=self.freq, **kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def log(self, iterable, header="", sync=False, verbose=True):
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = self.delimiter.join(
            [
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "max mem: {memory:.0f}",
            ]
        )

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
                if verbose:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )

                for name, meter in self.meters.items():
                    self.tb_writer.add_scalar(f"{header} {name}", meter.get_tb_val(), self.current_step)
            i += 1
            self.current_step += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {}".format(header, total_time_str))

    def close(self):
        self.tb_writer.close()


def sequence_loss(flow_preds, flow_gt, valid_flow_mask, gamma=0.8, max_flow=400):
    """Loss function defined over sequence of flow predictions"""

    if gamma > 1:
        raise ValueError(f"Gamma should be < 1, got {gamma}.")

    # exlude invalid pixels and extremely large diplacements
    norm_2 = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid_flow_mask = valid_flow_mask & (norm_2 < max_flow)

    flow_loss = 0
    num_predictions = len(flow_preds)
    for i, flow_pred in enumerate(flow_preds):
        weight = gamma ** (num_predictions - i - 1)
        abs_diff = (flow_pred - flow_gt).abs()
        flow_loss += weight * (abs_diff * valid_flow_mask[:, None, :, :]).mean()

    last_pred = flow_preds[-1]
    epe = ((last_pred - flow_gt) ** 2).sum(dim=1).sqrt()
    epe = epe[valid_flow_mask]

    metrics = {
        "flow_loss": flow_loss,
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    # TODO: Ideally, this should be part of the eval transforms preset, instead
    # of being part of the validation code. It's not obvious what a good
    # solution would be, because we need to unpad the predicted flows according
    # to the input images' size, and in some datasets (Kitti) images can have
    # variable sizes.

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def _redefine_print(is_main):
    """disables printing when not in main process"""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup_ddp(args):
    # Set the local_rank, rank, and world_size values as args fields
    # This is done differently depending on how we're running the script. We
    # currently support either torchrun or the custom run_with_submitit.py
    # If you're confused (like I was), this might help a bit
    # https://discuss.pytorch.org/t/what-is-the-difference-between-rank-and-local-rank/61940/2

    if all(key in os.environ for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE")):
        # if we're here, the script was called with torchrun. Otherwise
        # these args will be set already by the run_with_submitit script
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    elif "gpu" in args:
        # if we're here, the script was called by run_with_submitit.py
        args.local_rank = args.gpu
    else:
        raise ValueError("Sorry, I can't set up the distributed training ¯\_(ツ)_/¯.")

    _redefine_print(is_main=(args.rank == 0))

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=args.rank,
        world_size=args.world_size,
        init_method=args.dist_url,
    )


def reduce_across_processes(val):
    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t

def freeze_batch_norm(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
