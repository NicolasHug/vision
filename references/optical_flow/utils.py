import datetime
import os
import time
from collections import defaultdict
from collections import deque

import torch
import torch.distributed as dist
import torch.nn.functional as F


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt="{median:.4f} ({global_avg:.4f})"):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

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

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, **kwargs):
        self.meters[name] = SmoothedValue(**kwargs)

    def log_every(self, iterable, print_freq=5, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
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
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if print_freq is not None and i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
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
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


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


def map_orig_to_ours(orig, mine=None):
    # TODO: remove
    d = {}
    used_s_orig = set()
    used_s_mine = set()

    def assert_and_add(s_orig, s_mine):
        # print(s_orig, s_mine)
        # print(orig[s_orig].shape, mine[s_mine].shape)

        assert s_orig not in used_s_orig
        assert s_mine not in used_s_mine

        if mine is not None:
            assert s_mine in mine
        assert s_orig in orig
        if mine is not None:
            assert orig[s_orig].shape == mine[s_mine].shape
        d["module." + s_mine] = orig[s_orig]
        used_s_orig.add(s_orig)
        used_s_mine.add(s_mine)

    for encoder_orig, encoder_mine in (
        ("fnet", "feature_encoder"),
        ("cnet", "context_encoder"),
    ):
        for attr in ("bias", "weight"):
            s_orig = f"module.{encoder_orig}.conv1.{attr}"
            s_mine = f"{encoder_mine}.convnormrelu.0.{attr}"
            assert_and_add(s_orig, s_mine)

            s_orig = f"module.{encoder_orig}.conv2.{attr}"
            s_mine = f"{encoder_mine}.conv.{attr}"
            assert_and_add(s_orig, s_mine)

            for layer in (1, 2, 3):
                for block in (0, 1):
                    for conv in (1, 2):
                        s_orig = f"module.{encoder_orig}.layer{layer}.{block}.conv{conv}.{attr}"
                        s_mine = f"{encoder_mine}.layer{layer}.{block}.convnormrelu{conv}.0.{attr}"
                        assert_and_add(s_orig, s_mine)

            for layer in (2, 3):
                s_orig = f"module.{encoder_orig}.layer{layer}.0.downsample.0.{attr}"
                s_mine = f"{encoder_mine}.layer{layer}.0.downsample.0.{attr}"
                assert_and_add(s_orig, s_mine)

    encoder_orig, encoder_mine = "cnet", "context_encoder"
    for attr in (
        "bias",
        "weight",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ):
        s_orig = f"module.{encoder_orig}.norm1.{attr}"
        s_mine = f"{encoder_mine}.convnormrelu.1.{attr}"
        assert_and_add(s_orig, s_mine)
        for layer in (1, 2, 3):
            for block in (0, 1):
                for norm in (1, 2):
                    s_orig = f"module.{encoder_orig}.layer{layer}.{block}.norm{norm}.{attr}"
                    s_mine = f"{encoder_mine}.layer{layer}.{block}.convnormrelu{norm}.1.{attr}"
                    assert_and_add(s_orig, s_mine)
        for layer in (2, 3):
            s_orig = f"module.{encoder_orig}.layer{layer}.0.downsample.1.{attr}"
            s_mine = f"{encoder_mine}.layer{layer}.0.downsample.1.{attr}"
            assert_and_add(s_orig, s_mine)

    corr_orig, corr_mine = (
        "module.update_block.encoder.",
        "update_block.motion_encoder.",
    )
    for attr in ("bias", "weight"):
        for i in (1, 2):
            s_orig = f"{corr_orig}convc{i}.{attr}"
            s_mine = f"{corr_mine}convcorr{i}.0.{attr}"
            assert_and_add(s_orig, s_mine)
            s_orig = f"{corr_orig}convf{i}.{attr}"
            s_mine = f"{corr_mine}convflow{i}.0.{attr}"
            assert_and_add(s_orig, s_mine)
        s_orig = f"{corr_orig}conv.{attr}"
        s_mine = f"{corr_mine}conv.0.{attr}"
        assert_and_add(s_orig, s_mine)

    rec_orig, rec_mine = "module.update_block.gru", "update_block.recurrent_block"
    for attr in ("bias", "weight"):
        for i in (1, 2):
            for conv in ("convz", "convr", "convq"):
                s_orig = f"{rec_orig}.{conv}{i}.{attr}"
                s_mine = f"{rec_mine}.convgru{i}.{conv}.{attr}"
                assert_and_add(s_orig, s_mine)

    flow_orig, flow_mine = "module.update_block.flow_head", "update_block.flow_head"
    for attr in ("bias", "weight"):
        for i in (1, 2):
            s_orig = f"{flow_orig}.conv{i}.{attr}"
            s_mine = f"{flow_mine}.conv{i}.{attr}"
            assert_and_add(s_orig, s_mine)
    for s_orig, s_mine in zip(
        (
            "module.update_block.mask.0.weight",
            "module.update_block.mask.0.bias",
            "module.update_block.mask.2.weight",
            "module.update_block.mask.2.bias",
        ),
        (
            "mask_predictor.convrelu.0.weight",
            "mask_predictor.convrelu.0.bias",
            "mask_predictor.conv.weight",
            "mask_predictor.conv.bias",
        ),
    ):
        assert_and_add(s_orig, s_mine)

    if mine is not None:
        print(len(d), len(orig), len(mine))
        assert not (set(mine.keys()) - set(d.keys()))
    return d
