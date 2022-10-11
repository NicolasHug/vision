import io
import pickle
import warnings
from pathlib import Path
from typing import List

import numpy as np

import torch
import torchvision.transforms as T
import webdataset as wds

from common import args, DATASET_SIZE
from ffcv.fields.basics import IntDecoder
from ffcv.fields.decoders import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader as FFCVLoader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import NormalizeImage, RandomHorizontalFlip, ToTensor, ToTorchImage
from torch.utils import data
from torchdata.dataloader2 import adapter, DataLoader2, PrototypeMultiProcessingReadingService
from torchdata.datapipes.iter import FileLister, FileOpener, Header, IterDataPipe, TarArchiveLoader
from torchvision.datasets import ImageFolder


# TODO: maybe infinite buffer can / is already natively supported by torchdata?
INFINITE_BUFFER_SIZE = 1_000_000_000


class LenSetter(IterDataPipe):  # TODO: Not sure how much this is needed
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        yield from self.dp

    def __len__(self):
        return args.limit or DATASET_SIZE


class ArchiveLoader(IterDataPipe):
    def __init__(self, source_datapipe, loader):
        self.loader = pickle.load if loader == "pickle" else torch.load
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for filename in self.source_datapipe:
            with open(filename, "rb") as f:
                yield self.loader(f)


class ConcaterIterable(IterDataPipe):
    # TODO: This should probably be a built-in: https://github.com/pytorch/data/issues/648
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for iterable in self.source_datapipe:
            yield from iterable


def _make_dp_from_image_folder(*, root):
    dp = FileLister(str(root), recursive=True, masks=["*.JPEG"])
    if args.limit:
        dp = Header(dp, limit=args.limit)
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE).sharding_filter()
    return dp


def _drop_label(data):
    img_data, label = data
    return img_data


def _make_dp_from_archive(*, root, archive, archive_content, num_archives=None):
    ext = "pt" if archive == "torch" else "pkl"
    dp = FileLister(str(root), masks=[f"archive_{args.archive_size}*{archive_content}*.{ext}"])
    if num_archives is not None:
        dp = Header(dp, limit=num_archives)
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE)  # inter-archive shuffling
    dp = ArchiveLoader(dp, loader=archive)
    dp = ConcaterIterable(dp)
    dp = dp.map(_drop_label)
    dp = dp.shuffle(buffer_size=args.archive_size)  # intra-archive shuffling

    # TODO: we're sharding here but the big BytesIO or Tensors have already been
    # loaded by all workers, possibly in vain. Hopefully the new experimental MP
    # reading service will improve this?
    dp = dp.sharding_filter()

    return dp


def _read_tar_entry(data):
    _, io_stream = data
    return io_stream.read()


def _make_dp_from_tars(*, root, num_archives=None):

    dp = FileLister(str(root), masks=[f"archive_{args.archive_size}*.tar"])
    if num_archives is not None:
        dp = Header(dp, limit=num_archives)
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE)  # inter-archive shuffling
    dp = FileOpener(dp, mode="b")
    dp = TarArchiveLoader(dp, mode="r:")
    dp = dp.shuffle(buffer_size=args.archive_size)  # intra-archive shuffling
    dp = dp.sharding_filter()

    dp = dp.map(_read_tar_entry)
    return dp


def _make_webdataset(*, root, num_archives):
    archives = Path(root).glob(f"archive_{args.archive_size}*.tar")
    archives = [str(a) for a in archives]
    if num_archives is not None:
        archives = archives[:num_archives]
    return wds.WebDataset(archives)  # This will read and load the data as bytes


def make_dp(*, root, archive=None, archive_content=None):
    if args.limit is not None:
        num_archives = args.limit // args.archive_size
        if args.limit % args.archive_size != 0:
            warnings.warn(
                f"Requested limit={args.limit} samples but archive size is {args.archive_size}. "
                f"You'll get {num_archives} samples."
            )
    else:
        num_archives = None

    if archive in ("pickle", "torch"):
        dp = _make_dp_from_archive(
            root=root,
            archive=archive,
            archive_content=archive_content,
            num_archives=num_archives,
        )
    elif archive == "tar":
        dp = _make_dp_from_tars(root=root, num_archives=num_archives)
    else:
        dp = _make_dp_from_image_folder(root=root)

    dp = LenSetter(dp)
    return dp


def make_webdataset(*, root):
    # Don't use this without `with_DL`, because the intra-archive shuffling only happens there
    archives = Path(root).glob(f"archive_{args.archive_size}*.tar")
    archives = [str(a) for a in archives]
    return (
        wds.WebDataset(archives)
        .shuffle(len(archives), initial=len(archives))  # inter-archive shuffling
        .map(lambda sample: io.BytesIO(sample["jpeg"]))
    )


def make_mapstyle(*argz, **kwargs):
    mapstyle = ImageFolder(*argz, **kwargs)
    if args.limit is not None:
        # Note: all files are still `ls`ed above, even if we discard some here
        mapstyle.samples = mapstyle.samples[: args.limit]
        mapstyle.targets = mapstyle.targets[: args.limit]
        mapstyle.imgs = mapstyle.imgs[: args.limit]

    return mapstyle


def make_ffcv_dataloader(*, root, transforms, encoded):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    hflip_prob = 0.5
    crop_size = 224
    if transforms == "vision":
        img_pipeline: List[Operation] = [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToTorchImage(),
            T.RandomResizedCrop(crop_size, antialias=True),
            T.RandomHorizontalFlip(hflip_prob),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    elif transforms:
        IMAGENET_MEAN = np.array(IMAGENET_MEAN) * 255
        IMAGENET_STD = np.array(IMAGENET_STD) * 255
        img_pipeline = [
            RandomResizedCropRGBImageDecoder(
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
                output_size=(crop_size, crop_size),
            ),
            RandomHorizontalFlip(hflip_prob),
            NormalizeImage(  # Note: in original FFCV example, this is done on GPU
                IMAGENET_MEAN, IMAGENET_STD, np.float32
            ),
            ToTensor(),
            ToTorchImage(),
        ]
    else:
        if args.tiny:
            decoder = SimpleRGBImageDecoder()
        else:
            # See https://github.com/libffcv/ffcv-imagenet/blob/f134cbfff7f590954edc5c24275444b7dd2f57f6/train_imagenet.py#L265
            decoder = CenterCropRGBImageDecoder(output_size=(224, 224), ratio=224 / 256)
        img_pipeline = [decoder]

    # Note: args.limit is taken care of in iterate_over_epoch()
    return FFCVLoader(
        f"{root}/{'ffcv' if encoded else 'ffcv_decoded'}.beton",
        # Note: most of FFCV //ism is batch-wise, so when setting num_workers > 1
        # it's best to also set higher batch-size (>= num_workers)
        batch_size=16 if args.num_workers > 0 else 1,
        drop_last=False,
        num_workers=max(args.num_workers, 1),
        os_cache=False,  # Because otherwise the entire dataset is stored in RAM
        order=OrderOption.QUASI_RANDOM,
        pipelines={
            "img": img_pipeline,
            "label": [IntDecoder()],
        },
        batches_ahead=2,  # Same default as prefetch_factor from DataLoader
    )


# TODO: For now, this should be used with this branch of TorchData (https://github.com/pytorch/data/pull/815)
def post_adapter_fn(dp, n_prefetch_total=20):
    """
    Prefetching buffer at the end of DataLoader2
    """
    return dp.prefetch(n_prefetch_total)


def with_DL(obj, dl="default"):
    # Wrap obj in a data-loader iff --num-workers > 0
    # Also enables shuffling for some datasets when it can only be done properly here

    if args.num_workers == 0:
        if isinstance(obj, wds.WebDataset):
            obj = obj.shuffle(args.archive_size, initial=args.archive_size)  # intra-archive shuffling
        return obj

    batch_size = 16 if args.num_workers > 0 else 1

    if isinstance(obj, torch.utils.data.datapipes.datapipe.IterDataPipe):
        if dl.lower() in ("default", "v2"):
            obj = obj.batch(batch_size=batch_size)
            n_prefetch_worker = 10  # Prefetching at each worker level
            return DataLoader2(
                obj.prefetch(n_prefetch_worker),
                datapipe_adapter_fn=adapter.Shuffle(),
                reading_service=PrototypeMultiProcessingReadingService(num_workers=args.num_workers,
                                                                       post_adapter_fn=post_adapter_fn),
            )
        elif dl.lower() == "v1":
            return data.DataLoader(
                obj, batch_size=batch_size, collate_fn=lambda x: x, num_workers=args.num_workers, shuffle=True
            )
        else:
            raise ValueError("Bad dl, got {dl}")
    elif isinstance(obj, ImageFolder):
        return data.DataLoader(
            obj, batch_size=batch_size, collate_fn=lambda x: x, num_workers=args.num_workers, shuffle=True
        )
    elif isinstance(obj, wds.WebDataset):
        return (
            wds.WebLoader(
                obj,
                batch_size=None,
                num_workers=args.num_workers,
            )
            .shuffle(args.archive_size, initial=args.archive_size)  # intra-archive shuffling
            .batched(
                batchsize=batch_size,
                collation_fn=lambda batch: batch,
            )
        )

    raise ValueError("You shouldn't be here")
