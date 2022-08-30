import pickle
from pathlib import Path
from typing import List

import numpy as np

import torch
import torchvision
import torchvision.transforms as T
import webdataset as wds
from ffcv.fields.basics import IntDecoder
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader as FFCVLoader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import NormalizeImage, RandomHorizontalFlip, ToTensor, ToTorchImage
from torchdata.datapipes.iter import FileLister, FileOpener, IterDataPipe, TarArchiveLoader


# TODO: maybe infinite buffer can / is already natively supported by torchdata?
INFINITE_BUFFER_SIZE = 1_000_000_000


class _TinyImageNetLenSetter(IterDataPipe):
    def __init__(self, dp, root):
        self.dp = dp

        if "train" in str(root):
            self.size = 100_000
        elif "val" in str(root):
            self.size = 10_000
        else:
            raise ValueError("oops?")

    def __iter__(self):
        yield from self.dp

    def __len__(self):
        return self.size


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
    dp = FileLister(root, recursive=True, masks=["*.JPEG"])
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE).set_shuffle(False).sharding_filter()
    return dp


def _drop_label(data):
    img_data, label = data
    return img_data


def _make_dp_from_archive(*, root, archive, archive_content, archive_size):
    ext = "pt" if archive == "torch" else "pkl"
    dp = FileLister(str(root), masks=[f"archive_{archive_size}*{archive_content}*.{ext}"])
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE).set_shuffle(False)  # inter-archive shuffling
    dp = ArchiveLoader(dp, loader=archive)
    dp = ConcaterIterable(dp)
    dp = dp.map(_drop_label)
    dp = dp.shuffle(buffer_size=archive_size).set_shuffle(False)  # intra-archive shuffling

    # TODO: we're sharding here but the big BytesIO or Tensors have already been
    # loaded by all workers, possibly in vain. Hopefully the new experimental MP
    # reading service will improve this?
    dp = dp.sharding_filter()

    return dp


def _read_tar_entry(data):  # TODO: remove this
    _, io_stream = data
    return io_stream.read()


def _make_dp_from_tars(*, root, archive_size):

    dp = FileLister(str(root), masks=[f"archive_{archive_size}*.tar"])
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE).set_shuffle(False)  # inter-archive shuffling
    dp = FileOpener(dp, mode="b")
    dp = TarArchiveLoader(dp)
    dp = dp.shuffle(buffer_size=archive_size).set_shuffle(False)  # intra-archive shuffling
    dp = dp.sharding_filter()

    dp = dp.map(_read_tar_entry)
    return dp


def _make_webdataset(*, root, archive_size):
    archives = Path(root).glob(f"archive_{archive_size}*.tar")
    archives = [str(a) for a in archives]
    return wds.WebDataset(archives)  # This will read and load the data as bytes


def make_dp(*, root, archive=None, archive_content=None, archive_size=500):
    if archive in ("pickle", "torch"):
        dp = _make_dp_from_archive(
            root=root, archive=archive, archive_content=archive_content, archive_size=archive_size
        )
    elif archive == "tar":
        dp = _make_dp_from_tars(root=root, archive_size=archive_size)
    elif archive == "webdataset":
        dp = _make_webdataset(root=root, archive_size=archive_size)
    else:
        dp = _make_dp_from_image_folder(root=root)

    dp = _TinyImageNetLenSetter(dp, root=root)
    return dp


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
        img_pipeline = [SimpleRGBImageDecoder()]  # still have to decode

    return FFCVLoader(
        f"{root}/{'ffcv' if encoded else 'ffcv_decoded'}.beton",
        batch_size=1,
        drop_last=False,
        # Note: most of FFCV //ism is batch-wise, so when setting num_workers > 1
        # it's worth also setting a higher batch-size (>= num_workers)
        # For fairer comparison, maybe also set the same batch-size in
        # DataLoader[2] when doing this?
        num_workers=1,
        os_cache=False,  # Because otherwise the entire dataset is stored in RAM
        order=OrderOption.QUASI_RANDOM,
        pipelines={
            "img": img_pipeline,
            "label": [IntDecoder()],
        },
        batches_ahead=2,  # Same default as prefetch_factor from DataLoader
    )
