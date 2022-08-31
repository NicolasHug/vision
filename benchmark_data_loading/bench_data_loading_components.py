import argparse
import datetime
import io
from pathlib import Path
from time import time

import numpy as np

import torch
import torchvision
from dataset_helpers import make_dp, make_ffcv_dataloader
from ffcv.loader import Loader as FFCVLoader
from PIL import Image
from torchvision.io import decode_jpeg, ImageReadMode, read_file


########################################
# DATA-READING + DECODING + TRANSFORMS #
########################################

# ffcv_encoded_transforms = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=True, encoded=True)
# ffcv_decoded_transforms = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=True, encoded=False)
# ffcv_encoded_transforms_torchvision = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms="vision", encoded=True)
# ffcv_decoded_transforms_torchvision = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms="vision", encoded=False)


# def tensor_loader(path):
#     return decode(read_file(path))


# def pil_loader(path):
#     return Image.open(path).convert("RGB")


# mapstyle_ds_transforms_pil = torchvision.datasets.ImageFolder(
#     JPEG_FILES_ROOT, transform=ClassificationPresetTrain(on="pil")
# )
# mapstyle_ds_transforms_tensor = torchvision.datasets.ImageFolder(
#     JPEG_FILES_ROOT, loader=tensor_loader, transform=ClassificationPresetTrain(on="tensor")
# )
# no_archive_dp_transforms_pil = (
#     make_dp(root=JPEG_FILES_ROOT, archive=None).map(pil_loader).map(ClassificationPresetTrain(on="pil"))
# )
# no_archive_dp_transforms_tensor = (
#     make_dp(root=JPEG_FILES_ROOT, archive=None).map(tensor_loader).map(ClassificationPresetTrain(on="tensor"))
# )

# print("pickle bytesio->ToTensor()->decode_jpeg()->Transforms()")
# bench(
#     iterate_one_epoch,
#     inp=pickle_bytesio_dp.map(bytesio_to_tensor).map(decode).map(ClassificationPresetTrain(on="tensor")),
#     warmup=1,
#     num_exp=5,
#     unit="m",
#     num_images_per_call=len(mapstyle_ds),
# )

# print("torch bytesio->ToTensor()->decode_jpeg()->Transforms()")
# bench(
#     iterate_one_epoch,
#     inp=torch_bytesio_dp.map(bytesio_to_tensor).map(decode).map(ClassificationPresetTrain(on="tensor")),
#     warmup=1,
#     num_exp=5,
#     unit="m",
#     num_images_per_call=len(mapstyle_ds),
# )

# print("FFCV loading + decoding + transforms")
# bench(
#     iterate_one_epoch, inp=ffcv_encoded_transforms, warmup=1, num_exp=5, unit="m", num_images_per_call=len(mapstyle_ds)
# )

# print("File-based Mapstyle -> Image.open() -> PIL transforms")  # a.k.a the current torchvision transforms
# bench(
#     iterate_one_epoch,
#     inp=mapstyle_ds_transforms_pil,
#     warmup=1,
#     num_exp=5,
#     unit="m",
#     num_images_per_call=len(mapstyle_ds),
# )

# print("File-based Mapstyle -> decode_jpeg() -> Transforms")
# bench(
#     iterate_one_epoch,
#     inp=mapstyle_ds_transforms_tensor,
#     warmup=1,
#     num_exp=5,
#     unit="m",
#     num_images_per_call=len(mapstyle_ds),
# )

# print("File-based DP -> Image.open() -> PIL transforms")  # a.k.a the current torchvision transforms
# bench(
#     iterate_one_epoch,
#     inp=no_archive_dp_transforms_pil,
#     warmup=1,
#     num_exp=5,
#     unit="m",
#     num_images_per_call=len(mapstyle_ds),
# )

# print("File-based DP -> decode_jpeg() -> Transforms")
# bench(
#     iterate_one_epoch,
#     inp=no_archive_dp_transforms_tensor,
#     warmup=1,
#     num_exp=5,
#     unit="m",
#     num_images_per_call=len(mapstyle_ds),
# )

# print("pickle pre-decoded -> Transforms()")
# bench(
#     iterate_one_epoch,
#     inp=pickle_decoded_dp.map(ClassificationPresetTrain(on="tensor")),
#     warmup=1,
#     num_exp=5,
#     unit="m",
#     num_images_per_call=len(mapstyle_ds),
# )

# print("torch pre-decoded -> Transforms()")
# bench(
#     iterate_one_epoch,
#     inp=torch_decoded_dp.map(ClassificationPresetTrain(on="tensor")),
#     warmup=1,
#     num_exp=5,
#     unit="m",
#     num_images_per_call=len(mapstyle_ds),
# )

# print("FFCV loading (pre-decoded) + transforms")
# bench(
#     iterate_one_epoch, inp=ffcv_decoded_transforms, warmup=1, num_exp=5, unit="m", num_images_per_call=len(mapstyle_ds)
# )
