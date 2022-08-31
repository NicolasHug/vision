
import torchvision
from dataset_helpers import make_dp, make_ffcv_dataloader
from PIL import Image
from torchvision.io import read_file

from bench_data_reading import pickle_bytesio_dp, torch_bytesio_dp
from common import ARCHIVE_ROOT, JPEG_FILES_ROOT, bench, bytesio_to_tensor, decode, iterate_one_epoch
from dataset_helpers import make_dp, make_ffcv_dataloader
from bench_transforms import ClassificationPresetTrain
from bench_data_reading_decoding import pickle_decoded_dp, torch_decoded_dp


ffcv_encoded_transforms = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=True, encoded=True)
ffcv_decoded_transforms = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=True, encoded=False)
ffcv_encoded_transforms_torchvision = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms="vision", encoded=True)
ffcv_decoded_transforms_torchvision = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms="vision", encoded=False)


def tensor_loader(path):
    return decode(read_file(path))


def pil_loader(path):
    return Image.open(path).convert("RGB")


mapstyle_ds_transforms_pil = torchvision.datasets.ImageFolder(
    JPEG_FILES_ROOT, transform=ClassificationPresetTrain(on="pil")
)
mapstyle_ds_transforms_tensor = torchvision.datasets.ImageFolder(
    JPEG_FILES_ROOT, loader=tensor_loader, transform=ClassificationPresetTrain(on="tensor")
)
no_archive_dp_transforms_pil = (
    make_dp(root=JPEG_FILES_ROOT, archive=None).map(pil_loader).map(ClassificationPresetTrain(on="pil"))
)
no_archive_dp_transforms_tensor = (
    make_dp(root=JPEG_FILES_ROOT, archive=None).map(tensor_loader).map(ClassificationPresetTrain(on="tensor"))
)

print("pickle bytesio->ToTensor()->decode_jpeg()->Transforms()")
bench(iterate_one_epoch, inp=pickle_bytesio_dp.map(bytesio_to_tensor).map(decode).map(ClassificationPresetTrain(on="tensor")), unit="m")

print("torch bytesio->ToTensor()->decode_jpeg()->Transforms()")
bench( iterate_one_epoch, inp=torch_bytesio_dp.map(bytesio_to_tensor).map(decode).map(ClassificationPresetTrain(on="tensor")), unit="m")

print("FFCV loading + decoding + transforms")
bench(iterate_one_epoch, inp=ffcv_encoded_transforms, unit="m")

print("File-based Mapstyle -> Image.open() -> PIL transforms")  # a.k.a the current torchvision transforms
bench(iterate_one_epoch, inp=mapstyle_ds_transforms_pil, unit="m")

print("File-based Mapstyle -> decode_jpeg() -> Transforms")
bench(iterate_one_epoch, inp=mapstyle_ds_transforms_tensor, unit="m")

print("File-based DP -> Image.open() -> PIL transforms")  # a.k.a the current torchvision transforms
bench( iterate_one_epoch, inp=no_archive_dp_transforms_pil, unit="m")

print("File-based DP -> decode_jpeg() -> Transforms")
bench( iterate_one_epoch, inp=no_archive_dp_transforms_tensor, unit="m")

print("pickle pre-decoded -> Transforms()")
bench( iterate_one_epoch, inp=pickle_decoded_dp.map(ClassificationPresetTrain(on="tensor")), unit="m")

print("torch pre-decoded -> Transforms()")
bench(iterate_one_epoch, inp=torch_decoded_dp.map(ClassificationPresetTrain(on="tensor")), unit="m")

print("FFCV loading (pre-decoded) + transforms")
bench(iterate_one_epoch, inp=ffcv_decoded_transforms, unit="m")
