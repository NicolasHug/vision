import datetime
import io
from time import time
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from dataset_helpers import make_dp, make_ffcv_dataloader, make_mapstyle
from ffcv import libffcv
from ffcv.loader import Loader as FFCVLoader
from PIL import Image
from references.torchdata_benchmarks.presets import ClassificationPresetTrain
from torchvision.io import decode_jpeg, ImageReadMode, read_file
from distutils.dir_util import copy_tree
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fs", default="fsx_isolated")
args = parser.parse_args()


FS = "/" + args.fs
print(FS)

COMMON_ROOT = f"{FS}/nicolashug/tinyimagenet/081318/"
ARCHIVE_ROOT = COMMON_ROOT + "archives/train"
JPEG_FILES_ROOT = COMMON_ROOT + "train"

# Deactivate OMP / MKL parallelism: in most cases we'll run the data-loading
# pipeline within a parallelized DataLoader which will call this as well anyway.
torch.set_num_threads(1)


def bench(f, inp, num_exp=1000, warmup=10, unit="μ", num_images_per_call=1):
    def get_time_in_training(time_per_image, dataset_size="imagenet", num_epochs=10):
        if isinstance(dataset_size, str):
            dataset_size = {"tiny": 110_000, "imagenet": 1_281_167 + 50_000}[dataset_size]
        tot = time_per_image * dataset_size * num_epochs
        return datetime.timedelta(seconds=int(tot))

    # Computes PER IMAGE median times
    for _ in range(warmup):
        f(inp)

    times = []
    for _ in range(num_exp):
        start = time()
        f(inp)
        end = time()
        times.append((end - start))

    mul = {"μ": 1e6, "m": 1e3, "s": 1}[unit]
    times = torch.tensor(times) / num_images_per_call
    median_sec = torch.median(times)

    times_unit = times * mul
    median_unit = torch.median(times_unit)

    time_in_training = get_time_in_training(median_sec)
    s = f"{median_unit:.1f} {unit}{'s' if unit != 's' else ''}/img (std={torch.std(times_unit):.2f})"
    print(f"{s:30}   {int(1 / median_sec):15,}   {time_in_training}")
    print()
    return median_sec


def iterate_one_epoch(obj):
    if isinstance(obj, (torch.utils.data.datapipes.datapipe.IterDataPipe, FFCVLoader)):
        list(obj)
    else:
        # Need to reproduce "random" access
        indices = torch.randperm(len(obj))
        for i in indices:
            obj[i]


################
# DATA READING #
################


def bytesio_to_tensor(bytesio):
    return torch.frombuffer(bytesio.getbuffer(), dtype=torch.uint8)


mapstyle_ds = make_mapstyle(root=JPEG_FILES_ROOT)
no_archive_dp = make_dp(root=JPEG_FILES_ROOT, archive=None)

tar_dp = make_dp(root=ARCHIVE_ROOT, archive="tar")

pickle_bytesio_dp = make_dp(root=ARCHIVE_ROOT, archive="pickle", archive_content="bytesio")
pickle_tensor_dp = make_dp(root=ARCHIVE_ROOT, archive="pickle", archive_content="tensor")

torch_bytesio_dp = make_dp(root=ARCHIVE_ROOT, archive="torch", archive_content="bytesio")
torch_tensor_dp = make_dp(root=ARCHIVE_ROOT, archive="torch", archive_content="tensor")

# File-based
# ----------

print("File-based MapStyle")
bench(iterate_one_epoch, inp=mapstyle_ds, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print("File-based DP")
bench(iterate_one_epoch, inp=no_archive_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

# Archive-based
# -------------

print("tar archives")
bench(iterate_one_epoch, inp=tar_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print("pickle bytesio")
bench(iterate_one_epoch, inp=pickle_bytesio_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print("pickle bytesio->ToTensor()")
bench(
    iterate_one_epoch,
    inp=pickle_bytesio_dp.map(bytesio_to_tensor),
    warmup=1,
    num_exp=5,
    num_images_per_call=len(mapstyle_ds),
)

print("pickle tensor")
bench(iterate_one_epoch, inp=pickle_tensor_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print("torch bytesio")
bench(iterate_one_epoch, inp=torch_bytesio_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print("torch bytesio->ToTensor()")
bench(
    iterate_one_epoch,
    inp=torch_bytesio_dp.map(bytesio_to_tensor),
    warmup=1,
    num_exp=5,
    num_images_per_call=len(mapstyle_ds),
)

print("torch tensor")
bench(iterate_one_epoch, inp=torch_tensor_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print()

############
# DECODING #
############


from pathlib import Path
files = list(Path(f"{FS}/nicolashug/tinyimagenet/081318/train/n01443537/").glob("*.JPEG"))
tensors = [read_file(str(filepath)) for filepath in files]
np_arrays = [t.numpy() for t in tensors]

_dest = np.empty((64, 64, 3), dtype=np.uint8)
def decode_turbo(a):
    libffcv.imdecode(a, _dest, 64, 64, 64, 64, 0, 0, 1, 1, False, False)

bytesio_list = []
for filepath in files:
    with open(filepath, "rb") as f:
        bytesio_list.append(io.BytesIO(f.read()))

print("PIL.Image.open(bytesio).load()")
bench(lambda l: [Image.open(bytesio).convert("RGB").load() for bytesio in l], bytesio_list, num_images_per_call=len(bytesio_list))
print("decode_jpeg(tensor)")
bench(lambda l: [decode_jpeg(t, mode=ImageReadMode.RGB) for t in l], tensors, num_images_per_call=len(tensors))
print("libffcv.imdecode - using libjpeg-turbo")
bench(lambda l: [decode_turbo(a) for a in l], np_arrays, num_images_per_call=len(bytesio_list))
print()

################
# TRANSFORMING #
################

class ClassificationPresetTrain:
    def __init__(self, *, on):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        hflip_prob = 0.5
        crop_size = 224
        on = on.lower()
        if on not in ("tensor", "pil"):
            raise ValueError("oops")

        trans = []

        if on == "tensor":
            trans += [lambda x: x.contiguous()]
        

        trans += [transforms.RandomResizedCrop(crop_size, antialias=True)]
        if hflip_prob > 0:
            trans += [transforms.RandomHorizontalFlip(hflip_prob)]

        if on == "pil":
            trans += [transforms.PILToTensor()]

        trans += [
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


print("PIL tranforms")
pil_img = Image.open(bytesio_list[0]).convert("RGB")
bench(ClassificationPresetTrain(on="PIL"), pil_img, unit="m")
print("Tensor transforms")
tensor_img = decode_jpeg(tensors[0], mode=ImageReadMode.RGB)
bench(ClassificationPresetTrain(on="tensor"), tensor_img, unit="m")
print()


###########################
# DATA-READING + DECODING #
###########################

def decode(encoded_tensor):
    return decode_jpeg(encoded_tensor, mode=ImageReadMode.RGB)

pickle_decoded_dp = make_dp(root=ARCHIVE_ROOT, archive="pickle", archive_content="decoded")
torch_decoded_dp = make_dp(root=ARCHIVE_ROOT, archive="torch", archive_content="decoded")

ffcv_encoded = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=False, encoded=True)
ffcv_decoded = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=False, encoded=False)


print("pickle bytesio->ToTensor()->decode_jpeg()")
bench(
    iterate_one_epoch,
    inp=pickle_bytesio_dp.map(bytesio_to_tensor).map(decode),
    warmup=1,
    num_exp=5,
    num_images_per_call=len(mapstyle_ds),
)

print("torch bytesio->ToTensor()->decode_jpeg()")
bench(
    iterate_one_epoch,
    inp=torch_bytesio_dp.map(bytesio_to_tensor).map(decode),
    warmup=1,
    num_exp=5,
    num_images_per_call=len(mapstyle_ds),
)

print("FFCV loading + decoding")
bench(iterate_one_epoch, inp=ffcv_encoded, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print("pickle pre-decoded")
bench(iterate_one_epoch, inp=pickle_decoded_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print("torch pre-decoded")
bench(iterate_one_epoch, inp=torch_decoded_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print("FFCV loading (pre-decoded)")
bench(iterate_one_epoch, inp=ffcv_decoded, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

########################################
# DATA-READING + DECODING + TRANSFORMS #
########################################

ffcv_encoded_transforms = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=True, encoded=True)
ffcv_decoded_transforms = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=True, encoded=False)
ffcv_encoded_transforms_torchvision = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms="vision", encoded=True)
ffcv_decoded_transforms_torchvision = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms="vision", encoded=False)

mapstyle_ds_transforms = torchvision.datasets.ImageFolder(JPEG_FILES_ROOT, transform=ClassificationPresetTrain(on="pil"))

def tensor_loader(path):
    return decode(read_file(path))
mapstyle_ds_transforms_tensor = torchvision.datasets.ImageFolder(JPEG_FILES_ROOT, loader=tensor_loader, transform=ClassificationPresetTrain(on="tensor"))

print("pickle bytesio->ToTensor()->decode_jpeg()->Transforms()")
bench(
    iterate_one_epoch,
    inp=pickle_bytesio_dp.map(bytesio_to_tensor).map(decode).map(ClassificationPresetTrain(on="tensor")),
    warmup=1,
    num_exp=5,
    unit="m",
    num_images_per_call=len(mapstyle_ds),
)

print("torch bytesio->ToTensor()->decode_jpeg()->Transforms()")
bench(
    iterate_one_epoch,
    inp=torch_bytesio_dp.map(bytesio_to_tensor).map(decode).map(ClassificationPresetTrain(on="tensor")),
    warmup=1,
    num_exp=5,
    unit="m",
    num_images_per_call=len(mapstyle_ds),
)

print("FFCV loading + decoding + transforms")
bench(iterate_one_epoch, inp=ffcv_encoded_transforms, warmup=1, num_exp=5, unit="m", num_images_per_call=len(mapstyle_ds))


print("File-based Mapstyle -> decode_jpeg() -> Transforms")  # a.k.a the current torchvision transforms
bench(iterate_one_epoch, inp=mapstyle_ds_transforms_tensor, warmup=1, num_exp=5, unit="m", num_images_per_call=len(mapstyle_ds))

print("File-based Mapstyle -> Image.open() -> PIL transforms")  # a.k.a the current torchvision transforms
bench(iterate_one_epoch, inp=mapstyle_ds_transforms, warmup=1, num_exp=5, unit="m", num_images_per_call=len(mapstyle_ds))

print("FFCV loading + decoding + transforms (torchvision)")
bench(iterate_one_epoch, inp=ffcv_encoded_transforms_torchvision, warmup=1, num_exp=5, unit="m", num_images_per_call=len(mapstyle_ds))

print("pickle pre-decoded -> Transforms()")
bench(iterate_one_epoch, inp=pickle_decoded_dp.map(ClassificationPresetTrain(on="tensor")), warmup=1, num_exp=5, unit="m", num_images_per_call=len(mapstyle_ds))

print("torch pre-decoded -> Transforms()")
bench(iterate_one_epoch, inp=torch_decoded_dp.map(ClassificationPresetTrain(on="tensor")), warmup=1, num_exp=5, unit="m", num_images_per_call=len(mapstyle_ds))

print("FFCV loading (pre-decoded) + transforms")
bench(iterate_one_epoch, inp=ffcv_decoded_transforms, warmup=1, num_exp=5, unit="m", num_images_per_call=len(mapstyle_ds))

print("FFCV loading (pre-decoded) + transforms (torchvision)")
bench(iterate_one_epoch, inp=ffcv_decoded_transforms_torchvision, warmup=1, num_exp=5, unit="m", num_images_per_call=len(mapstyle_ds))

# Need to check if FFCV is only using 1 CPU or more.
# Looks a bit fishy, tried again setting context num_workers=1
# If it's still way faster maybe it's because the pre-fetching still happens in the background even with 1 thread.

# Benchmark FFCV with torchvision transforms
# Benchmark mapstyle entire pipeline

# FFCV and DataLoader with num_workers=12
# FFCV with num_workers=12 and batch_size>1
# Maybe one day: FFCV with its normal tranforms but overriding the memory allocation
