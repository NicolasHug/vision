from ffcv.fields.basics import IntDecoder
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader as FFCVLoader
import torch
from references.torchdata_benchmarks.presets import ClassificationPresetTrain
import torchvision
import datetime
from time import time
from torchvision.io import read_file, decode_jpeg
from PIL import Image
import io
import argparse
from references.torchdata_benchmarks.helpers import make_dp

def get_time_in_training(time_per_image, dataset_size="imagenet", num_epochs=10):
    if isinstance(dataset_size, str):
        dataset_size = {"tiny": 110_000, "imagenet": 1_281_167 + 50_000}[dataset_size]
    tot = time_per_image * dataset_size * num_epochs
    return datetime.timedelta(seconds=int(tot))

def bench(f, inp, num_exp=1000, warmup=10, unit="μ", num_images_per_call=1):
    # Computes PER IMAGE median times
    for _ in range(warmup):
        f(inp)

    times = []
    for _ in range(num_exp):
        start = time()
        f(inp)
        end = time()
        times.append((end - start))
    
    mul = {"μ":1e6, "m": 1e3, "s": 1}[unit]
    times = torch.tensor(times) / num_images_per_call
    median_sec = torch.median(times)

    times_unit = times * mul
    median_unit = torch.median(times_unit)

    time_in_training = get_time_in_training(median_sec)
    s = f"Median = {median_unit:.1f} {unit}{'s' if unit != 's' else ''}, std={torch.std(times_unit):.2f}"
    print(f"{s:30} Accounts for {time_in_training}")
    return median_sec

# FS = "/fsx_isolated"
FS = "/ontap_isolated"
filepath = f"{FS}/nicolashug/tinyimagenet/081318/train/n01443537/n01443537_0.JPEG"
tensor = read_file(filepath)
with open(filepath, "rb") as f:
    bytesio = io.BytesIO(f.read())

print("Image.open(bytesio).load()")
bench(lambda x: Image.open(x).load(), bytesio)
print("decode_jpeg(tensor)")
bench(decode_jpeg, tensor)
print()

print("PIL tranforms")
pil_img = Image.open(bytesio)
bench(ClassificationPresetTrain(crop_size=224, on_pil_images=True), pil_img, unit="m")
print("Tensor transforms")
tensor_img = decode_jpeg(tensor)
bench(ClassificationPresetTrain(crop_size=224, on_pil_images=False), tensor_img, unit="m")
print()


def get_datasets():
    # Ugly cause we have to use the args statefulness
    common_root = f"{FS}/nicolashug/tinyimagenet/081318/"
    mapstyle_root = common_root + "train"
    archives_root = common_root + "archives/train"

    args = argparse.ArgumentParser().parse_args()
    args.no_decode = True
    args.tiny = True
    no_transforms = lambda x: x
    args.archive_size = 500
    args.archive_content = None

    args.archive = None
    no_archive_dp = make_dp(mapstyle_root, transforms=no_transforms, args=args)

    args.archive = "tar"
    tar_dp = make_dp(archives_root, transforms=no_transforms, args=args)

    args.archive = "pickle"                                                     

    args.archive_content = "bytesio"                                            
    pickle_bytesio_dp = make_dp(archives_root, transforms=no_transforms, args=args)

    args.archive_content = "tensor"                                             
    pickle_tensor_dp = make_dp(archives_root, transforms=no_transforms, args=args)

    args.archive = "torch"                                                      

    args.archive_content = "bytesio"                                            
    torch_bytesio_dp = make_dp(archives_root, transforms=no_transforms, args=args)

    args.archive_content = "tensor"                                             
    torch_tensor_dp = make_dp(archives_root, transforms=no_transforms, args=args)

    def loader(img_path):
        # reading only, no-decoding
        with open(img_path, "rb") as f:
            f.read()
    mapstyle_ds = torchvision.datasets.ImageFolder(mapstyle_root, loader=loader)

    def make_ffcv_dataloader():
        return FFCVLoader(
            f"{archives_root}/ffcv.beton",
            batch_size=1,
            drop_last=False,
            num_workers=1,
            os_cache=True,
            pipelines={
                # "img": [SimpleRGBImageDecoder()],
                # "label": [IntDecoder()],
                "img": [SimpleRGBImageDecoder()],
                "label": [IntDecoder()],
            },
            batches_ahead=2,  # Same default as prefetch_factor from DataLoader
        )
    ffcv_dl = make_ffcv_dataloader()

    return mapstyle_ds, no_archive_dp, tar_dp, pickle_bytesio_dp, pickle_tensor_dp, torch_bytesio_dp, torch_tensor_dp, ffcv_dl

def iterate_one_epoch(obj):
    if isinstance(obj, (torch.utils.data.datapipes.datapipe.IterDataPipe, FFCVLoader)):
        list(obj)
    else:
        # Need to reproduce "random" access
        indices = torch.randperm(len(obj))
        for i in indices:
            obj[i]

mapstyle_ds, no_archive_dp, tar_dp, pickle_bytesio_dp, pickle_tensor_dp, torch_bytesio_dp, torch_tensor_dp, ffcv_dl = get_datasets()


print("Reading mapstyle")
bench(iterate_one_epoch, inp=mapstyle_ds, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print("Reading DP (no archive)")
bench(iterate_one_epoch, inp=no_archive_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print("Reading tar")
bench(iterate_one_epoch, inp=tar_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))
                                                                            
print("Reading pickle bytesio")
bench(iterate_one_epoch, inp=pickle_bytesio_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print("Reading pickle tensor")
bench(iterate_one_epoch, inp=pickle_tensor_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))
                                                                            
print("Reading torch bytesio")
bench(iterate_one_epoch, inp=torch_bytesio_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))
                                                                            
print("Reading torch tensor")
bench(iterate_one_epoch, inp=torch_tensor_dp, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))

print("FFCV loading + decoding")
bench(iterate_one_epoch, inp=ffcv_dl, warmup=1, num_exp=5, num_images_per_call=len(mapstyle_ds))
