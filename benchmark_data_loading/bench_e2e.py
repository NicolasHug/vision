from bench_data_reading import pickle_bytesio_dp, wds
from bench_transforms import ClassificationPresetTrain
from common import ARCHIVE_ROOT, args, bench, bytesio_to_tensor, decode, iterate_one_epoch, JPEG_FILES_ROOT, suppress
from dataset_helpers import make_dp, make_ffcv_dataloader, make_mapstyle, with_DL
from PIL import Image
from torchvision.io import read_file


ffcv_encoded_transforms = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=True, encoded=True)

if args.tiny:
    ffcv_decoded_transforms = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=True, encoded=False)
    ffcv_decoded_transforms_torchvision = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms="vision", encoded=False)


def tensor_loader(path):
    return decode(read_file(path))


def pil_loader(path):
    return Image.open(path).convert("RGB")


mapstyle_ds_transforms_pil = make_mapstyle(root=JPEG_FILES_ROOT, transform=ClassificationPresetTrain(on="pil"))
mapstyle_ds_transforms_tensor = make_mapstyle(
    root=JPEG_FILES_ROOT, loader=tensor_loader, transform=ClassificationPresetTrain(on="tensor")
)
no_archive_dp_transforms_pil = (
    make_dp(root=JPEG_FILES_ROOT, archive=None).map(pil_loader).map(ClassificationPresetTrain(on="pil"))
)
no_archive_dp_transforms_tensor = (
    make_dp(root=JPEG_FILES_ROOT, archive=None).map(tensor_loader).map(ClassificationPresetTrain(on="tensor"))
)

if __name__ == "__main__":
    with suppress():
        print("tar archives (WebDataset) bytesio->ToTensor()->decode_jpeg()->Transforms()")
        dp = with_DL(wds.map(bytesio_to_tensor).map(decode).map(ClassificationPresetTrain(on="tensor")))
        bench(iterate_one_epoch, inp=dp, unit="m")

    with suppress():
        print("pickle bytesio->ToTensor()->decode_jpeg()->Transforms()")
        dp = with_DL(pickle_bytesio_dp.map(bytesio_to_tensor).map(decode).map(ClassificationPresetTrain(on="tensor")))
        bench(iterate_one_epoch, inp=dp, unit="m")

    # with suppress():
    #     print("torch bytesio->ToTensor()->decode_jpeg()->Transforms()")
    #     dp = with_DL(torch_bytesio_dp.map(bytesio_to_tensor).map(decode).map(ClassificationPresetTrain(on="tensor")))
    #     bench(iterate_one_epoch, inp=dp, unit="m")

    with suppress():
        print("FFCV loading + decoding + transforms")
        bench(iterate_one_epoch, inp=ffcv_encoded_transforms, unit="m")

    with suppress():
        print("File-based Mapstyle -> Image.open() -> PIL transforms")  # a.k.a the current torchvision pipeline
        bench(iterate_one_epoch, inp=with_DL(mapstyle_ds_transforms_pil), unit="m")

    with suppress():
        print("File-based Mapstyle -> decode_jpeg() -> Transforms")
        bench(iterate_one_epoch, inp=with_DL(mapstyle_ds_transforms_tensor), unit="m")

    # with suppress():
    #     print("File-based DP -> Image.open() -> PIL transforms")
    #     bench(iterate_one_epoch, inp=with_DL(no_archive_dp_transforms_pil), unit="m")

    # with suppress():
    #     print("File-based DP -> decode_jpeg() -> Transforms")
    #     bench(iterate_one_epoch, inp=with_DL(no_archive_dp_transforms_tensor), unit="m")

    # if args.tiny:
    #     print("pickle pre-decoded -> Transforms()")
    #     dp = with_DL(pickle_decoded_dp.map(ClassificationPresetTrain(on="tensor")))
    #     bench(iterate_one_epoch, inp=dp, unit="m")

    #     print("torch pre-decoded -> Transforms()")
    #     dp = with_DL(torch_decoded_dp.map(ClassificationPresetTrain(on="tensor")))
    #     bench(iterate_one_epoch, inp=dp, unit="m")

    #     print("FFCV loading (pre-decoded) + transforms")
    #     bench(iterate_one_epoch, inp=ffcv_decoded_transforms, unit="m")
