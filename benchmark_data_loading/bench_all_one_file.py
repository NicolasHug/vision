# TODO: Remove this file. I only use it because the script doesn't terminate
# with the proto reading service so I can't run multiple files

from bench_data_reading import pickle_bytesio_dp, torch_bytesio_dp, wds, tar_dp, mapstyle_ds, no_archive_dp 
from bench_data_reading_decoding import pickle_decoded_dp, torch_decoded_dp
from bench_transforms import ClassificationPresetTrain
from common import ARCHIVE_ROOT, args, bench, bytesio_to_tensor, bytes_to_tensor, decode, iterate_one_epoch, JPEG_FILES_ROOT, suppress
from dataset_helpers import make_dp, make_ffcv_dataloader, make_mapstyle, with_DL
from PIL import Image
from torchvision.io import read_file


ffcv_encoded_transforms = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=True, encoded=True)

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

    print("File reading: no decoding, no transforms")
    with suppress():
        print("File-based MapStyle DLV1")
        bench(iterate_one_epoch, with_DL(mapstyle_ds, dl="v1"))

    with suppress():
        print("File-based DP DLV2")
        bench(iterate_one_epoch, lambda: with_DL(no_archive_dp, dl="v2"))

    with suppress():
        print("tar archives->ToTensor() DLV1")
        bench(iterate_one_epoch, with_DL(tar_dp.map(bytes_to_tensor), dl="v1"))

    with suppress():
        print("tar archives->ToTensor() DLV2")
        bench(iterate_one_epoch, lambda: with_DL(tar_dp.map(bytes_to_tensor), dl="v2"))

    print("E2E: read -> decode -> transform")

    with suppress():
        print("tar archives->ToTensor()->decode_jpeg()->Transforms() DLV2")
        bench(iterate_one_epoch, lambda: with_DL(tar_dp.map(bytes_to_tensor).map(decode).map(ClassificationPresetTrain(on="tensor")), dl="v2"))

    with suppress():
        print("tar archives->ToTensor()->decode_jpeg()->Transforms() DLV1")
        bench(iterate_one_epoch, with_DL(tar_dp.map(bytes_to_tensor).map(decode).map(ClassificationPresetTrain(on="tensor")), dl="v1"))

    with suppress():
        print("FFCV loading + decoding + transforms")
        bench(iterate_one_epoch, inp=ffcv_encoded_transforms)

    with suppress():
        print("File-based Mapstyle -> Image.open() -> PIL transforms DLV1")  # a.k.a the current torchvision pipeline
        bench(iterate_one_epoch, inp=with_DL(mapstyle_ds_transforms_pil, dl="v1"))

    with suppress():
        print("File-based DP -> Image.open() -> PIL transforms DLV2")
        bench(iterate_one_epoch, inp=lambda: with_DL(no_archive_dp_transforms_pil, dl="v2"))

    with suppress():
        print("File-based Mapstyle -> decode_jpeg() -> Transforms DLV1")
        bench(iterate_one_epoch, inp=with_DL(mapstyle_ds_transforms_tensor, dl="v1"))

    with suppress():
        print("File-based DP -> decode_jpeg() -> Transforms DLV2")
        bench(iterate_one_epoch, inp=lambda: with_DL(no_archive_dp_transforms_tensor, dl="v2"))

    for prefetch_main in (0, 8, 16):
        for prefetch_worker in (0, 2, 4, 16):
            args.prefetch_main = prefetch_main
            args.prefetch_worker = prefetch_worker
            print(args)
            with suppress():
                print("tar archives->ToTensor()->decode_jpeg()->Transforms() DLV2")
                bench(iterate_one_epoch, lambda: with_DL(tar_dp.map(bytes_to_tensor).map(decode).map(ClassificationPresetTrain(on="tensor")), dl="v2"))
