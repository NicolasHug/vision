from common import ARCHIVE_ROOT, bench, bytesio_to_tensor, iterate_one_epoch, JPEG_FILES_ROOT
from dataset_helpers import make_dp, make_mapstyle, make_webdataset, with_DL


def just_read_the_file(img_path):
    with open(img_path, "rb") as f:
        return f.read()


mapstyle_ds = make_mapstyle(root=JPEG_FILES_ROOT, loader=just_read_the_file)
no_archive_dp = make_dp(root=JPEG_FILES_ROOT, archive=None).map(just_read_the_file)

tar_dp = make_dp(root=ARCHIVE_ROOT, archive="tar")

wds = make_webdataset(root=ARCHIVE_ROOT)

pickle_bytesio_dp = make_dp(root=ARCHIVE_ROOT, archive="pickle", archive_content="bytesio")
pickle_tensor_dp = make_dp(root=ARCHIVE_ROOT, archive="pickle", archive_content="tensor")

torch_bytesio_dp = make_dp(root=ARCHIVE_ROOT, archive="torch", archive_content="bytesio")
torch_tensor_dp = make_dp(root=ARCHIVE_ROOT, archive="torch", archive_content="tensor")


if __name__ == "__main__":

    print("File-based MapStyle")
    bench(iterate_one_epoch, with_DL(mapstyle_ds))

    print("File-based DP")
    bench(iterate_one_epoch, with_DL(no_archive_dp))

    print("tar archives (WebDataset)")
    bench(iterate_one_epoch, with_DL(wds))

    print("tar archives")
    bench(iterate_one_epoch, with_DL(tar_dp))

    print("pickle bytesio")
    bench(iterate_one_epoch, with_DL(pickle_bytesio_dp))

    print("pickle bytesio->ToTensor()")
    bench(iterate_one_epoch, with_DL(pickle_bytesio_dp.map(bytesio_to_tensor)))

    print("pickle tensor")
    bench(iterate_one_epoch, with_DL(pickle_tensor_dp))

    print("torch bytesio")
    bench(iterate_one_epoch, with_DL(torch_bytesio_dp))

    print("torch bytesio->ToTensor()")
    bench(iterate_one_epoch, with_DL(torch_bytesio_dp.map(bytesio_to_tensor)))

    print("torch tensor")
    bench(iterate_one_epoch, with_DL(torch_tensor_dp))

    print()
