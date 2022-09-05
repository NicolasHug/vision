from bench_data_reading import pickle_bytesio_dp, torch_bytesio_dp
from common import ARCHIVE_ROOT, args, bench, bytesio_to_tensor, decode, iterate_one_epoch, suppress
from dataset_helpers import make_dp, make_ffcv_dataloader, with_DL


ffcv_encoded = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=False, encoded=True)

if args.tiny:
    pickle_decoded_dp = make_dp(root=ARCHIVE_ROOT, archive="pickle", archive_content="decoded")
    torch_decoded_dp = make_dp(root=ARCHIVE_ROOT, archive="torch", archive_content="decoded")
    ffcv_decoded = make_ffcv_dataloader(root=ARCHIVE_ROOT, transforms=False, encoded=False)
else:
    pickle_decoded_dp = torch_decoded_dp = ffcv_decoded = None

if __name__ == "__main__":

    with suppress():
        print("pickle bytesio->ToTensor()->decode_jpeg()")
        bench(iterate_one_epoch, inp=with_DL(pickle_bytesio_dp.map(bytesio_to_tensor).map(decode)))

    with suppress():
        print("torch bytesio->ToTensor()->decode_jpeg()")
        bench(iterate_one_epoch, inp=with_DL(torch_bytesio_dp.map(bytesio_to_tensor).map(decode)))

    with suppress():
        print("FFCV loading + decoding")
        bench(iterate_one_epoch, inp=ffcv_encoded)

    if args.tiny:
        print("pickle pre-decoded")
        bench(iterate_one_epoch, inp=with_DL(pickle_decoded_dp))

        print("torch pre-decoded")
        bench(iterate_one_epoch, inp=with_DL(torch_decoded_dp))

        print("FFCV loading (pre-decoded)")
        bench(iterate_one_epoch, inp=ffcv_decoded)
