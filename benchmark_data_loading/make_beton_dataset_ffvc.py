from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter
from torchvision.datasets import ImageFolder


# TODO: merge this with make_archives.py

# root = "/datasets01_ontap/tinyimagenet/081318/train/"
root = "/ontap_isolated/nicolashug/imagenet_full_size/061417/train"
ds = ImageFolder(root)

# write_path = "/data/home/nicolashug/cluster/work/downloads/imagenet_train_jpg.beton"
# write_path = "/fsx_isolated/nicolashug/tinyimagenet/081318/archives/train/ffcv_decoded.beton"
write_path = "/ontap_isolated/nicolashug/imagenet_full_size/061417/archives/train/ffcv.beton"
writer = DatasetWriter(
    write_path,
    {
        "img": RGBImageField(write_mode="jpg"),
        "label": IntField(),
    },
    num_workers=24,
)

print(writer)

writer.from_indexed_dataset(ds, shuffle_indices=True)

print("done")
