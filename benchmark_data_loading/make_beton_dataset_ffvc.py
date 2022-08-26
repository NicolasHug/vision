from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter
from torchvision.datasets import ImageFolder

root = "/datasets01_ontap/tinyimagenet/081318/train/"
ds = ImageFolder(root)

# write_path = "/data/home/nicolashug/cluster/work/downloads/imagenet_train_jpg.beton"
write_path = "/fsx_isolated/nicolashug/tinyimagenet/081318/archives/train/ffcv_decoded.beton"
writer = DatasetWriter(
    write_path,
    {
        "img": RGBImageField(write_mode="raw"),
        "label": IntField(),
    },
    num_workers=24,
)

print(writer)

writer.from_indexed_dataset(ds, shuffle_indices=True)

print("done")
