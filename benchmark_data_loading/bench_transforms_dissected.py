import torch

import torchvision.transforms as transforms
from bench_decoding import bytesio_list, decoded_tensors
from common import bench
from PIL import Image

# from bench_transforms import ClassificationPresetTrain

# TODO: move that stuff in the bench_transforms.py file


class ToContiguous(torch.nn.Module):
    # Can't be lambda otherwise datapipes fail
    def forward(self, x):
        return x.contiguous()


class RandomCrop(transforms.RandomResizedCrop):
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return transforms.functional.crop(img, i, j, h, w)


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
            trans += [ToContiguous()]

        trans += [
            RandomCrop(size=(1, 1)),  # Note: size is ignored here
            transforms.Resize(size=(crop_size, crop_size), antialias=True),
        ]

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


pil_imgs = [Image.open(bytesio).convert("RGB") for bytesio in bytesio_list]

for input_type in ("PIL", "Tensor"):
    if input_type.lower() == "tensor":
        tf = ClassificationPresetTrain(on="tensor")
        inputs = decoded_tensors
    else:
        tf = ClassificationPresetTrain(on="pil")
        inputs = pil_imgs

    print(f"{input_type.upper()} TRANSFORMS")
    bench(lambda l: [tf(t) for t in l], inputs, unit="m", num_images_per_call=len(inputs))

    for current_tf in tf.transforms.transforms:
        name = current_tf.__class__.__name__
        print(name)
        if name == "Resize":
            if input_type.lower() == "tensor":
                sizes = [t.shape[1:] for t in inputs]
            else:
                sizes = [tuple(reversed(t.size)) for t in inputs]

            avg_crop_size = torch.tensor(sizes).float().mean(dim=0).int().tolist()
            print(f"{avg_crop_size = }, resized to 224")

        bench(lambda l: [current_tf(t) for t in l], inputs, unit="m", num_images_per_call=len(inputs))
        inputs = [current_tf(t) for t in inputs]
