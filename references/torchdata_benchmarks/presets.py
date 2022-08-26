import torch
from torchvision.transforms import transforms


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        on_pil_images,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        hflip_prob=0.5,
    ):
        trans = []
        if not on_pil_images:
            trans += [lambda x: x.contiguous()]

        trans += [transforms.RandomResizedCrop(crop_size, antialias=True)]
        if hflip_prob > 0:
            trans += [transforms.RandomHorizontalFlip(hflip_prob)]

        if on_pil_images:
            trans += [transforms.PILToTensor()]

        trans += [
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        on_pil_images,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):

        trans = []
        if not on_pil_images:
            trans += [lambda x: x.contiguous()]
        trans += [
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor() if on_pil_images else torch.nn.Identity(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ]
        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)
