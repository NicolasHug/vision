import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class PresetEval(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.transforms = Compose(
            [
                ToTensor(),
                Scale(),
                CheckDtype(),
            ]
        )

    def __call__(self, img1, img2, flow, valid):
        return self.transforms(img1, img2, flow, valid)


class FlowAugmentor(torch.nn.Module):
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        super().__init__()

        transforms = [
            ToTensor(),
            AsymmetricColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14, p=0.2
            ),  # TODO: these are different for sparse
            RandomApply([RandomErase()], p=0.5),
            MaybeResizeAndCrop(crop_size=crop_size, min_scale=min_scale, max_scale=max_scale),
        ]

        if do_flip:
            transforms += [RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.1)]

        transforms += [Scale(), CheckDtype()]
        self.transforms = Compose(transforms)

    def __call__(self, img1, img2, flow, valid):
        return self.transforms(img1, img2, flow, valid)


class CheckDtype(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype

    def forward(self, *args):
        assert all(x.dtype == self.dtype for x in args if x is not None)
        return args


class Scale(torch.nn.Module):
    # TODO: find a better name
    # ALso: Calling this before converting the images to cuda seems to affect epe quite a bit

    def forward(self, img1, img2, flow, valid):
        img1 = F.convert_image_dtype(img1, dtype=torch.float32) * 2 - 1
        img2 = F.convert_image_dtype(img2, dtype=torch.float32) * 2 - 1

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        return img1, img2, flow, valid


class ToTensor(torch.nn.Module):
    def forward(self, img1, img2, flow, valid):
        img1 = F.pil_to_tensor(img1)
        img2 = F.pil_to_tensor(img2)

        if isinstance(flow, np.ndarray):
            flow = torch.from_numpy(flow).permute((2, 0, 1))

        return img1, img2, flow, valid


class AsymmetricColorJitter(T.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.2):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.p = p

    def forward(self, img1, img2, flow, valid):

        if torch.rand(1) < self.p:
            # asymmetric
            img1 = super().forward(img1)
            img2 = super().forward(img2)
        else:
            # symmetric
            batch = torch.stack([img1, img2])
            batch = super().forward(batch)
            img1, img2 = batch[0], batch[1]

        return img1, img2, flow, valid


class RandomErase(torch.nn.Module):
    def forward(self, img1, img2, flow, valid):
        bounds = [50, 100]
        ht, wd = img2.shape[:2]
        mean_color = img2.view(3, -1).float().mean(axis=-1).round()
        for _ in range(torch.randint(1, 3, size=(1,)).item()):
            x0 = torch.randint(0, wd, size=(1,)).item()
            y0 = torch.randint(0, ht, size=(1,)).item()
            dx, dy = torch.randint(bounds[0], bounds[1], size=(2,))
            img2[:, y0 : y0 + dy, x0 : x0 + dx] = mean_color[:, None, None]

        return img1, img2, flow, valid


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, img1, img2, flow, valid):
        if torch.rand(1) > self.p:
            return img1, img2, flow, valid

        img1 = F.hflip(img1)
        img2 = F.hflip(img2)
        flow = F.hflip(flow) * torch.tensor([-1, 1])[:, None, None]
        if valid is not None:
            valid = F.hflip(valid)
        return img1, img2, flow, valid


class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(self, img1, img2, flow, valid):
        if torch.rand(1) > self.p:
            return img1, img2, flow, valid

        img1 = F.vflip(img1)
        img2 = F.vflip(img2)
        flow = F.vflip(flow) * torch.tensor([1, -1])[:, None, None]
        if valid is not None:
            valid = F.vflip(valid)
        return img1, img2, flow, valid


class MaybeResizeAndCrop(torch.nn.Module):
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5):
        super().__init__()
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

    def forward(self, img1, img2, flow, valid):
        # randomly sample scale
        h, w = img1.shape[-2:]
        min_scale = max((self.crop_size[0] + 8) / h, (self.crop_size[1] + 8) / w)

        scale = 2 ** torch.FloatTensor(1).uniform_(self.min_scale, self.max_scale).item()
        scale_x = scale
        scale_y = scale
        if torch.rand(1) < self.stretch_prob:
            scale_x *= 2 ** torch.FloatTensor(1).uniform_(-self.max_stretch, self.max_stretch).item()
            scale_y *= 2 ** torch.FloatTensor(1).uniform_(-self.max_stretch, self.max_stretch).item()

        scale_x = max(scale_x, min_scale)
        scale_y = max(scale_y, min_scale)

        new_h, new_w = round(h * scale_y), round(w * scale_x)

        if torch.rand(1).item() < self.spatial_aug_prob:
            # rescale the images
            img1 = F.resize(img1, size=(new_h, new_w))
            img2 = F.resize(img2, size=(new_h, new_w))
            flow = F.resize(flow, size=(new_h, new_w))
            #  TODO : HANDLE VALID HERE
            flow = flow * torch.tensor([scale_x, scale_y])[:, None, None]

        y0 = torch.randint(0, img1.shape[1] - self.crop_size[0], size=(1,)).item()
        x0 = torch.randint(0, img1.shape[2] - self.crop_size[1], size=(1,)).item()

        img1 = img1[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        img2 = img2[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        flow = flow[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        return img1, img2, flow, valid


class RandomApply(T.RandomApply):
    def forward(self, img1, img2, flow, valid):
        if self.p < torch.rand(1):
            return img1, img2, flow, valid
        for t in self.transforms:
            img1, img2, flow, valid = t(img1, img2, flow, valid)
        return img1, img2, flow, valid


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2, flow, valid):
        for t in self.transforms:
            img1, img2, flow, valid = t(img1, img2, flow, valid)
        return img1, img2, flow, valid


class SparseFlowAugmentor:
    pass
