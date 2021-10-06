import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class FlowAugmentor:
    # TODO: maybe common class with SparseAugmentor?
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        if torch.rand(1).item() < self.asymmetric_color_aug_prob:
            img1 = self.jitter(img1)
            img2 = self.jitter(img2)
        else:
            batch = torch.stack([img1, img2])
            batch = self.jitter(batch)
            img1, img2 = batch[0], batch[1]

        return img1, img2

    def eraser_transform(self, img):
        """Occlusion augmentation"""
        if torch.rand(1).item() > self.eraser_aug_prob:
            return img

        bounds = [50, 100]
        ht, wd = img.shape[:2]
        mean_color = img.view(3, -1).float().mean(axis=-1).round()
        for _ in range(torch.randint(1, 3, size=(1,)).item()):
            x0 = torch.randint(0, wd, size=(1,)).item()
            y0 = torch.randint(0, ht, size=(1,)).item()
            dx, dy = torch.randint(bounds[0], bounds[1], size=(2,))
            img[:, y0 : y0 + dy, x0 : x0 + dx] = mean_color[:, None, None]
        return img

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        h, w = img1.shape[-2:]
        min_scale = max((self.crop_size[0] + 8) / h, (self.crop_size[1] + 8) / w)

        scale = 2 ** torch.FloatTensor(1).uniform_(self.min_scale, self.max_scale).item()
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
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
            flow = flow * torch.tensor([scale_x, scale_y])[:, None, None]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:
                img1 = F.hflip(img1)
                img2 = F.hflip(img2)
                flow = F.hflip(flow) * torch.tensor([-1, 1])[:, None, None]

            if np.random.rand() < self.v_flip_prob:
                img1 = F.vflip(img1)
                img2 = F.vflip(img2)
                flow = F.vflip(flow) * torch.tensor([1, -1])[:, None, None]

        y0 = torch.randint(0, img1.shape[1] - self.crop_size[0], size=(1,)).item()
        x0 = torch.randint(0, img1.shape[2] - self.crop_size[1], size=(1,)).item()

        img1 = img1[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        img2 = img2[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        flow = flow[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img2 = self.eraser_transform(img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        return img1, img2, flow


class SparseFlowAugmentor:
    pass
