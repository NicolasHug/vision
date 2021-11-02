import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


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
            # asymmetric: different transform for img1 and img2
            img1 = super().forward(img1)
            img2 = super().forward(img2)
        else:
            # symmetric: same transform for img1 and img2
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
    # This transform will resize the input with a given proba, and then crop it.
    # These are the reversed operations of the built-in RandomResizedCrop,
    # although the order of the operations doesn't matter too much.
    # The reason we don't rely on RandomResizedCrop is because of a significant
    # difference in the parametrization of both transforms.
    #
    # There *is* a mapping between the inputs of MaybeResizeAndCrop and those of
    # RandomResizedCrop, but the issue is that the parameters are sampled at
    # random, with different distributions. Plotting (the equivalent of) `scale`
    # and `ratio` from MaybeResizeAndCrop shows that the distributions of these
    # parameter are very different from what can be obtained from the
    # parametrization of RandomResizedCrop. I tried training RAFT by using
    # RandomResizedCrop and tweaking the parameters a bit, but couldn't get
    # an epe as good as with MaybeResizeAndCrop.
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, stretch_prob=0.8):
        super().__init__()
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.stretch_prob = stretch_prob
        self.resize_prob = 0.8
        self.max_stretch = 0.2

    def forward(self, img1, img2, flow, valid):
        # randomly sample scale
        h, w = img1.shape[-2:]
        # Note: in original code, they use + 1 instead of + 8 for sparse datasets (e.g. Kitti)
        # It shouldn't matter much
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

        if torch.rand(1).item() < self.resize_prob:
            # rescale the images
            img1 = F.resize(img1, size=(new_h, new_w))
            img2 = F.resize(img2, size=(new_h, new_w))
            if valid is None:
                flow = F.resize(flow, size=(new_h, new_w))
                flow = flow * torch.tensor([scale_x, scale_y])[:, None, None]
            else:
                flow, valid = self._resize_sparse_flow(flow, valid, scale_x=scale_x, scale_y=scale_y)

        # Note: For sparse datasets (Kitti), the original code uses a "margin"
        # See e.g. https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py#L220:L220
        # We don't, not sure it matters much
        y0 = torch.randint(0, img1.shape[1] - self.crop_size[0], size=(1,)).item()
        x0 = torch.randint(0, img1.shape[2] - self.crop_size[1], size=(1,)).item()

        img1 = img1[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        img2 = img2[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        flow = flow[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        if valid is not None:
            valid = valid[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        return img1, img2, flow, valid

    def _resize_sparse_flow(self, flow, valid, scale_x=1.0, scale_y=1.0):
        # This resizes both the flow and the valid mask (which is assumed to be reasonably sparse)
        # There are as-many non-zero values in the original flow as in the resized flow (up to OOB)
        # So for example if scale_x = scale_y = 2, the sparsity of the output flow is multiplied by 4

        h, w = flow.shape[-2:]

        h_new = int(round(h * scale_y))
        w_new = int(round(w * scale_x))
        flow_new = torch.zeros(size=[2, h_new, w_new], dtype=flow.dtype)
        valid_new = torch.zeros(size=[h_new, w_new], dtype=valid.dtype)

        jj, ii = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")

        valid_bool = valid.to(bool)
        ii_valid, jj_valid = ii[valid_bool], jj[valid_bool]

        ii_valid_new = torch.round(ii_valid.to(float) * scale_y).to(torch.long)
        jj_valid_new = torch.round(jj_valid.to(float) * scale_x).to(torch.long)

        within_bounds_mask = (0 <= ii_valid_new) & (ii_valid_new < h_new) & (0 <= jj_valid_new) & (jj_valid_new < w_new)

        ii_valid = ii_valid[within_bounds_mask]
        jj_valid = jj_valid[within_bounds_mask]
        ii_valid_new = ii_valid_new[within_bounds_mask]
        jj_valid_new = jj_valid_new[within_bounds_mask]

        valid_flow_new = flow[:, ii_valid, jj_valid]
        valid_flow_new[0] *= scale_x
        valid_flow_new[1] *= scale_y

        flow_new[:, ii_valid_new, jj_valid_new] = valid_flow_new
        valid_new[ii_valid_new, jj_valid_new] = 1

        return flow_new, valid_new


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
