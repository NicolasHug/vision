import torch
import transforms as T


class OpticalFlowPresetEval(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.DropAlphaChannel(),
                T.Scale(),
                T.ValidateModelInput(),
            ]
        )

    def __call__(self, img1, img2, flow, valid):
        return self.transforms(img1, img2, flow, valid)


class OpticalFlowPresetTrain(torch.nn.Module):
    def __init__(
        self,
        # MaybeRandomResizeAndCrop params
        crop_size,
        min_scale=-0.2,
        max_scale=0.5,
        stretch_prob=0.8,
        # AsymmetricColorJitter params
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.5 / 3.14,
        # Random[H,V]Flip params
        asymmetric_jitter_prob=0.2,
        do_flip=True,
    ):
        super().__init__()

        transforms = [
            T.ToTensor(),
            T.DropAlphaChannel(),
            T.AsymmetricColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=asymmetric_jitter_prob
            ),
            T.RandomApply([T.RandomErase()], p=0.5),
            T.MaybeResizeAndCrop(
                crop_size=crop_size, min_scale=min_scale, max_scale=max_scale, stretch_prob=stretch_prob
            ),
        ]

        if do_flip:
            transforms += [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.1)]

        transforms += [T.Scale(), T.MakeValidFlowMask(), T.ValidateModelInput()]
        self.transforms = T.Compose(transforms)

    def __call__(self, img1, img2, flow, valid):
        return self.transforms(img1, img2, flow, valid)
