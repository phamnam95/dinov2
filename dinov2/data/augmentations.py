# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms
import torch
import torch.nn.functional as F

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


class RandomResizedCrop3D(object):
    def __init__(self, output_size, scale=(0.32, 1.0)):
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)
        self.out_d, self.out_h, self.out_w = output_size
        self.scale = scale

    def __call__(self, x):
        # x: Tensor [C, D, H, W]
        assert isinstance(x, torch.Tensor) and x.dim() == 4
        _, D, H, W = x.shape
        s = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
        # Use cubic root of scale to preserve volume fraction across dims
        f = max(min(s ** (1.0 / 3.0), 1.0), 0.0)
        cd = max(int(D * f), 1)
        ch = max(int(H * f), 1)
        cw = max(int(W * f), 1)
        if cd > D: cd = D
        if ch > H: ch = H
        if cw > W: cw = W
        sd = 0 if D == cd else torch.randint(0, D - cd + 1, (1,)).item()
        sh = 0 if H == ch else torch.randint(0, H - ch + 1, (1,)).item()
        sw = 0 if W == cw else torch.randint(0, W - cw + 1, (1,)).item()
        x = x[:, sd:sd+cd, sh:sh+ch, sw:sw+cw]
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=(self.out_d, self.out_h, self.out_w), mode="trilinear", align_corners=False)
        x = x.squeeze(0)
        return x


class RandomFlip3D(object):
    def __init__(self, p=0.5, dims=(3,)):
        # dims in [2,3,4]: flipping axes for (D,H,W): depth=2,height=3,width=4 relative to [C,D,H,W]
        self.p = p
        self.dims = tuple(dims)

    def __call__(self, x):
        if torch.rand(()) < self.p:
            for d in self.dims:
                x = torch.flip(x, dims=(d,))
        return x


class GaussianBlur3D(object):
    def __init__(self, sigma=1.0, p=1.0, kernel_size=5):
        self.sigma = sigma
        self.p = p
        self.kernel_size = kernel_size

    def __call__(self, x):
        if torch.rand(()) >= self.p:
            return x
        # x: [C,D,H,W]
        device = x.device
        k = self.kernel_size
        coords = torch.arange(k, dtype=torch.float32, device=device) - (k - 1) / 2.0
        g = torch.exp(-(coords**2) / (2 * self.sigma**2))
        g = g / g.sum()
        g3 = (g[:, None, None] * g[None, :, None] * g[None, None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,k,k,k]
        weight = g3.repeat(x.shape[0], 1, 1, 1, 1)  # [C,1,k,k,k]
        x = x.unsqueeze(0)
        x = F.conv3d(x, weight, padding=k // 2, groups=weight.shape[0])
        x = x.squeeze(0)
        return x


class IntensityJitter3D(object):
    def __init__(self, brightness=0.4, contrast=0.4, p=0.8):
        self.b = brightness
        self.c = contrast
        self.p = p

    def __call__(self, x):
        if torch.rand(()) >= self.p:
            return x
        # brightness
        if self.b > 0:
            delta = (torch.rand((), device=x.device) * 2 - 1) * self.b
            x = x + delta
        # contrast
        if self.c > 0:
            factor = 1.0 + (torch.rand((), device=x.device) * 2 - 1) * self.c
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) * factor + mean
        return x


class DataAugmentationDINO3D(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=(32, 224, 224),
        local_crops_size=(16, 96, 96),
        flip_dims=(3,),
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        if isinstance(global_crops_size, int):
            global_crops_size = (global_crops_size, global_crops_size, global_crops_size)
        if isinstance(local_crops_size, int):
            local_crops_size = (local_crops_size, local_crops_size, local_crops_size)
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using 3D data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        self.geometric_augmentation_global = transforms.Compose(
            [RandomResizedCrop3D(global_crops_size, scale=global_crops_scale), RandomFlip3D(p=0.5, dims=flip_dims)]
        )
        self.geometric_augmentation_local = transforms.Compose(
            [RandomResizedCrop3D(local_crops_size, scale=local_crops_scale), RandomFlip3D(p=0.5, dims=flip_dims)]
        )

        intensity_jitter = IntensityJitter3D(brightness=0.4, contrast=0.4, p=0.8)
        global_transfo1_extra = GaussianBlur3D(p=1.0, sigma=1.0)
        global_transfo2_extra = transforms.Compose([GaussianBlur3D(p=0.1, sigma=1.0)])
        local_transfo_extra = GaussianBlur3D(p=0.5, sigma=1.0)

        def _normalize(x):
            # Expect input float in [0,1]; clip for safety
            return x.clamp(0.0, 1.0)

        self.normalize = _normalize
        self.global_transfo1 = transforms.Compose([intensity_jitter, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([intensity_jitter, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([intensity_jitter, local_transfo_extra, self.normalize])

    def __call__(self, volume_tensor):
        # volume_tensor: Tensor [C, D, H, W], values expected in [0,1]
        output = {}

        vol1_base = self.geometric_augmentation_global(volume_tensor)
        global_crop_1 = self.global_transfo1(vol1_base)
        vol2_base = self.geometric_augmentation_global(volume_tensor)
        global_crop_2 = self.global_transfo2(vol2_base)
        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        local_crops = [self.local_transfo(self.geometric_augmentation_local(volume_tensor)) for _ in range(self.local_crops_number)]
        output["local_crops"] = local_crops
        output["offsets"] = ()
        return output
