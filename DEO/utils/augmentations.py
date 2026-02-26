"""
Data augmentation utilities for DINO training.
"""

import random

import kornia.augmentation as K
import torch
from PIL import Image
from torchvision import transforms


class GaussianBlurKornia(object):
    """
    Apply Gaussian Blur to a torch.Tensor using Kornia (no PIL conversion).
    """

    def __init__(self, p=0.5, kernel_size=23, sigma=(0.1, 2.0)):
        self.prob = p
        self.blur = K.RandomGaussianBlur(kernel_size=kernel_size, sigma=sigma, p=1.0)

    def __call__(self, img):
        if random.random() < self.prob:
            # img: Tensor, shape [C, H, W]
            if img.dim() == 3:
                img = img.unsqueeze(0)  # [1, C, H, W]
            out = self.blur(img)
            return out.squeeze(0)
        else:
            return img


class SolarizationKornia(object):
    """
    Apply Solarization to a torch.Tensor using Kornia (no PIL conversion).
    """

    def __init__(self, p=0.5, threshold=0.5):
        self.p = p
        self.threshold = threshold
        self.solarize = K.RandomSolarize(thresholds=threshold, p=1.0)

    def __call__(self, img):
        if random.random() < self.p:
            # img: Tensor, shape [C, H, W]
            if img.dim() == 3:
                img = img.unsqueeze(0)  # [1, C, H, W]
            out = self.solarize(img)
            return out.squeeze(0)
        else:
            return img


class DataAugmentationDINO(object):
    """
    Data augmentation for DINO training with RGB and optional multispectral support.
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        dist_arch,
        ms_arch,
    ):
        if ms_arch:
            flip_and_color_jitter = transforms.RandomHorizontalFlip(p=0.5)
        else:
            flip_and_color_jitter = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                            )
                        ],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                ]
            )

        # normalization
        if dist_arch == "ms":
            normalize = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (
                            1184.382,
                            1120.771,
                            1136.260,
                        ),
                        (
                            650.284,
                            712.125,
                            965.231,
                        ),
                    ),
                ]
            )
        elif ms_arch:
            normalize = transforms.Normalize(
                (
                    # original order
                    # 1184.382,
                    # 1120.771,
                    # 1136.260,
                    # reversed order
                    1136.26026392,
                    1120.77120066,
                    1184.3824625,
                    1263.73947144,
                    1645.40315151,
                    1846.87040806,
                    1762.59530783,
                    1972.62420416,
                    1732.16362238,
                    1247.91870117,
                ),
                (
                    # original order
                    # 650.2842772,
                    # 712.12507725,
                    # 965.23119807,
                    # reversed order
                    965.23119807,
                    712.12507725,
                    650.2842772,
                    948.9819932,
                    1108.06650639,
                    1258.36394548,
                    1233.1492281,
                    1364.38688993,
                    1310.36996126,
                    1087.6020813,
                ),
            )

        else:
            normalize = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

        # Constants for crop sizes
        GLOBAL_CROP_SIZE = 224
        LOCAL_CROP_SIZE = 96

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    GLOBAL_CROP_SIZE,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                # utils.GaussianBlur(1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    GLOBAL_CROP_SIZE,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                # utils.GaussianBlur(0.1),
                # utils.Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    LOCAL_CROP_SIZE,
                    scale=local_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                # utils.GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class DataAugmentationDINOMS(object):
    """
    Data augmentation for multispectral DINO training.
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
    ):
        EPS = 1e-6  # Small constant for numerical stability
        self.EPS = EPS

        flip_all = transforms.RandomHorizontalFlip(p=0.5)

        color_jitter_and_blur_rgb_global = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        color_jitter_and_blur_rgb_local = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        # normalization
        self.normalize_hr = transforms.Normalize(
            (
                0.4182007312774658,
                0.4214799106121063,
                0.3991275727748871,
                1263.73947144,
                1645.40315151,
                1846.87040806,
                1762.59530783,
                1972.62420416,
                1732.16362238,
                1247.91870117,
            ),
            (
                0.28774282336235046,
                0.27541765570640564,
                0.2764017581939697,
                948.9819932,
                1108.06650639,
                1258.36394548,
                1233.1492281,
                1364.38688993,
                1310.36996126,
                1087.6020813,
            ),
        )

        self.normalize_s2 = transforms.Normalize(
            (
                # original order
                # 1184.382,
                # 1120.771,
                # 1136.260,
                # reversed order
                1136.26026392,
                1120.77120066,
                1184.3824625,
                1263.73947144,
                1645.40315151,
                1846.87040806,
                1762.59530783,
                1972.62420416,
                1732.16362238,
                1247.91870117,
            ),
            (
                # original order
                # 650.2842772,
                # 712.12507725,
                # 965.23119807,
                # reversed order
                965.23119807,
                712.12507725,
                650.2842772,
                948.9819932,
                1108.06650639,
                1258.36394548,
                1233.1492281,
                1364.38688993,
                1310.36996126,
                1087.6020813,
            ),
        )

        # Constants for crop sizes
        GLOBAL_CROP_SIZE = 224
        LOCAL_CROP_SIZE = 96

        # first global ms crop
        self.global_transfo_all = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    GLOBAL_CROP_SIZE,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_all,
            ]
        )

        # first global rgb transform
        self.global_transfo1_rgb = transforms.Compose(
            [
                color_jitter_and_blur_rgb_global,
                GaussianBlurKornia(1.0),
            ]
        )

        # second global rgb transform
        self.global_transfo2_rgb = transforms.Compose(
            [
                color_jitter_and_blur_rgb_global,
                GaussianBlurKornia(0.1),
                SolarizationKornia(0.2),
            ]
        )

        # transformation for the local small crops
        self.local_crops_number = local_crops_number

        self.local_transfo_all = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    LOCAL_CROP_SIZE,
                    scale=local_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_all,
            ]
        )

        self.local_transfo_rgb = transforms.Compose(
            [
                color_jitter_and_blur_rgb_local,
                GaussianBlurKornia(p=0.5),
            ]
        )

    def __call__(self, image):
        crops_list = []

        crops = self.global_transfo_all(image)  # first global crop

        orig_scale = crops[:3].max().item()
        crops_rgb = (
            crops[:3] / orig_scale if orig_scale > self.EPS else crops[:3]
        )  # scale RGB to [0, 1]
        crops_rgb = self.global_transfo1_rgb(
            crops_rgb.clamp(0.0, 1.0)
        )  # first RGB augs for first global crop

        crops = torch.cat((crops_rgb, crops[3:]), dim=0)
        crops = self.normalize_hr(crops)
        crops_list.append(crops)

        crops = self.global_transfo_all(image)  # second global crop

        orig_scale = crops[:3].max().item()
        crops_rgb = (
            crops[:3] / orig_scale if orig_scale > self.EPS else crops[:3]
        )  # scale RGB to [0, 1]
        crops_rgb = self.global_transfo2_rgb(
            crops_rgb.clamp(0.0, 1.0)
        )  # second RGB augs for second global crop

        crops = torch.cat((crops_rgb, crops[3:]), dim=0)
        crops = self.normalize_hr(crops)
        crops_list.append(crops)

        # local crops, mirrors global crops
        for _ in range(self.local_crops_number):
            crops = self.local_transfo_all(image)

            orig_scale = crops[:3].max().item()
            crops_rgb = crops[:3] / orig_scale if orig_scale > self.EPS else crops[:3]
            crops_rgb = self.local_transfo_rgb(crops_rgb.clamp(0.0, 1.0))

            crops = torch.cat((crops_rgb, crops[3:]), dim=0)
            crops = self.normalize_hr(crops)
            crops_list.append(crops)
        return crops_list
