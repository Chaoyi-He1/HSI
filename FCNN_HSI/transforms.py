import random
from typing import List, Union
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
import torch
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, img_pos):
        for t in self.transforms:
            image, target = t(image, target, img_pos)

        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.flip_prob = prob

    def __call__(self, image, target, img_pos):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
            img_pos = F.hflip(img_pos)
        return image, target, img_pos


class Resize(object):
    def __init__(self, size: Union[int, List[int]], resize_mask: bool = True):
        self.size = size  # [h, w]
        self.resize_mask = resize_mask

    def __call__(self, image, target, img_pos):
        image = F.resize(image, self.size)
        if self.resize_mask is True:
            target = F.resize(target, self.size)
        img_pos = F.resize(img_pos, self.size)

        return image, target, img_pos


class RandomCrop(object):
    def __init__(self, size: int):
        self.size = size

    def pad_if_smaller(self, img, fill=0):
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        min_size = min(img.shape[-2:])
        if min_size < self.size:
            ow, oh = img.size
            padh = self.size - oh if oh < self.size else 0
            padw = self.size - ow if ow < self.size else 0
            img = F.pad(img, [0, 0, padw, padh], fill=fill)
        return img

    def __call__(self, image, target, img_pos):
        image = self.pad_if_smaller(image)
        target = self.pad_if_smaller(target)
        img_pos = self.pad_if_smaller(img_pos, fill=-1)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        img_pos = F.crop(img_pos, *crop_params)
        return image, target, img_pos
