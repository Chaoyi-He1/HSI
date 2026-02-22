import random
from typing import List, Union
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
import torch
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target_list):
        for t in self.transforms:
            target_list_ = t(target_list)

        return target_list_


class ToTensor(object):
    def __call__(self, target_list):
        for i, target in enumerate(target_list):
            target_list[i] = F.to_tensor(target) if i != 1 else torch.as_tensor(target, dtype=torch.int64)
            if len(target_list[i].shape) == 2:
                target_list[i] = target_list[i].unsqueeze(0)
        return target_list
            
        # image = F.to_tensor(image)
        # target = torch.as_tensor(np.array(target), dtype=torch.int64)
        # target = target.unsqueeze(0) if len(target.shape) != 3 else target
        # return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize(object):
    def __init__(self, size: Union[int, List[int]], resize_mask: bool = True):
        self.size = size  # [h, w]
        self.resize_mask = resize_mask

    def __call__(self, image, target=None):
        image = F.resize(image, self.size)
        if self.resize_mask is True:
            target = F.resize(target, self.size)

        return image, target


class RandomCrop(object):
    def __init__(self, size: tuple):
        self.size = size

    def pad_if_smaller(self, img, fill=0):
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        # min_size = min(img.shape[-2:])
        if img.shape[1] < self.size[0] or img.shape[2] < self.size[1]:
            ow, oh = img.shape[1], img.shape[2]
            padh = self.size[1] - oh if oh < self.size[1] else 0
            padw = self.size[0] - ow if ow < self.size[0] else 0
            img = F.pad(img, [0, 0, padw, padh], fill=fill)
        return img

    def __call__(self, target_list):
        for i, target in enumerate(target_list):
            target_list[i] = self.pad_if_smaller(target)
        crop_params = T.RandomCrop.get_params(target_list[0], (self.size[0], self.size[1]))
        for i, target in enumerate(target_list):
            target_list[i] = F.crop(target, *crop_params)
        return target_list
        # image = self.pad_if_smaller(image)
        # target = self.pad_if_smaller(target)
        # crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        # image = F.crop(image, *crop_params)
        # target = F.crop(target, *crop_params)
        # return image, target
