from PIL import Image
import torch
import os
import scipy.io as sio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import random


class HSI_Segmentation(Dataset):
    def __init__(self, data_path: str = "", filter_path: str = "", img_type: str = "raw", transforms=None, train=True):
        """
        Parameters:
            data_path: the path of the "HSI Dataset folder"
            label_type: can be either "gray" or "viz"
            img_type: can be either "OSP" or "PCA"
            transforms: augmentation methods for images.
        """
        super(HSI_Segmentation, self).__init__()
        assert os.path.isdir(data_path), "path '{}' does not exist.".format(data_path)
        self.img_folder_list = os.listdir(data_path)
        self.img_type = img_type

        self.img_files = [os.path.join(data_path, img_folder, file)
                          for img_folder in self.img_folder_list
                          for file in os.listdir(os.path.join(data_path, img_folder))
                          if os.path.splitext(file)[-1].lower() == ".mat" and img_type in file]

        self.img_files.sort()
        self.img_files = self.img_files[:500] if train else self.img_files[500:]
        self.filter_file = filter_path
        self.filters = sio.loadmat(self.filter_file)["responsivity"].astype(np.float16)  # shape: (89, 71)
        self.filters = torch.as_tensor(self.filters)
        self.origin_img_files = [img_file.replace("ALL71channel", "raw")
                                 for img_file in self.img_files]

        self.transforms = transforms

        # Check if all files exist
        all_exist = True
        for path in self.img_files:
            if not os.path.isfile(path):
                all_exist = False
                assert all_exist, "path '{}' does not exist.".format(path)
                break
        
        all_exist = True
        for path in self.origin_img_files:
            if not os.path.isfile(path):
                all_exist = False
                assert all_exist, "path '{}' does not exist.".format(path)
                break

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, filter) where target is the image segmentation.
        """
        assert os.path.isfile(self.img_files[index]), "path '{}' does not exist.".format(self.img_files[index])
        img = sio.loadmat(self.img_files[index])["filtered_img"].astype(np.float16) \
            if self.img_type != "rgb" else Image.open(self.img_files[index])
        
        raw = sio.loadmat(self.origin_img_files[index])["data"].astype(np.float16) \
            if self.img_type != "rgb" else Image.open(self.origin_img_files[index])
        if self.transforms is not None:
            img = self.transforms(img)
            raw = self.transforms(raw)

        return img, raw

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def collate_fn(batch):
        images, raw_imgs = list(zip(*batch))
        lst_i = []
        lst_r = []
        for img, raw in zip(images, raw_imgs):
            lst_i.append(img.permute(1, 2, 0).contiguous())
            lst_r.append(raw.permute(1, 2, 0).contiguous())
            # list_img = img.permute(1, 2, 0).flatten(0, 1).contiguous().split(1) 
            # list_raw = raw.permute(1, 2, 0).flatten(0, 1).contiguous().split(1) 
            # for i, r in zip(list_img, list_raw):
            #     lst_i.append(i.squeeze())
            #     lst_r.append(r.squeeze())
        # pairs = random.sample(list(zip(lst_i, lst_r)), 100000)
        # lst_i, lst_r = zip(*pairs)
        flattened_imgs = torch.stack(lst_i)
        flattened_raws = torch.stack(lst_r)

        return flattened_imgs, flattened_raws
