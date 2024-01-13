import os
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class HSI_Segmentation(data.Dataset):
    def __init__(self, data_path: str = "", label_type: str = "gray", img_type: str = "OSP", transforms=None):
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

        if img_type != 'rgb':
            self.img_files = [os.path.join(data_path, img_folder, file)
                              for img_folder in self.img_folder_list
                              for file in os.listdir(os.path.join(data_path, img_folder))
                              if os.path.splitext(file)[-1].lower() == ".mat" and img_type in file]
        else:
            self.img_files = [os.path.join(data_path, img_folder, file)
                              for img_folder in self.img_folder_list
                              for file in os.listdir(os.path.join(data_path, img_folder))
                              if os.path.splitext(file)[-1].lower() == ".jpg" and img_type in file]
            
        self.img_files.sort()
        rgb = "rgb" if img_type != 'rgb' else ''
        self.mask_files = [img.replace(img.split(os.sep)[-1], rgb
                                       + os.path.splitext(os.path.basename(img))[0].replace("_OSP10channel", "")
                                       + "_gray.png") for img in self.img_files]
        # self.mask_files = [img.replace(img.split(os.sep)[-1], "label_" + label_type
        #                                + ".png") for img in self.img_files]
        
        assert (len(self.img_files) == len(self.mask_files))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = sio.loadmat(self.img_files[index])["filtered_img"].astype(np.float16) \
            if self.img_type != "rgb" else Image.open(self.img_files[index])
        # img = np.ascontiguousarray(img.transpose(2, 0, 1))
        # img = (img - np.min(img)) * 255 / np.max(img)
        # img = Image.fromarray(img)
        target = Image.open(self.mask_files[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        channel = batched_imgs.shape[1]
        pos = (batched_targets != 255)[0, 0, :, :]
        batched_imgs = batched_imgs[..., pos]
        batched_targets = batched_targets[..., pos]
        batched_imgs = batched_imgs.permute(0, 2, 1).contiguous().squeeze()
        batched_targets = batched_targets.permute(0, 2, 1).contiguous().squeeze()
        return batched_imgs, batched_targets


class HSI_Transformer(data.Dataset):
    def __init__(self, data_path: str = "", label_type: str = "gray", img_type: str = "OSP", 
                 sequence_length: int = 10):
        """
        Parameters:
            data_path: the path of the "HSI Dataset folder"
            label_type: can be either "gray" or "viz"
            img_type: can be either "OSP" or "PCA"
            transforms: augmentation methods for images.
        """
        super(HSI_Transformer, self).__init__()
        assert os.path.isdir(data_path), "path '{}' does not exist.".format(data_path)
        self.img_folder_list = os.listdir(data_path)
        self.img_type = img_type
        self.sequence_length = sequence_length

        if img_type != 'rgb':
            self.img_files = [os.path.join(data_path, img_folder, file)
                              for img_folder in self.img_folder_list
                              for file in os.listdir(os.path.join(data_path, img_folder))
                              if os.path.splitext(file)[-1].lower() == ".mat" and img_type in file]
        else:
            self.img_files = [os.path.join(data_path, img_folder, file)
                              for img_folder in self.img_folder_list
                              for file in os.listdir(os.path.join(data_path, img_folder))
                              if os.path.splitext(file)[-1].lower() == ".jpg" and img_type in file]
            
        self.img_files.sort()
        rgb = "rgb" if img_type == 'rgb' else ''
        self.mask_files = [img.replace(img.split(os.sep)[-1],
                                       os.path.splitext(
                                           os.path.basename(img))[0].replace("_OSP10channel", "").replace(rgb, "")
                                       + "_" + label_type + '.mat') for img in self.img_files]
        # self.mask_files = [img.replace(img.split(os.sep)[-1], "label_" + label_type
        #                                + ".png") for img in self.img_files]
        
        assert (len(self.img_files) == len(self.mask_files))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = sio.loadmat(self.img_files[index])["filtered_img"].astype(np.float16) \
            if self.img_type != "rgb" else Image.open(self.img_files[index])
        # img = np.ascontiguousarray(img.transpose(2, 0, 1))
        # img = (img - np.min(img)) * 255 / np.max(img)
        # img = Image.fromarray(img)
        target = sio.loadmat(self.mask_files[index])["overlay"].astype(int)

        img = torch.from_numpy(img)
        target = torch.from_numpy(target)
        
        img = img.reshape(-1, self.sequence_length, img.shape[2]).permute(2, 1, 0).contiguous()
        target = target.reshape(-1, self.sequence_length, target.shape[2]).permute(2, 1, 0).contiguous()
        return img, target

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = torch.stack(images, dim=0).flatten(start_dim=0, end_dim=1)
        batched_targets = torch.stack(targets, dim=0).flatten(start_dim=0, end_dim=1)
        return batched_imgs[batched_targets != 255], batched_targets[batched_targets != 255]
    