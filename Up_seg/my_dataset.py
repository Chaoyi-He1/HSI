import os
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Subset
from collections import defaultdict
import torch.nn.functional as F


class HSI_Drive_V1_(data.Dataset):
    def __init__(self, data_path: str = "", use_MF: bool = True, use_dual: bool = True,
                 use_OSP: bool = True, use_raw: bool = False, use_rgb: bool = False, use_cache: bool = False,
                 use_attention: bool = False, use_large_mlp: bool = False, num_attention: int = 10,
                 transform=None, down_sample_rate: int = 2):
        self.use_MF = use_MF
        self.use_dual = use_dual
        self.use_OSP = use_OSP
        self.use_raw = use_raw
        self.use_rgb = use_rgb
        self.use_cache = use_cache
        self.down_sample_rate = down_sample_rate
        
        self.use_attention = use_attention
        self.use_large_mlp = use_large_mlp
        self.num_attention = num_attention
        
        self.transform = transform
        
        self.data_folder_path = os.path.join(data_path, "cubes_float32")
        if not use_raw:
            self.data_folder_path = os.path.join(self.data_folder_path, "Dual_HVI") if use_dual else os.path.join(self.data_folder_path, "Sin_HVI")
        
        path_ext = ""
        
        if use_raw:
            path_ext += ""
        elif use_dual and not use_raw:
            path_ext += "/Dual_HVI"
        elif not use_dual and not use_raw:
            path_ext += "/Sin_HVI"
        
        name_ext = "_MF_TC_N_fl32"
        
        self.data_paths = [os.path.join(self.data_folder_path, file) for file in os.listdir(self.data_folder_path) if file.endswith(".mat")]
        self.label_paths = [file.replace("cubes_float32", "labels").replace(path_ext, "").replace(name_ext, "").replace(".mat", ".png") for file in self.data_paths]
        self.rgb_paths = [file.replace("cubes_float32", "RGB").replace(path_ext, "").replace(name_ext, "_color").replace(".mat", ".png") for file in self.data_paths]
        self.end_label_paths = [file.replace("cubes_float32", "end_labels").replace(path_ext, "").replace(name_ext, "").replace(".mat", ".png") for file in self.data_paths]
        
        for i in range(len(self.data_paths) - 1, -1, -1):
            if not os.path.isfile(self.label_paths[i]):
                del self.data_paths[i]
                del self.label_paths[i]
        
        assert len(self.data_paths) == len(self.label_paths) and len(self.data_paths) > 0, "The number of data files and label files are not equal."

        self.hsi_drive_original_label = {
            0: "Unlabeled",
            1: "Road",
            2: "Road marks",
            3: "Vegetation",
            4: "Painted Metal",
            5: "Sky",
            6: "Concrete/Stone/Brick",
            7: "Pedestrian/Cyclist",
            8: "Water",
            9: "Unpainted Metal",
            10: "Glass/Transparent Plastic",
        }
        self.selected_labels = [1, 2, 3, 4, 5, 6, 7, 9, 10]
        if use_cache:
            self.cache_data()

        
    def relabeling(self, label, end_label):
        for k, v in self.hsi_drive_original_label.items():
            if k not in self.selected_labels:
                label[label == k] = 255
        # relabel the label from 0 to end, with 255 as the background
        for i, k in enumerate(self.selected_labels):
            # relabel the label matrix from 0 to end where in end_label has the same label, otherwise, label it as 255
            # if end_label and label have no common label, then dont consider end_label
            if k not in end_label:
                label[label == k] = i
            else:
                relabel_index = (end_label == k) & (label == k)
                label[relabel_index] = i
                ignore_index = (end_label != k) & (label == k)
                label[ignore_index] = 255
        return label
    
    def cache_data(self):
        data_dict = []
        for i in range(len(self.data_paths)):
            if not self.use_rgb:
                img = sio.loadmat(self.data_paths[i])["filtered_img"] if not self.use_raw else sio.loadmat(self.data_paths[i])["cube_fl32"]
                img = img.transpose(1, 2, 0) if self.use_raw else img / 1e3
            else:
                img = np.array(Image.open(self.rgb_paths[i]))
            
            label = np.array(Image.open(self.label_paths[i]))
            end_label = np.array(Image.open(self.end_label_paths[i]))
            label = self.relabeling(label, end_label)
            img_pos = np.indices(img.shape[:2]).transpose(1, 2, 0)  
            
            lists = self.transform([img, label, img_pos]) if self.transform else (img, label, img_pos)
            img, label, img_pos = lists[0], lists[1], lists[2]
            
            if self.use_OSP and not self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]
                img = img[OSP_index[:self.num_attention], ...]
            elif self.use_OSP and self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]
                img = img[OSP_index[:self.num_attention], ...]
            elif self.use_attention and self.use_dual and not self.use_raw and not self.use_rgb:
                attention_index = [163, 40, 58, 218, 4, 230, 76, 121, 176, 224] if self.use_large_mlp \
                    else [223, 163, 58, 230, 40, 193, 145, 187, 248, 181]
                img = img[attention_index[:self.num_attention], ...]
            
            data_dict.append((img, label, img_pos))
        # print(len(data_dict[7]))
        # randomly select 5000 samples from each class
        data_dict_selected = []
        if len(data_dict) > 50000:
            data_dict_selected = np.random.choice(data_dict, size=50000, replace=False).tolist()
        else:
            data_dict_selected = data_dict
        del data_dict
        
        self.data_list = data_dict_selected
        # get all unique labels in the dataset and check if all selected labels are present
        unique_labels = set()
        for _, label, _ in self.data_list:
            unique_labels.update(np.unique(label))
        assert len(unique_labels) == 10, \
            f"Not all selected labels {self.selected_labels} are present in the dataset. Found labels: {unique_labels}"
        print(f"Cached {len(self.data_list)} samples from {len(self.data_paths)} original data files.")
    
    def __getitem__(self, index):
        if not self.use_cache:
            if not self.use_rgb:
                img = sio.loadmat(self.data_paths[index])["filtered_img"] if not self.use_raw else sio.loadmat(self.data_paths[index])["cube_fl32"]
                img = img.transpose(1, 2, 0) if self.use_raw else img / 1e3
            else:
                img = np.array(Image.open(self.rgb_paths[index]))
            
            label = np.array(Image.open(self.label_paths[index]))
            end_label = np.array(Image.open(self.end_label_paths[index]))
            label = self.relabeling(label, end_label)
            img_pos = np.indices(img.shape[:2]).transpose(1, 2, 0)  
            
            lists = self.transform([img, label, img_pos]) if self.transform else (img, label, img_pos)
            img, label, img_pos = lists[0], lists[1], lists[2]
            
            if self.use_OSP and not self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]
                img = img[OSP_index[:self.num_attention], ...]
            elif self.use_OSP and self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]
                img = img[OSP_index[:self.num_attention], ...]
            elif self.use_attention and self.use_dual and not self.use_raw and not self.use_rgb:
                attention_index = [163, 40, 58, 218, 4, 230, 76, 121, 176, 224] if self.use_large_mlp \
                    else [223, 163, 58, 230, 40, 193, 145, 187, 248, 181]
                img = img[attention_index[:self.num_attention], ...]

            img = img.to(dtype=torch.float16)
            label = label.to(dtype=torch.int64)
            img_pos = img_pos if img_pos is not None else None
            
        elif self.use_cache:
            img = self.data_list[index][0].to(dtype=torch.float16)
            label = self.data_list[index][1].to(dtype=torch.int64)
            img_pos = self.data_list[index][2]

        # down sample the image and label
        img, label, img_pos = self.down_sample(img, label, img_pos, down_sample_rate=self.down_sample_rate)
        return img, label, img_pos

    def down_sample(self, img, label, img_pos, down_sample_rate):
        '''
        Down sample the image and label by the given rate.
        For each N x N pixel block, get the average value of the block,
        and use the average value as the new pixel value.
            N = img.H / 2^down_sample_rate
        Arguments:
            img: (C, H, W), tensor
            label: (H, W), tensor
            img_pos: (H, W, 2), tensor or None
            down_sample_rate: int
        '''
        # down sample the image based on the kernel size
        img = F.avg_pool2d(img, kernel_size=2 ** down_sample_rate, stride=2 ** down_sample_rate)
        # Downsample label by taking the union of all labels in each block (excluding 255)
        block_size = 2 ** down_sample_rate
        h, w = label.shape
        out_h, out_w = h // block_size, w // block_size
        new_label = torch.zeros((out_h, out_w, 9), dtype=torch.int64, device=label.device)
        # Reshape to (out_h, block_size, out_w, block_size)
        label_blocks = label[:out_h * block_size, :out_w * block_size].reshape(out_h, block_size, out_w, block_size)
        # Flatten each block to (out_h, out_w, block_size*block_size)
        label_blocks = label_blocks.permute(0, 2, 1, 3).reshape(out_h, out_w, -1)
        for k in range(9):
            mask = (label_blocks == k).any(dim=2)
            new_label[:, :, k] = mask

        return img, new_label, img_pos

    def __len__(self):
        return len(self.data_paths) if not self.use_cache else len(self.data_list)
    
    @staticmethod
    def collate_fn(batch):
        images, targets, img_pos = list(zip(*batch))
        batched_imgs = torch.stack(images, dim=0)
        batched_imgs = batched_imgs.flatten(0, 1) if len(batched_imgs.shape) == 3 else batched_imgs
        batched_targets = torch.stack(targets, dim=0).to(dtype=torch.int64)
        batched_targets = batched_targets.flatten(0, 1) if len(batched_targets.shape) == 2 else batched_targets
        batched_img_pos = np.vstack(img_pos) if img_pos[0] is not None else None
        # print max and min of label ignoring 255
        # print(f"Max label: {torch.max(batched_targets[batched_targets != 255])}, Min label: {torch.min(batched_targets[batched_targets != 255])}")
        return batched_imgs, batched_targets, batched_img_pos