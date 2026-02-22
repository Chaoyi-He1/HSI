import os
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Subset
from collections import defaultdict


class HSI_Drive_V1_(data.Dataset):
    def __init__(self, data_path: str = "", use_MF: bool = True, use_dual: bool = True,
                 use_OSP: bool = True, use_raw: bool = False, use_rgb: bool = False, use_cache: bool = False,
                 use_attention: bool = False, use_large_mlp: bool = False, num_attention: int = 10,
                 R_idx: int = 1, for_train: bool = True):
        self.use_MF = use_MF
        self.use_dual = use_dual
        self.use_OSP = use_OSP
        self.use_raw = use_raw
        self.use_rgb = use_rgb
        self.use_cache = use_cache
        self.R_idx = R_idx
        self.for_train = for_train
        
        self.use_attention = use_attention
        self.use_large_mlp = use_large_mlp
        self.num_attention = num_attention
        
        self.data_folder_path = os.path.join(data_path, "cubes_float32")
        
        path_ext = ""
        name_ext = "_MF_TC_N_fl32"
        
        self.data_paths = [os.path.join(self.data_folder_path, file) for file in os.listdir(self.data_folder_path) if file.endswith(".mat")]
        self.label_paths = [file.replace("cubes_float32", "labels").replace(path_ext, "").replace(name_ext, "").replace(".mat", ".png") for file in self.data_paths]
        self.rgb_paths = [file.replace("cubes_float32", "RGB").replace(path_ext, "").replace(name_ext, "_color").replace(".mat", ".png") for file in self.data_paths]
        self.end_label_paths = [file.replace("cubes_float32", "end_labels").replace(path_ext, "").replace(name_ext, "").replace(".mat", ".png") for file in self.data_paths]
        
        # Load R matrix from "/data2/chaoyi/HSI_Dataset/HSI Drive/v1/R_full_32devices/R_full_device_xx.mat"
        if self.for_train:
            self.R = []
            for i in range(1, 33):
                print(f"Loading R matrix for device {i}...")
                R_matrix_path = os.path.join(data_path, "R_full_32devices", f"R_full_device_{i:02d}.mat")
                self.R.append(sio.loadmat(R_matrix_path)["R"]) # shape 401 x 252
                
                # pre-process R matrix, follow the steps in "HSI_Drive_Filtering_05_01.m"
                self.R[i-1] = np.abs(self.R[i-1])
                self.R[i-1] = self.R[i-1] / self.R[i-1].max(axis=0, keepdims=True)
                self.R[i-1] = self.R[i-1][39:400:15, :]  # rows 40,55,...,400 in MATLAB
                assert self.R[i-1].shape == (25, 252), "R matrix shape is not correct after preprocessing."
            self.R = np.stack(self.R, axis=0)  # shape (32, 25, 252)
        else:
            R_matrix_path = os.path.join(data_path, "R_full_32devices", f"R_full_device_{self.R_idx:02d}.mat")
            self.R = sio.loadmat(R_matrix_path)["R"] # shape 401 x 252
            
            # pre-process R matrix, follow the steps in "HSI_Drive_Filtering_05_01.m"
            self.R = np.abs(self.R)
            self.R = self.R / self.R.max(axis=0, keepdims=True)
            self.R = self.R[39:400:15, :]  # rows 40,55,...,400 in MATLAB
            assert self.R.shape == (25, 252), "R matrix shape is not correct after preprocessing."
        
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
        data_dict = defaultdict(list)
        for i in range(len(self.data_paths)):
            if not self.use_rgb:
                img = sio.loadmat(self.data_paths[i])["cube_fl32"]
                img = img.transpose(1, 2, 0)
            else:
                img = np.array(Image.open(self.rgb_paths[i]))
            
            label = np.array(Image.open(self.label_paths[i]))
            end_label = np.array(Image.open(self.end_label_paths[i]))
            label = self.relabeling(label, end_label)
            
            img = img.reshape(-1, img.shape[-1])    # reshape to (num_pixels, 25)
            label = label.reshape(-1)
            
            # remove 255 label
            img = img[label != 255, :]
            label = label[label != 255]
            
            if self.for_train:
                rand_idx = np.random.randint(0, self.R.shape[0], size=img.shape[0])
                R_mtx = self.R[rand_idx]  # shape (num_pixels, 25, 252)
                img = np.einsum('ij,ijk->ik', img, R_mtx) / 1e3  # shape (num_pixels, 252)
            else:
                img = img @ self.R / 1e3  # shape (num_pixels, 252)
            
            if self.use_OSP and not self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_OSP and self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_attention and self.use_dual and not self.use_raw and not self.use_rgb:
                attention_index = [229, 180, 6, 162, 168, 32, 24, 13, 1, 3] if self.use_large_mlp \
                    else [223, 163, 58, 230, 40, 193, 145, 187, 248, 181]
                img = img[:, attention_index[:self.num_attention]]
            
            for p in range(img.shape[0]):
                data_dict[label[p]].append(img[p])
        # print(len(data_dict[7]))
        # randomly select 5000 samples from each class
        data_dict_selected = {}
        for k, v in data_dict.items():
            v = np.array(v)
            if len(v) > 50000:
                data_dict_selected[k] = v[np.random.choice(len(v), 50000, replace=False), :]
            else:
                data_dict_selected[k] = v
        del data_dict
        # generate the data and label lists based on the data_dict
        self.data_list = []
        self.label_list = []
        for k, v in data_dict_selected.items():
            self.data_list.extend(v)
            self.label_list.extend([k] * len(v))
    
    def __getitem__(self, index):
        if not self.use_cache:
            if not self.use_rgb:
                img = sio.loadmat(self.data_paths[index])["cube_fl32"]
                img = img.transpose(1, 2, 0) # shape H x W x 25
            else:
                img = np.array(Image.open(self.rgb_paths[index]))
                
            label = np.array(Image.open(self.label_paths[index]))
            end_label = np.array(Image.open(self.end_label_paths[index]))
            label = self.relabeling(label, end_label)
            img_pos = np.indices(img.shape[:2]).transpose(1, 2, 0)
            
            img = img.reshape(-1, img.shape[-1])
            img_pos = img_pos.reshape(-1, 2)
            label = label.reshape(-1)
            
            if self.for_train:
                rand_idx = np.random.randint(0, self.R.shape[0], size=img.shape[0])
                R_mtx = self.R[rand_idx]  # shape (num_pixels, 25, 252)
                img = np.einsum('ij,ijk->ik', img, R_mtx) / 1e3  # shape (num_pixels, 252)
            else:
                img = img @ self.R / 1e3  # shape (num_pixels, 252)
            
            # if self.use_OSP and not self.use_dual and not self.use_raw and not self.use_rgb:
            #     OSP_index = [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]
            #     img = img[:, OSP_index[:self.num_attention]]
            # elif self.use_OSP and self.use_dual and not self.use_raw and not self.use_rgb:
            #     OSP_index = [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]
            #     img = img[:, OSP_index[:self.num_attention]]
            # elif self.use_attention and self.use_dual and not self.use_raw and not self.use_rgb:
            #     attention_index = [163, 40, 58, 218, 4, 230, 76, 121, 176, 224] if self.use_large_mlp \
            #         else [223, 163, 58, 230, 40, 193, 145, 187, 248, 181]
            #     img = img[:, attention_index[:self.num_attention]]
            
            img = torch.from_numpy(img).to(dtype=torch.float32)
            label = torch.from_numpy(label).to(dtype=int)
        elif self.use_cache:
            img = torch.from_numpy(self.data_list[index]).to(dtype=torch.float32)
            label = torch.tensor(self.label_list[index], dtype=torch.int64)
            img_pos = None
        # rescale img to uint8
        # img = (img - img.min()) / (img.max() - img.min()) * 255.0
        # img = img.to(dtype=torch.uint8).to(dtype=torch.float32)
        return img, label, img_pos
    
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
