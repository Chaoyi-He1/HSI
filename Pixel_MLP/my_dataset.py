import os
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Subset
from collections import defaultdict


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
        self.label_type = label_type
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
        channel = "_ALL71channel" if img_type == 'ALL' else "_OSP10channel"
        self.mask_files = [img.replace(img.split(os.sep)[-1],
                                       os.path.splitext(
                                           os.path.basename(img))[0].replace(channel, "").replace(rgb, "")
                                       + "_" + label_type + '.mat') for img in self.img_files]
        rgb = "rgb" if img_type != 'rgb' else ''
        self.label_mask = [img.replace(img.split(os.sep)[-1], rgb
                                       + os.path.splitext(os.path.basename(img))[0].replace(channel, "")
                                       + "_gray.png") for img in self.img_files]
        # self.mask_files = [img.replace(img.split(os.sep)[-1], "label_" + label_type
        #                                + ".png") for img in self.img_files]
        self.sanity_check()
        self.label_mapping = {
            "road": 0,
            "sidewalk": 1,
            "building": 2,
            "wall": 3,
            "fence": 4,
            "pole": 5,
            "traffic light": 6,
            "traffic sign": 7,
            "tree": 8,
            "terrain": 9,
            "sky": 10,
            "person": 11,
            "rider": 12,
            "car": 13,
            "truck": 14,
            "bus": 15,
            "train": 16,
            "motorcycle": 17,
            "bicycle": 18,
            "background": 19,
        }
    
    def sanity_check(self):
        # if the mask file is not exist, then remove the corresponding image file and label file and mask file from the list
        for i in range(len(self.img_files) - 1, -1, -1):
            if not os.path.isfile(self.mask_files[i]):
                del self.img_files[i]
                del self.mask_files[i]
                del self.label_mask[i]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        label_index = next((i for i, k in enumerate(self.label_mapping.keys()) if k in self.label_type.lower()), None)
        pixel_index = Image.open(self.label_mask[index])
        pixel_index = np.array(pixel_index) == label_index
        
        img = sio.loadmat(self.img_files[index])["filtered_img"].astype(np.float16) \
            if self.img_type != "rgb" else np.array(Image.open(self.img_files[index])).astype(np.float16)
        img = img[pixel_index, :]
        img = img[: img.shape[0] // self.sequence_length * self.sequence_length, :]
        
        # check if img is empty
        if img.shape[0] == 0:
            return self.__getitem__(np.random.randint(0, len(self.img_files)))
        # img = np.ascontiguousarray(img.transpose(2, 0, 1))
        # img = (img - np.min(img)) * 255 / np.max(img)
        # img = Image.fromarray(img)
        target = sio.loadmat(self.mask_files[index])["overlay"].astype(np.float16)
        target = target[pixel_index] 
        target = target[: target.shape[0] // self.sequence_length * self.sequence_length]
           
        img = torch.from_numpy(img)
        target = torch.from_numpy(target)
        
        img = img.view(-1, self.sequence_length, img.shape[1]).contiguous()
        target = target.view(-1, self.sequence_length).contiguous()
        return img, target

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = torch.stack(images, dim=0).flatten(0, 1)
        batched_targets = torch.stack(targets, dim=0).flatten(0, 1)
        # if batched_imgs.shape[0] > 50000, randomly select 50000 samples
        if batched_imgs.shape[0] > 50000:
            index = np.random.choice(batched_imgs.shape[0], 50000, replace=False)
            batched_imgs = batched_imgs[index]
            batched_targets = batched_targets[index]
        return batched_imgs, batched_targets


class HSI_Transformer_all(data.Dataset):
    def __init__(self, data_path: str = "", label_type: str = "gray", img_type: str = "OSP", 
                 sequence_length: int = 10):
        """
        Parameters:
            data_path: the path of the "HSI Dataset folder"
            label_type: can be either "gray" or "viz"
            img_type: can be either "OSP" or "PCA"
            transforms: augmentation methods for images.
        """
        super(HSI_Transformer_all, self).__init__()
        assert os.path.isdir(data_path), "path '{}' does not exist.".format(data_path)
        self.img_folder_list = os.listdir(data_path)
        self.img_type = img_type
        self.label_type = label_type
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
        if img_type == "ALL":
            channel = "_ALL71channel" 
        elif img_type == "OSP":
            channel = "_OSP10channel"
        else:
            channel = "_RAW"
        self.mask_files = [[img.replace(img.split(os.sep)[-1],
                                       os.path.splitext(
                                           os.path.basename(img))[0].replace(channel, "").replace(rgb, "")
                                       + "_" + "Roadlabel" + '.mat'),
                            img.replace(img.split(os.sep)[-1],
                                       os.path.splitext(
                                           os.path.basename(img))[0].replace(channel, "").replace(rgb, "")
                                       + "_" + "Building_Concrete_label" + '.mat'),
                            img.replace(img.split(os.sep)[-1],
                                       os.path.splitext(
                                           os.path.basename(img))[0].replace(channel, "").replace(rgb, "")
                                       + "_" + "Building_Glass_label" + '.mat'),
                            img.replace(img.split(os.sep)[-1],
                                       os.path.splitext(
                                           os.path.basename(img))[0].replace(channel, "").replace(rgb, "")
                                       + "_" + "Car_white_label" + '.mat'),
                            img.replace(img.split(os.sep)[-1],
                                       os.path.splitext(
                                           os.path.basename(img))[0].replace(channel, "").replace(rgb, "")
                                       + "_" + "Treelabel" + '.mat'),
                            img.replace(img.split(os.sep)[-1],
                                       os.path.splitext(
                                           os.path.basename(img))[0].replace(channel, "").replace(rgb, "")
                                       + "_" + "Skylabel" + '.mat'),] for img in self.img_files]
        rgb = "rgb" if img_type != 'rgb' else ''
        self.label_mask = [img.replace(img.split(os.sep)[-1], rgb
                                       + os.path.splitext(os.path.basename(img))[0].replace(channel, "")
                                       + "_gray.png") for img in self.img_files]
        # self.mask_files = [img.replace(img.split(os.sep)[-1], "label_" + label_type
        #                                + ".png") for img in self.img_files]
        self.sanity_check()
        self.label_mapping = {
            "road": 0,
            "sidewalk": 1,
            "building": 2,
            "wall": 3,
            "fence": 4,
            "pole": 5,
            "traffic light": 6,
            "traffic sign": 7,
            "tree": 8,
            "terrain": 9,
            "sky": 10,
            "person": 11,
            "rider": 12,
            "car": 13,
            "truck": 14,
            "bus": 15,
            "train": 16,
            "motorcycle": 17,
            "bicycle": 18,
            "background": 19,
        }
        self.endmember_label = {
            "Roadlabel": 0,
            "Building_Concrete_label": 1,
            "Building_Glass_label": 2,
            "Car_white_label": 3,
            "Treelabel": 4,
            "sky": 5,
        }
    
    def sanity_check(self):
        # if the mask file is not exist, then remove the corresponding image file and label file and mask file from the list
        for i in range(len(self.img_files) - 1, -1, -1):
            if not os.path.isfile(self.mask_files[i][0]) or not os.path.isfile(self.mask_files[i][1]) \
               or not os.path.isfile(self.mask_files[i][2]) or not os.path.isfile(self.mask_files[i][3]) \
               or not os.path.isfile(self.mask_files[i][4]):
                del self.img_files[i]
                del self.mask_files[i]
                del self.label_mask[i]
    
    def creat_endmember_label(self, index, img):
        img_h, img_w = img.shape[:2]
        endmember_label = np.zeros((img_h, img_w), dtype=np.uint8) + len(self.endmember_label)
        for i, k in enumerate(self.endmember_label.keys()):
            if os.path.isfile(self.mask_files[index][i]):
                endmember_label[sio.loadmat(self.mask_files[index][i])["overlay"].astype(bool)] = self.endmember_label[k]
        
        img_labels = np.array(Image.open(self.label_mask[index]))
        pixel_index = np.zeros((img_h, img_w), dtype=bool)
        for i, k in enumerate(self.label_mapping.keys()):
            if any(k.lower() in key.lower() for key in self.endmember_label.keys()):
                pixel_index = np.logical_or(pixel_index, img_labels == self.label_mapping[k])
        
        return endmember_label, pixel_index

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        
        img = sio.loadmat(self.img_files[index])["filtered_img"].astype(np.float16) \
            if self.img_type != "rgb" else np.array(Image.open(self.img_files[index])).astype(np.float16)
        img_pos = np.indices(img.shape[:2]).transpose(1, 2, 0)

        endmember_label, pixel_index = self.creat_endmember_label(index, img)
        # Check if endmember_label only contains "5", if so, raise an error
        if np.unique(endmember_label).shape[0] == 1 and np.unique(endmember_label)[0] == len(self.endmember_label):
            return self.__getitem__(np.random.randint(0, len(self.img_files)))
        
        img = img[pixel_index, :]
        img_pos = img_pos[pixel_index, :]
        # only select the [6, 44, 11, 70, 3, 56, 35, 50, 49, 67, 47, 22] channels
        # img = img[:, [6, 44, 11, 70, 3, 56, 35, 50, 49, 67, 47, 22]]
        
        if img.shape[0] == 0:
            return self.__getitem__(np.random.randint(0, len(self.img_files)))
        img = img[:img.shape[0] // self.sequence_length * self.sequence_length, :]
        img_pos = img_pos[:img_pos.shape[0] // self.sequence_length * self.sequence_length, :]
        endmember_label = endmember_label[pixel_index]
        endmember_label = endmember_label[:endmember_label.shape[0] // self.sequence_length * self.sequence_length]
           
        img = torch.from_numpy(img).view(-1, self.sequence_length, img.shape[1]).contiguous()
        img_pos = np.reshape(img_pos, (-1, self.sequence_length, 2))
        target = torch.from_numpy(endmember_label).view(-1, self.sequence_length).contiguous()
        
        return img, target, img_pos

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def collate_fn(batch):
        images, targets, img_pos = list(zip(*batch))
        batched_imgs = torch.stack(images, dim=0).flatten(0, 1)
        batched_targets = torch.stack(targets, dim=0).flatten(0, 1).to(dtype=torch.long)
        batched_img_pos = np.vstack(img_pos, axis=0)
        if batched_imgs.shape[0] > 50000:
            index = np.random.choice(batched_imgs.shape[0], 50000, replace=False)
            batched_imgs = batched_imgs[index]
            batched_targets = batched_targets[index]
            batched_img_pos = batched_img_pos[index]
        return batched_imgs, batched_targets, batched_img_pos
    
    
class HSI_Drive(data.Dataset):
    def __init__(self, data_path: str = "", use_MF: bool = True, use_dual: bool = True,
                 use_OSP: bool = True, use_raw: bool = False, use_cache: bool = False, use_rgb: bool = False):
        self.use_MF = use_MF
        self.use_dual = use_dual
        self.use_OSP = use_OSP
        self.use_raw = use_raw
        self.use_cache = use_cache
        self.use_rgb = use_rgb
        
        self.data_folder_path = os.path.join(data_path, "cubes_fl32")
        self.data_folder_path = os.path.join(self.data_folder_path, "MF") if use_MF else self.data_folder_path
        if not use_raw:
            self.data_folder_path = os.path.join(self.data_folder_path, "Dual_HVI") if use_dual else os.path.join(self.data_folder_path, "Sin_HVI")
        
        path_ext = ""
        if use_MF:
            path_ext += "/MF"
        
        if use_raw:
            path_ext += ""
        elif use_dual and not use_raw:
            path_ext += "/Dual_HVI"
        elif not use_dual and not use_raw:
            path_ext += "/Sin_HVI"
        
        name_ext = "_MF_TC" if use_MF else "_TC"
        
        self.data_paths = [os.path.join(self.data_folder_path, file) for file in os.listdir(self.data_folder_path) if file.endswith(".mat")]
        self.label_paths = [file.replace("cubes_fl32", "labels").replace(path_ext, "").replace(name_ext, "").replace(".mat", ".png") for file in self.data_paths]
        self.rgb_paths = [file.replace("cubes_fl32", "RGB").replace(path_ext, "").replace(name_ext, "_pseudocolor").replace(".mat", ".png") for file in self.data_paths]
        
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
        self.selected_labels = [0, 1, 2, 4, 7]
        if use_cache:
            self.cache_data()
        
    def relabeling(self, label):
        for k, v in self.hsi_drive_original_label.items():
            if k not in self.selected_labels:
                label[label == k] = 255
        # relabel the label from 0 to end, with 255 as the background
        for i, k in enumerate(self.selected_labels):
            label[label == k] = i
        return label
    
    def cache_data(self):
        data_dict = defaultdict(list)
        for i in range(len(self.data_paths)):
            if not self.use_rgb:
                img = sio.loadmat(self.data_paths[i])["filtered_img"] if not self.use_raw else sio.loadmat(self.data_paths[i])["cube"]
                img = img.transpose(1, 2, 0) if self.use_raw else img
            else:
                img = np.array(Image.open(self.rgb_paths[i]))
                
            label = np.array(Image.open(self.label_paths[i]))
            label = self.relabeling(label)
            
            img = img.reshape(-1, img.shape[-1])
            label = label.reshape(-1)
            
            # remove 255 label
            img = img[label != 255, :]
            label = label[label != 255]
            
            if self.use_OSP and not self.use_dual and not self.use_raw and not self.use_rgb:
                img = img[:, [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]]
            elif self.use_OSP and self.use_dual and not self.use_raw and not self.use_rgb:
                img = img[:, [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]]
            
            for p in range(img.shape[0]):
                data_dict[label[p]].append(img[p])
        
        # randomly select 5000 samples from each class
        data_dict_selected = {}
        for k, v in data_dict.items():
            v = np.array(v)
            if len(v) > 50000:
                data_dict_selected[k] = v[np.random.choice(len(v), 50000, replace=False), :]
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
                img = sio.loadmat(self.data_paths[index])["filtered_img"] if not self.use_raw else sio.loadmat(self.data_paths[index])["cube"]
                img = img.transpose(1, 2, 0) if self.use_raw else img
            else:
                img = np.array(Image.open(self.rgb_paths[index]))
                
            label = np.array(Image.open(self.label_paths[index]))
            label = self.relabeling(label)
            img_pos = np.indices(img.shape[:2]).transpose(1, 2, 0)
            
            img = img.reshape(-1, img.shape[-1])
            img_pos = img_pos.reshape(-1, 2)
            label = label.reshape(-1)
            
            if self.use_OSP and not self.use_dual and not self.use_raw and not self.use_rgb:
                img = img[:, [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]]
            elif self.use_OSP and self.use_dual and not self.use_raw and not self.use_rgb:
                img = img[:, [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]]
            
            img = torch.from_numpy(img).to(dtype=torch.float32)
            label = torch.from_numpy(label).to(dtype=int)
        elif self.use_cache:
            img = torch.from_numpy(self.data_list[index]).to(dtype=torch.float16)
            label = torch.tensor(self.label_list[index], dtype=torch.int64)
            img_pos = None

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

class HSI_Drive_V1(data.Dataset):
    def __init__(self, data_path: str = "", use_MF: bool = True, use_dual: bool = True,
                 use_OSP: bool = True, use_raw: bool = False, use_rgb: bool = False, use_cache: bool = False,
                 use_attention: bool = False, use_large_mlp: bool = False, num_attention: int = 10):
        self.use_MF = use_MF
        self.use_dual = use_dual
        self.use_OSP = use_OSP
        self.use_raw = use_raw
        self.use_rgb = use_rgb
        self.use_cache = use_cache
        
        self.use_attention = use_attention
        self.use_large_mlp = use_large_mlp
        self.num_attention = num_attention
        
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
        
    def relabeling(self, label):
        for k, v in self.hsi_drive_original_label.items():
            if k not in self.selected_labels:
                label[label == k] = 255
        # relabel the label from 0 to end, with 255 as the background
        for i, k in enumerate(self.selected_labels):
            label[label == k] = i
        return label
    
    def cache_data(self):
        data_dict = defaultdict(list)
        for i in range(len(self.data_paths)):
            if not self.use_rgb:
                img = sio.loadmat(self.data_paths[i])["filtered_img"] if not self.use_raw else sio.loadmat(self.data_paths[i])["cube_fl32"]
                img = img.transpose(1, 2, 0) if self.use_raw else img / 1e3
            else:
                img = np.array(Image.open(self.rgb_paths[i]))
            
            label = np.array(Image.open(self.label_paths[i]))
            label = self.relabeling(label)
            
            img = img.reshape(-1, img.shape[-1])
            label = label.reshape(-1)
            
            # remove 255 label
            img = img[label != 255, :]
            label = label[label != 255]
            
            if self.use_OSP and not self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_OSP and self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_attention and self.use_dual and not self.use_raw and not self.use_rgb:
                attention_index = [193, 181, 169, 163, 175, 187, 58, 223, 184, 40] if self.use_large_mlp \
                    else [202, 250, 140, 212, 4, 169, 76, 58, 205, 14]
                img = img[:, attention_index[:self.num_attention]]
            
            for p in range(img.shape[0]):
                data_dict[label[p]].append(img[p])
        
        # randomly select 5000 samples from each class
        data_dict_selected = {}
        for k, v in data_dict.items():
            v = np.array(v)
            if len(v) > 50000:
                data_dict_selected[k] = v[np.random.choice(len(v), 50000, replace=False), :]
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
                img = sio.loadmat(self.data_paths[index])["filtered_img"] if not self.use_raw else sio.loadmat(self.data_paths[index])["cube_fl32"]
                img = img.transpose(1, 2, 0) if self.use_raw else img / 1e3
            else:
                img = np.array(Image.open(self.rgb_paths[index]))
                
            label = np.array(Image.open(self.label_paths[index]))
            label = self.relabeling(label)
            img_pos = np.indices(img.shape[:2]).transpose(1, 2, 0)
            
            img = img.reshape(-1, img.shape[-1])
            img_pos = img_pos.reshape(-1, 2)
            label = label.reshape(-1)
            
            if self.use_OSP and not self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_OSP and self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_attention and self.use_dual and not self.use_raw and not self.use_rgb:
                attention_index = [193, 181, 169, 163, 175, 187, 58, 223, 184, 40] if self.use_large_mlp \
                    else [202, 250, 140, 212, 4, 169, 76, 58, 205, 14]
                img = img[:, attention_index[:self.num_attention]]
            
            img = torch.from_numpy(img).to(dtype=torch.float32)
            label = torch.from_numpy(label).to(dtype=int)
        elif self.use_cache:
            img = torch.from_numpy(self.data_list[index]).to(dtype=torch.float16)
            label = torch.tensor(self.label_list[index], dtype=torch.int64)
            img_pos = None

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


class HSI_Drive_V1_(data.Dataset):
    def __init__(self, data_path: str = "", use_MF: bool = True, use_dual: bool = True,
                 use_OSP: bool = True, use_raw: bool = False, use_rgb: bool = False, use_cache: bool = False,
                 use_attention: bool = False, use_large_mlp: bool = False, num_attention: int = 10):
        self.use_MF = use_MF
        self.use_dual = use_dual
        self.use_OSP = use_OSP
        self.use_raw = use_raw
        self.use_rgb = use_rgb
        self.use_cache = use_cache
        
        self.use_attention = use_attention
        self.use_large_mlp = use_large_mlp
        self.num_attention = num_attention
        
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
        data_dict = defaultdict(list)
        for i in range(len(self.data_paths)):
            if not self.use_rgb:
                img = sio.loadmat(self.data_paths[i])["filtered_img"] if not self.use_raw else sio.loadmat(self.data_paths[i])["cube_fl32"]
                img = img.transpose(1, 2, 0) if self.use_raw else img / 1e3
            else:
                img = np.array(Image.open(self.rgb_paths[i]))
            
            label = np.array(Image.open(self.label_paths[i]))
            end_label = np.array(Image.open(self.end_label_paths[i]))
            label = self.relabeling(label, end_label)
            
            img = img.reshape(-1, img.shape[-1])
            label = label.reshape(-1)
            
            # remove 255 label
            img = img[label != 255, :]
            label = label[label != 255]
            
            if self.use_OSP and not self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_OSP and self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_attention and self.use_dual and not self.use_raw and not self.use_rgb:
                attention_index = [163, 40, 58, 218, 4, 230, 76, 121, 176, 224] if self.use_large_mlp \
                    else [223, 163, 58, 230, 40, 193, 145, 187, 248, 181]
                img = img[:, attention_index[:self.num_attention]]
            
            for p in range(img.shape[0]):
                data_dict[label[p]].append(img[p])
        # print(len(data_dict[7]))
        # randomly select 5000 samples from each class
        data_dict_selected = {}
        for k, v in data_dict.items():
            v = np.array(v)
            if len(v) > 5000:
                data_dict_selected[k] = v[np.random.choice(len(v), 5000, replace=False), :]
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
                img = sio.loadmat(self.data_paths[index])["filtered_img"] if not self.use_raw else sio.loadmat(self.data_paths[index])["cube_fl32"]
                img = img.transpose(1, 2, 0) if self.use_raw else img / 1e3
            else:
                img = np.array(Image.open(self.rgb_paths[index]))
                
            label = np.array(Image.open(self.label_paths[index]))
            end_label = np.array(Image.open(self.end_label_paths[index]))
            label = self.relabeling(label, end_label)
            img_pos = np.indices(img.shape[:2]).transpose(1, 2, 0)
            
            img = img.reshape(-1, img.shape[-1])
            img_pos = img_pos.reshape(-1, 2)
            label = label.reshape(-1)
            
            if self.use_OSP and not self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_OSP and self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_attention and self.use_dual and not self.use_raw and not self.use_rgb:
                attention_index = [163, 40, 58, 218, 4, 230, 76, 121, 176, 224] if self.use_large_mlp \
                    else [223, 163, 58, 230, 40, 193, 145, 187, 248, 181]
                img = img[:, attention_index[:self.num_attention]]
            
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


class HSI_Drive_V1_visual(data.Dataset):
    def __init__(self, data_path: str = "", use_MF: bool = True, use_dual: bool = True,
                 use_OSP: bool = True, use_raw: bool = False, use_rgb: bool = False, use_cache: bool = False,
                 use_attention: bool = False, use_large_mlp: bool = False, num_attention: int = 10):
        self.use_MF = use_MF
        self.use_dual = use_dual
        self.use_OSP = use_OSP
        self.use_raw = use_raw
        self.use_rgb = use_rgb
        self.use_cache = use_cache
        
        self.use_attention = use_attention
        self.use_large_mlp = use_large_mlp
        self.num_attention = num_attention
        
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
        
    def relabeling(self, label):
        for k, v in self.hsi_drive_original_label.items():
            if k not in self.selected_labels:
                label[label == k] = 255
        # relabel the label from 0 to end, with 255 as the background
        for i, k in enumerate(self.selected_labels):
            label[label == k] = i
        return label
    
    def cache_data(self):
        data_dict = defaultdict(list)
        for i in range(len(self.data_paths)):
            if not self.use_rgb:
                img = sio.loadmat(self.data_paths[i])["filtered_img"] if not self.use_raw else sio.loadmat(self.data_paths[i])["cube_fl32"]
                img = img.transpose(1, 2, 0) if self.use_raw else img / 1e3
            else:
                img = np.array(Image.open(self.rgb_paths[i]))
            
            label = np.array(Image.open(self.label_paths[i]))
            end_label = np.array(Image.open(self.end_label_paths[i]))
            label = self.relabeling(label, end_label)
            
            img = img.reshape(-1, img.shape[-1])
            label = label.reshape(-1)
            
            # remove 255 label
            img = img[label != 255, :]
            label = label[label != 255]
            
            if self.use_OSP and not self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_OSP and self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_attention and self.use_dual and not self.use_raw and not self.use_rgb:
                attention_index = [163, 40, 58, 218, 4, 230, 76, 121, 176, 224] if self.use_large_mlp \
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
                img = sio.loadmat(self.data_paths[index])["filtered_img"] if not self.use_raw else sio.loadmat(self.data_paths[index])["cube_fl32"]
                img = img.transpose(1, 2, 0) if self.use_raw else img / 1e3
            else:
                img = np.array(Image.open(self.rgb_paths[index]))
            rgb_img = np.array(Image.open(self.rgb_paths[index]))
            
            label = np.array(Image.open(self.label_paths[index]))
            # end_label = np.array(Image.open(self.end_label_paths[index]))
            label = self.relabeling(label)
            img_pos = np.indices(img.shape[:2]).transpose(1, 2, 0)
            
            img = img.reshape(-1, img.shape[-1])
            img_pos = img_pos.reshape(-1, 2)
            rgb_img = rgb_img.reshape(-1, 3)
            label = label.reshape(-1)
            
            if self.use_OSP and not self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_OSP and self.use_dual and not self.use_raw and not self.use_rgb:
                OSP_index = [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]
                img = img[:, OSP_index[:self.num_attention]]
            elif self.use_attention and self.use_dual and not self.use_raw and not self.use_rgb:
                attention_index = [163, 40, 58, 218, 4, 230, 76, 121, 176, 224] if self.use_large_mlp \
                    else [223, 163, 58, 230, 40, 193, 145, 187, 248, 181]
                img = img[:, attention_index[:self.num_attention]]
            
            img = torch.from_numpy(img).to(dtype=torch.float32)
            label = torch.from_numpy(label).to(dtype=int)
        elif self.use_cache:
            img = torch.from_numpy(self.data_list[index]).to(dtype=torch.float16)
            label = torch.tensor(self.label_list[index], dtype=torch.int64)
            img_pos = None

        # get the label file name
        label_name = self.label_paths[index].split("/")[-1].split(".")[0]
        return img, label, img_pos, rgb_img, label_name
    
    def __len__(self):
        return len(self.data_paths) if not self.use_cache else len(self.data_list)
    
    @staticmethod
    def collate_fn(batch):
        images, targets, img_pos, rgb_img, label_name = list(zip(*batch))
        batched_imgs = torch.stack(images, dim=0)
        batched_imgs = batched_imgs.flatten(0, 1) if len(batched_imgs.shape) == 3 else batched_imgs
        batched_targets = torch.stack(targets, dim=0).to(dtype=torch.int64)
        batched_targets = batched_targets.flatten(0, 1) if len(batched_targets.shape) == 2 else batched_targets
        batched_img_pos = np.vstack(img_pos) if img_pos[0] is not None else None
        batched_rgb_img = np.stack(rgb_img, axis=0)
        # print max and min of label ignoring 255
        # print(f"Max label: {torch.max(batched_targets[batched_targets != 255])}, Min label: {torch.min(batched_targets[batched_targets != 255])}")
        return batched_imgs, batched_targets, batched_img_pos, batched_rgb_img, label_name
    

def stratified_split(dataset, train_ratio=0.8):
    # split the dataset into train and validation set, the dataset is for pixel-wise classification
    # the label of each img is in shape (H*W, 1), as a tensor, so we need to split the dataset based on the label
    # make sure the train and validation all have the same distribution of the label

    label_dict = defaultdict(list)
    for i in range(len(dataset)):
        _, target, _ = dataset[i]
        label_dict[tuple(target.unique().tolist())].append(i)
        
    train_indices, train_labels = [], []
    val_indices, val_labels = [], []
    for labels, indices in label_dict.items():
        np.random.shuffle(indices)
        split = int(np.floor(train_ratio * len(indices)))
        
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])
        
        #change labels tuple to list
        labels = list(labels)
        if indices[:split] is not None:
            train_labels.extend(labels)
        if indices[split:] is not None:
            val_labels.extend(labels)
    
    #check unique labels in train and val
    train_labels = set(train_labels)
    val_labels = set(val_labels)
    print(f"Unique labels in train: {train_labels}, Unique labels in val: {val_labels}")
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset
        