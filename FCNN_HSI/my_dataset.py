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
        return batched_imgs, batched_targets


class HSI_Transformer_all(data.Dataset):
    def __init__(self, data_path: str = "", label_type: str = "gray", img_type: str = "OSP", 
                 transform=None):
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
        self.transforms = transform

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
                                       + "_" + "Treelabel" + '.mat'),] for img in self.img_files]
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
            "vegetation": 8,
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
        
        return endmember_label

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        
        img = sio.loadmat(self.img_files[index])["filtered_img"].astype(np.float16) \
            if self.img_type != "rgb" else np.array(Image.open(self.img_files[index])).astype(np.float16)

        endmember_label = self.creat_endmember_label(index, img)
        # Check if endmember_label only contains "5", if so, raise an error
        if np.unique(endmember_label).shape[0] == 1 and np.unique(endmember_label)[0] == len(self.endmember_label):
            return self.__getitem__(np.random.randint(0, len(self.img_files)))
           
        # img = torch.from_numpy(img)
        # target = torch.from_numpy(endmember_label)
        if self.transforms is not None:
            img, target = self.transforms(img, endmember_label)
        
        return img, target

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = torch.stack(images, dim=0)
        batched_targets = torch.stack(targets, dim=0).to(dtype=torch.long)
        return batched_imgs, batched_targets


class HSI_Drive(data.Dataset):
    def __init__(self, data_path: str = "", use_MF: bool = True, use_dual: bool = True,
                 use_OSP: bool = True, transforms=None):
        self.use_MF = use_MF
        self.use_dual = use_dual
        self.use_OSP = use_OSP
        
        self.transforms = transforms
        
        self.data_folder_path = os.path.join(data_path, "cubes_fl32")
        self.data_folder_path = os.path.join(self.data_folder_path, "MF") if use_MF else self.data_folder_path
        self.data_folder_path = os.path.join(self.data_folder_path, "Dual_HVI") if use_dual else os.path.join(self.data_folder_path, "Sin_HVI")
        
        path_ext = ""
        if use_MF:
            path_ext += "/MF"
        if use_dual:
            path_ext += "/Dual_HVI"
        elif not use_dual:
            path_ext += "/Sin_HVI"
        
        name_ext = "_MF_TC" if use_MF else "_TC"
        
        self.data_paths = [os.path.join(self.data_folder_path, file) for file in os.listdir(self.data_folder_path) if file.endswith(".mat")]
        self.label_paths = [file.replace("cubes_fl32", "labels").replace(path_ext, "").replace(name_ext, "").replace(".mat", ".png") for file in self.data_paths]
        
        for i in range(len(self.data_paths) - 1, -1, -1):
            if not os.path.isfile(self.label_paths[i]):
                del self.data_paths[i]
                del self.label_paths[i]
        
        assert len(self.data_paths) == len(self.label_paths) and len(self.data_paths) > 0, "The number of data files and label files are not equal."

        self.hsi_drive_original_label = {
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
        self.selected_labels = [1, 3, 4, 5, 6, 9, 10]
        
    def relabeling(self, label):
        for k, v in self.hsi_drive_original_label.items():
            if k not in self.selected_labels:
                label[label == k] = 255
        # relabel the label from 0 to end, with 255 as the background
        for i, k in enumerate(self.selected_labels):
            label[label == k] = i + 1
        return label
        
    
    def __getitem__(self, index):
        img = sio.loadmat(self.data_paths[index])["filtered_img"]
        label = np.array(Image.open(self.label_paths[index]))
        label = self.relabeling(label)
        img_pos = np.indices(img.shape[:2]).transpose(1, 2, 0)
        
        if self.use_OSP and not self.use_dual:
            img = img[:, :, [60, 44, 17, 27, 53, 4, 1, 20, 71, 13]]
        elif self.use_OSP and self.use_dual:
            img = img[:, :, [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]]
        
        img = torch.from_numpy(img).to(dtype=torch.float32).permute(2, 0, 1)
        label = torch.from_numpy(label).to(dtype=torch.int64)
        img_pos = torch.from_numpy(img_pos).permute(2, 0, 1)
        
        if self.transforms is not None:
            img, label, img_pos = self.transforms(img, label, img_pos)
        
        img_pos = img_pos.permute(1, 2, 0).numpy()

        return img, label, img_pos
    
    def __len__(self):
        return len(self.data_paths)
    
    @staticmethod
    def collate_fn(batch):
        images, targets, img_pos = list(zip(*batch))
        batched_imgs = torch.stack(images, dim=0)
        batched_targets = torch.stack(targets, dim=0)
        batched_img_pos = np.stack(img_pos, axis=0)
        # print max and min of label ignoring 255
        # print(f"Max label: {torch.max(batched_targets[batched_targets != 255])}, Min label: {torch.min(batched_targets[batched_targets != 255])}")
        return batched_imgs, batched_targets, batched_img_pos


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
        