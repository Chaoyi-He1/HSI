import os
import scipy.io as sio
import torch.utils.data as data
from PIL import Image


class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)

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

        self.img_files = [os.path.join(data_path, img_folder, file)
                          for img_folder in self.img_folder_list
                          for file in os.listdir(img_folder)
                          if os.path.splitext(file)[-1].lower() == ".mat" and img_type in file]
        self.img_files.sort()
        self.mask_files = [img.replace(os.path.splitext(img)[-1], ".png")
                              .replace(os.path.splitext(img)[-2], "label_" + label_type)
                           for img in self.img_files]

        assert (len(self.img_files) == len(self.mask_files))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = sio.loadmat(self.img_files[index])
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
        return batched_imgs, batched_targets
