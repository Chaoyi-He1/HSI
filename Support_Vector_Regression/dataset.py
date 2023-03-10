from PIL import Image
import torch
import os
import scipy.io
import neo
from pathlib import Path
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, images_path: str, filter_path: str):
        self.images_path = images_path
        self.filter_path = filter_path
        try:
            self.images_path = str(Path(self.images_path))
            # parent = str(Path(path).parent) + os.sep
            if os.path.isfile(self.images_path):  # file
                # Read the data from "data/my_train.txt" or "data/val_data.txt"，read each line as a file
                with open(self.images_path, "r") as f:
                    f = f.read().splitlines()
            else:
                raise Exception("%s does not exist" % self.images_path)
            self.img_files = [x for x in f]
            # Prevent different systems from sorting differently, resulting in differences in shape files
            self.img_files.sort()
        except Exception as e:
            raise FileNotFoundError("Error loading data from {}. {}".format(self.images_path, e))

        try:
            self.filter_path = str(Path(self.filter_path))
            if os.path.isfile(self.filter_path):
                self.mat_contents = scipy.io.loadmat(self.filter_path)
            else:
                raise Exception("%s does not exist" % self.images_path)
            self.filter_respond = torch.as_tensor(self.mat_contents["responsivity"])
        except Exception as e:
            raise FileNotFoundError("Error loading data from {}. {}".format(self.images_path, e))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        r = neo.io.Spike2IO(filename=self.img_files[item])
        img = r.read_block()

        if self.transform is not None:
            img = self.transform(img)

        return img

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

