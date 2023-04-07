from PIL import Image
import torch
import os
import scipy.io as sio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import random
from tqdm import tqdm


data_path = "/data2/chaoyi/HSI Dataset/V2/train/"
img_folder_list = os.listdir(data_path)
img_type = "ALL"

img_files = [os.path.join(data_path, img_folder, file)
             for img_folder in img_folder_list
             for file in os.listdir(os.path.join(data_path, img_folder))
             if os.path.splitext(file)[-1].lower() == ".mat" and img_type in file]

for img_file in tqdm(img_files):
    img = sio.loadmat(img_file)["filtered_img"]