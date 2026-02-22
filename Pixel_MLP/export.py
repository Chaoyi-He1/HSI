import time
import os
import datetime
import random
import torch

from src import get_model
from train_utils import train_one_epoch, evaluate, evaluate_sr, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
from my_dataset import *
import transforms as T
from torch.utils.data import DataLoader, random_split


ckpt = torch.load("EC_dataset/Pixel_MLP/multi_train/HSI_drive/OSP/model_499.pth", weights_only=False)
model = ckpt['model']
# print the model's each layer shape
for name, param in model.items():
    print(f"{name}: {param.shape}")