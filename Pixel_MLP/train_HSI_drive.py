import time
import os
import datetime

import torch

from src import get_model
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
from my_dataset import HSI_Segmentation, HSI_Transformer, HSI_Transformer_all
import transforms as T
from torch.utils.tensorboard import SummaryWriter


def create_model(model_name="mlp_pixel", num_classes=2, in_chans=10):
    model = get_model(model_name, num_classes=num_classes, in_channels=in_chans)
    return model