import time
import os
import datetime
import torch
import numpy as np
import pandas as pd
from src import get_model
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
from my_dataset import HSI_Segmentation, HSI_Transformer, HSI_Transformer_all
import transforms as T
from torch.utils.tensorboard import SummaryWriter


def create_model(model_name="transformer", num_classes=2, in_chans=10):
    model = get_model(model_name, num_classes=num_classes, in_channels=in_chans)
    return model


def main(args):
    device = torch.device(args.device)
    
    num_classes = args.num_classes
    model = create_model(num_classes=num_classes, in_chans=3 if args.img_type == "rgb" else 71)
    model.to(device)
    
    if args.resume.endswith(".pth"):
        checkpoint = torch.load(args.resume, map_location='cpu')  
        model.load_state_dict(checkpoint['model'])
    
    # Save model.atten parameters to csv file, model.atten in shape (1, 1, 71)
    atten = model.atten.detach().cpu().numpy().flatten()
    atten_path = os.path.join(args.output_dir, "atten.csv")
    pd.DataFrame(atten).to_csv(atten_path, index=False, header=False)
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    parser.add_argument('--img_type', default='OSP', help='image type: OSP or PCA or rgb')
    parser.add_argument('--output-dir', default='./Pixel_MLP/multi_train/OSP/', help='path where to save')
    parser.add_argument('--resume', default='./Pixel_MLP/multi_train/OSP/model_053.pth', help='resume from checkpoint')
    args = parser.parse_args()
    
    main(args)