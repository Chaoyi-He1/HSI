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
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
from my_dataset import *


def create_model(model_name="transformer", num_classes=2, in_chans=10):
    model = get_model(model_name, num_classes=num_classes, in_channels=in_chans)
    return model


def overlay_labels(img, labels, endmember_label_color, hsi_drive_label, transparency):
    mask = labels != 255
    unique_labels = np.unique(labels[mask])
    for label in unique_labels:
        label_mask = labels == label
        img[label_mask] = endmember_label_color[hsi_drive_label[label]] * (1 - transparency) + img[label_mask] * transparency
    return img

        
def main(args):
    device = torch.device(args.device)
    
    whole_img_dataset = HSI_Drive_V1(data_path=args.data_path,
                                     use_MF=args.use_MF,
                                     use_dual=args.use_dual,
                                     use_OSP=args.use_OSP,
                                     use_raw=args.use_raw,
                                     use_cache=False,
                                     use_rgb=args.use_rgb,
                                     use_attention=args.use_attention,
                                     use_large_mlp=args.use_large_mlp,
                                     num_attention=args.num_attention,)
    
    test_sampler = torch.utils.data.SequentialSampler(whole_img_dataset)
    
    val_data_loader = torch.utils.data.DataLoader(
        whole_img_dataset, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=whole_img_dataset.collate_fn, drop_last=False)
    
    
    hsi_drive_label = {
        "Road": 0,  # gray
        "Road marks": 1,    # yellow
        "Vegatation": 2,    # green
        "Painted metal": 3, # red
        "Sky": 4,        # blue
        "Concrete/Stone/Brick": 5,  # brown
        "Pedestrian/Cyclist": 6,    # pink
        "Unpainted metal": 7,   # purple
        "Glass/Transparent Plastic": 8, # light blue
        "Unknown": 255  
    }
    endmember_label_color = {
        "Road": np.array([128, 128, 128]),  # gray
        "Road marks": np.array([255, 255, 0]),    # yellow
        "Vegatation": np.array([0, 255, 0]),    # green
        "Painted metal": np.array([255, 0, 0]), # red
        "Sky": np.array([0, 0, 255]),        # blue
        "Concrete/Stone/Brick": np.array([165, 42, 42]),  # brown
        "Pedestrian/Cyclist": np.array([255, 192, 203]),    # pink
        "Unpainted metal": np.array([128, 0, 128]),   # purple
        "Glass/Transparent Plastic": np.array([173, 216, 230]), # light blue
    }
    
    print("Creating model")
    if args.use_rgb:
        in_chans = 3
    elif args.use_OSP:
        in_chans = args.num_attention
    elif args.use_attention:
        in_chans = args.num_attention
    elif not args.use_OSP and args.use_dual and not args.use_raw:
        in_chans = 252
    elif not args.use_OSP and not args.use_dual and not args.use_raw:
        in_chans = 71
    elif args.use_raw:
        in_chans = 25
    model = create_model(num_classes=9, in_chans=in_chans, large=args.use_large_mlp)
    model.to(device)
    
    for image, target, img_pos, rgb_img, name in val_data_loader:
        # transform img to tensor, use model to predict in gpu
        image, target = image.to(device), target.to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            pred = model(image) # pred: (-1, 10, dim)
            
        pred = torch.argmax(pred, dim=-1).view(-1,).cpu().numpy()
        
        img_pos = np.reshape(img_pos, (219, 406, 2))
        img_original_label = target.view(219, 406).cpu().numpy()
        pred = pred.view(219, 406).cpu().numpy()
        rgb_img = rgb_img.reshape(219, 406, 3)  # np.array, shape (H, W, C)
        
        # visualize the prediction, use the rgb image as background, with transparency factor adjustable
        # plot 2 results in one figure, subplot one is the "img_original_label" (ground truth), the other subplot is the "pred" (prediction)
        # the color of the prediction is the same as the ground truth according to the "endmember_label_color"
        # the position of the prediction is the same as the ground truth according to the "img_pos"
        transparency = 0.3
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        
        # subplot 1, use rgb image as background and plot the ground truth on it, label is in shape (219, 406) with position in shape (219, 406, 2)
        # add the color according to the "endmember_label_color" to the rgb image at the position of the label, ignore the "Unknown" (255) ground truth label
        # make the rgb image transparent by the "transparency"
        img_to_plot = overlay_labels(rgb_img.copy(), img_original_label, endmember_label_color, hsi_drive_label, transparency)
        ax[0].imshow(img_to_plot)
        ax[0].set_title("Ground Truth")
        
        # subplot 2, use rgb image as background and plot the prediction on it, label is in shape (-1,) woth position in shape (-1, 2)
        # add the color according to the "endmember_label_color" to the rgb image at the position of the label
        # make the rgb image transparent by the "transparency"
        img_to_plot = overlay_labels(rgb_img.copy(), pred, endmember_label_color, hsi_drive_label, transparency)
        ax[1].imshow(img_to_plot)
        ax[1].set_title("Prediction")
        
        # add color's label name to the figure
        for i, (label, color) in enumerate(endmember_label_color.items()):
            ax[0].text(10, 60 + 50 * i, label, color=color / 255, fontsize=15)
            ax[1].text(10, 60 + 50 * i, label, color=color / 255, fontsize=15)
        # save the figure
        plt.savefig(os.path.join(args.output_dir, name + ".png"))
        plt.close(fig)
        
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-classes', default=7, type=int, help='num_classes')
    parser.add_argument('--img_type', default='ALL', help='image type: OSP or PCA or rgb')
    parser.add_argument('--output-dir', default='./Pixel_MLP/multi_train/OSP/', help='path where to save')
    parser.add_argument('--resume', default='Pixel_MLP/multi_train/OSP/model_336.pth', help='resume from checkpoint')
    args = parser.parse_args()
    
    main(args)