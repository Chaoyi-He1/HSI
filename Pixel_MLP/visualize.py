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


def create_model(model_name="transformer", num_classes=2, in_chans=10):
    model = get_model(model_name, num_classes=num_classes, in_channels=in_chans)
    return model


def creat_endmember_label(img, endmember_label_dict, endmember_files, original_label_file, original_label_dict):
    img_h, img_w = img.shape[:2]
    endmember_label = np.zeros((img_h, img_w), dtype=np.uint8) + len(endmember_label_dict) - 1
    # skip background
    for i, label in enumerate(k for k in endmember_label_dict.keys() if k != 6):
        if os.path.exists(endmember_files[i]):
            endmember_label[sio.loadmat(endmember_files[i])["overlay"].astype(bool)] = label
    
    img_original_label = np.array(Image.open(original_label_file))
    pixel_index = np.zeros((img_h, img_w), dtype=bool)
    # skip background
    for label, name in ((k, v) for k, v in original_label_dict.items() if v != "background"):
        if any(name.lower() in value.lower() for value in endmember_label_dict.values()):
            pixel_index = np.logical_or(pixel_index, img_original_label == label)
    
    return endmember_label, pixel_index
        

def main(args):
    device = torch.device(args.device)
    
    rgb = "rgb" if args.img_type == 'rgb' else ''
    channel = "_ALL71channel" if args.img_type == 'ALL' else "_OSP10channel"
    
    num_classes = args.num_classes
    model = create_model(num_classes=num_classes, in_chans=3 if args.img_type == "rgb" else 12)
    model.to(device)
    
    if args.resume.endswith(".pth"):
        checkpoint = torch.load(args.resume, map_location='cpu')  
        model.load_state_dict(checkpoint['model'])
    
    data_path = '/data2/chaoyi/HSI_Dataset/V2/test'
    test_pics_folders = [
        '20210410_114953_02', '20210409_142547_01', '20210409_152426_305174',
        '20210410_105622_01', '20210409_155305_02', '20210409_144751_58',
        '20210409_170413_00' 
    ]
    origin_label = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic light",
        7: "traffic sign",
        8: "tree",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle",
        19: "background",
    }
    endmember_label = {
        0: "Roadlabel",
        1: "Building_Concrete_label",
        2: "Building_Glass_label",
        3: "Car_white_label",
        4: "Treelabel",
        5: "Skylabel",
        6: "background",
    }
    endmember_label_color = {
        "Roadlabel": np.array([255, 0, 0]), # red
        "Building_Concrete_label": np.array([0, 255, 0]), # green
        "Building_Glass_label": np.array([0, 0, 255]), # blue
        "Car_white_label": np.array([255, 255, 0]), # yellow
        "Treelabel": np.array([255, 0, 255]), # magenta
        "Skylabel": np.array([0, 255, 255]), # cyan
        "background": np.array([0, 0, 0]), # black
    }
    
    for pic_folder in test_pics_folders:
        
        if args.img_type != "rgb":
            for file in os.listdir(os.path.join(data_path, pic_folder)):
                if os.path.splitext(file)[-1].lower() == ".mat" and args.img_type in file and args.img_type != "rgb":
                    pic_path = os.path.join(data_path, pic_folder, file)
                    break
        else:
            for file in os.listdir(os.path.join(data_path, pic_folder)):
                if os.path.splitext(file)[-1].lower() == ".jpg" and args.img_type in file and args.img_type == "rgb":
                    pic_path = os.path.join(data_path, pic_folder, file)
                    break
        
        endmember_groundtruth_files = [
            pic_path.replace(pic_path.split(os.sep)[-1], 
                             os.path.splitext(
                                 os.path.basename(pic_path))[0].replace(channel, "").replace(rgb, "") 
                             + "_" + "Roadlabel" + '.mat'),
            pic_path.replace(pic_path.split(os.sep)[-1], 
                             os.path.splitext(
                                 os.path.basename(pic_path))[0].replace(channel, "").replace(rgb, "") 
                             + "_" + "Building_Concrete_label" + '.mat'),
            pic_path.replace(pic_path.split(os.sep)[-1],
                                os.path.splitext(
                                    os.path.basename(pic_path))[0].replace(channel, "").replace(rgb, "") 
                                + "_" + "Building_Glass_label" + '.mat'),
            pic_path.replace(pic_path.split(os.sep)[-1],
                                os.path.splitext(
                                    os.path.basename(pic_path))[0].replace(channel, "").replace(rgb, "") 
                                + "_" + "Car_white_label" + '.mat'),
            pic_path.replace(pic_path.split(os.sep)[-1],
                                os.path.splitext(
                                    os.path.basename(pic_path))[0].replace(channel, "").replace(rgb, "") 
                                + "_" + "Treelabel" + '.mat'),
            pic_path.replace(pic_path.split(os.sep)[-1],
                                os.path.splitext(
                                    os.path.basename(pic_path))[0].replace(channel, "").replace(rgb, "") 
                                + "_" + "Skylabel" + '.mat'),
        ]
        rgb = "rgb" if args.img_type != 'rgb' else ''
        original_label_file = pic_path.replace(pic_path.split(os.sep)[-1], rgb
                                       + os.path.splitext(os.path.basename(pic_path))[0].replace(channel, "")
                                       + "_gray.png")
        
        img = sio.loadmat(pic_path)["filtered_img"].astype(np.float16) \
            if args.img_type != "rgb" else np.array(Image.open(pic_path)).astype(np.float16)
        rgb_img = np.array(Image.open(os.path.join(data_path, pic_folder, "rgb" + pic_folder + ".jpg")))
        img_pos = np.indices(img.shape[:2]).transpose(1, 2, 0)
        img_original_label = np.array(Image.open(original_label_file))
        endmember_label_img, pixel_index = creat_endmember_label(img, endmember_label, endmember_groundtruth_files, original_label_file, origin_label)
        
        # take out the pixel according to the pixel_index
        img = img[pixel_index, :]
        img_pos = img_pos[pixel_index, :]
        img_original_label = img_original_label[pixel_index]
        endmember_label_img = endmember_label_img[pixel_index]
        
        img = img[:img.shape[0] // 10 * 10, :]
        img = img[:, [6, 44, 11, 70, 3, 56, 35, 50, 49, 67, 47, 22]]
        img_pos = img_pos[:img_pos.shape[0] // 10 * 10, :]
        img_original_label = img_original_label[:img_original_label.shape[0] // 10 * 10]
        endmember_label_img = endmember_label_img[:endmember_label_img.shape[0] // 10 * 10]
        
        # transform img to tensor, use model to predict in gpu
        img = torch.from_numpy(img).to(device).view(-1, 10, img.shape[1]) # img: (-1, 10, dim)
        img_pos = np.reshape(img_pos, (-1, 10, 2))
        img_original_label = torch.from_numpy(img_original_label).to(device).view(-1, 10)
        endmember_label_img = torch.from_numpy(endmember_label_img).to(device).view(-1, 10)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            pred = model(img) # pred: (-1, 10, dim)
            del img
        pred = torch.argmax(pred, dim=-1).view(-1,).cpu().numpy()
        img_pos = np.reshape(img_pos, (-1, 2))
        img_original_label = img_original_label.view(-1,).cpu().numpy()
        endmember_label_img = endmember_label_img.view(-1,).cpu().numpy()
        
        # visualize the prediction, use the rgb image as background, with transparency factor adjustable
        # plot 2 results in one figure, subplot one is the "endmember_label_img" (ground truth), the other subplot is the "pred" (prediction)
        # the color of the prediction is the same as the ground truth according to the "endmember_label_color"
        # the position of the prediction is the same as the ground truth according to the "img_pos"
        transparency = 0.3
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        # subplot 1, use rgb image as background and plot the ground truth on it, label is in shape (-1,) woth position in shape (-1, 2)
        # add the color according to the "endmember_label_color" to the rgb image at the position of the label
        # make the rgb image transparent by the "transparency"
        img_to_plot = rgb_img.copy() 
        for i, label in enumerate(endmember_label_img):
            # this pixel's origin rgb value will be set as background color with transparency, the color of the label will be added to the background color
            img_to_plot[img_pos[i, 0], img_pos[i, 1]] = endmember_label_color[endmember_label[label]] * (1 - transparency) + img_to_plot[img_pos[i, 0], img_pos[i, 1]] * transparency
        ax[0].imshow(img_to_plot)
        ax[0].set_title("Ground Truth")
        # subplot 2, use rgb image as background and plot the prediction on it, label is in shape (-1,) woth position in shape (-1, 2)
        # add the color according to the "endmember_label_color" to the rgb image at the position of the label
        img_to_plot = rgb_img.copy()
        for i, label in enumerate(pred):
            img_to_plot[img_pos[i, 0], img_pos[i, 1]] = endmember_label_color[endmember_label[label]] * (1 - transparency) + img_to_plot[img_pos[i, 0], img_pos[i, 1]] * transparency
        ax[1].imshow(img_to_plot)
        ax[1].set_title("Prediction")
        
        # add color's label name to the figure
        for i, (label, color) in enumerate(endmember_label_color.items()):
            ax[0].text(10, 60 + 50 * i, label, color=color / 255, fontsize=15)
            ax[1].text(10, 60 + 50 * i, label, color=color / 255, fontsize=15)
        # save the figure
        plt.savefig(os.path.join(args.output_dir, pic_folder + ".png"))
        
        
    
    
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