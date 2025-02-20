# Use shap to explain the each input feature's contribution to the model's prediction
import shap
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
from torch.utils.data import DataLoader, random_split
import scipy.io as sio


def create_model(model_name="mlp_pixel", num_classes=2, in_chans=10, large=True):
    model = get_model(model_name, num_classes=num_classes, in_channels=in_chans, large=large)
    return model


def main(args):
    device = torch.device(args.device)
    
    whole_dataset = HSI_Drive_V1_(data_path=args.data_path,
                                  use_MF=args.use_MF,
                                  use_dual=args.use_dual,
                                  use_OSP=args.use_OSP,
                                  use_raw=args.use_raw,
                                  use_cache=args.use_cache,
                                  use_rgb=args.use_rgb,
                                  use_attention=args.use_attention,
                                  use_large_mlp=args.use_large_mlp,
                                  num_attention=args.num_attention,)
    
    train_dataset, val_dataset = random_split(whole_dataset, [int(0.8*len(whole_dataset)), len(whole_dataset)-int(0.8*len(whole_dataset))])
    
    # test shap on the whole dataset
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=whole_dataset.collate_fn, drop_last=True)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=whole_dataset.collate_fn, drop_last=False)
    
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
    
    num_classes = args.num_classes
    
    model = create_model(num_classes=num_classes, in_chans=in_chans, large=args.use_large_mlp)
    model.to(device)
    
    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    
    # Define a prediction function
    def model_predict(x):
        model.eval()
        # separately send the input to device for prediction
        with torch.no_grad(), torch.cuda.amp.autocast():
            x = torch.tensor(x).to(device)
            y = model(x).cpu().numpy()
        return y
    
    # Generate SHAP values
    # generate X_train, X_test from val_dataset with 500 samples from each class, 
    # pick the best 500 samples from each class to represent the class 
    # based on the model's prediction. pick those samples with the highest softmax probability
    X_train = []
    result_dict = {}
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, targets, _ in train_data_loader:
            images = images.to(device)
            output = model(images)
            pred = output.softmax(1)
            for i in range(len(pred)):
                label = targets[i].item()
                if label not in result_dict:
                    result_dict[label] = []
                result_dict[label].append((pred[i][label].item(), images[i].to(dtype=torch.float).cpu().numpy()))
    for k in result_dict.keys():
        result_dict[k] = sorted(result_dict[k], key=lambda x: x[0], reverse=True)
        X_train += [x[1] for x in result_dict[k][:20]]
    X_train = np.array(X_train)
    
    X_test = []
    result_dict = {}
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, targets, _ in val_data_loader:
            images = images.to(device)
            output = model(images)
            pred = output.softmax(1)
            for i in range(len(pred)):
                label = targets[i].item()
                if label not in result_dict:
                    result_dict[label] = []
                result_dict[label].append((pred[i][label].item(), images[i].to(dtype=torch.float).cpu().numpy()))
    for k in result_dict.keys():
        result_dict[k] = sorted(result_dict[k], key=lambda x: x[0], reverse=True)
        X_test += [x[1] for x in result_dict[k][:500]]
    X_test = np.array(X_test)   
    
    summarized_background = shap.kmeans(X_train, 180)  # Use a subset of your data for the explainer
    explainer = shap.KernelExplainer(model_predict, summarized_background)  # Use a subset of your data for the explainer
    shap_values = explainer.shap_values(X_test)  # X_test is your test dataset
    
    # Save the SHAP values to a csv file
    shap_values_dict = {}
    OSP_index = [42, 34, 16, 230, 95, 243, 218, 181, 11, 193]
    for i, shap_array in enumerate(shap_values):
        shap_values_dict["channel_index_{}".format(OSP_index[i])] = shap_array
    
    sio.savemat('shap_values.mat', {**shap_values_dict, 'X_test': X_test})
        
    
    feature_names = [f'Feature {i}' for i in range(X_test.shape[1])]
    class_names = ["Road", "Road marks", "Vegetation", "Painted metal",
                   "Sky", "Concrete/Stone/Brick", "Pedestrian/Cyclist",
                   "Unpainted Metal", "Glass/Transparent Plastic"]
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, class_names=class_names, plot_size=(10, 8))
    # Adjust the labels and spacing
    plt.gca().yaxis.label.set_size(12)
    plt.gca().tick_params(axis='y', labelsize=12)
    plt.gca().xaxis.label.set_size(12)
    plt.gca().tick_params(axis='x', labelsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    plt.savefig("shap_summary_plot.png")
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data_path', default='/data2/chaoyi/HSI_Dataset/HSI Drive/v1/', help='dataset')    # /data2/chaoyi/HSI_Dataset/HSI Drive/v1/   /data2/chaoyi/HSI_Dataset/HSI Drive/Image_dataset/
    parser.add_argument('--img_type', default='ALL', help='image type: OSP or ALL or rgb')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    
    parser.add_argument('--use_MF', default=True, type=bool, help='use MF')
    parser.add_argument('--use_dual', default=True, type=bool, help='use dual')
    parser.add_argument('--use_OSP', default=True, type=bool, help='use OSP')
    parser.add_argument('--use_raw', default=False, type=bool, help='use raw')
    parser.add_argument('--use_cache', default=True, type=bool, help='use cache')
    parser.add_argument('--use_rgb', default=False, type=bool, help='use rgb')
    
    parser.add_argument('--use_attention', default=False, type=bool, help='use attention')
    parser.add_argument('--use_large_mlp', default=True, type=bool, help='use large mlp')
    parser.add_argument('--num_attention', default=10, type=int, help='num_attention')
    
    parser.add_argument('--use_sr', default=False, type=bool, help='use sr')
    parser.add_argument('--cal_IoU', default=True, type=bool, help='calculate IoU')

    parser.add_argument('--device', default='cuda', help='device')

    parser.add_argument('--num-classes', default=9, type=int, help='num_classes')

    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')

    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')

    parser.add_argument('--momentum', default=0.8, type=float, metavar='M',
                        help='momentum') 

    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--print-freq', default=5, type=int, help='print frequency')

    parser.add_argument('--output-dir', default='./Pixel_MLP/multi_train/OSP', help='path where to save')

    parser.add_argument('--resume', default='./Pixel_MLP/multi_train/HSI_drive/OSP/model_499.pth', help='resume from checkpoint')

    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )


    parser.add_argument('--world-size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    
    main(args)