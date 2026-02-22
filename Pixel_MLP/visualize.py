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
from sklearn.metrics import confusion_matrix
import train_utils.distributed_utils as utils
import seaborn as sn


def create_model(model_name="mlp_pixel", num_classes=2, in_chans=10, large=True):
    model = get_model(model_name, num_classes=num_classes, in_channels=in_chans, large=large)
    return model


def overlay_labels(img, labels, true_label, endmember_label_color, hsi_drive_label, transparency):
    mask = true_label != 255
    unique_labels = np.unique(labels[mask])
    for label in unique_labels:
        label_mask = (labels == label) & mask
        img[label_mask] = endmember_label_color[hsi_drive_label[label]] * (1 - transparency) + img[label_mask] * transparency
    # Black color for unknown
    img[true_label == 255] = np.array([0, 0, 0]) * (1 - transparency) + img[true_label == 255] * transparency
    
    return img


def draw_vertical_color_bar(fig, endmember_label_color):
    num_labels = len(endmember_label_color)
    color_bar = np.zeros((num_labels * 25, 25, 3), dtype=np.uint8)  # Adjust height scaling for better fit
    
    for i, (label, color) in enumerate(endmember_label_color.items()):
        color_bar[i * 25:(i + 1) * 25, :] = color
    
    # Create a new axis for the color bar on the right side
    ax_bar = fig.add_axes([0.885, 0.15, 0.02, 0.7])
    ax_bar.imshow(color_bar, aspect='equal')
    ax_bar.set_yticks(np.arange(12.5, num_labels * 25, 25))
    ax_bar.set_yticklabels(endmember_label_color.keys(), fontsize=10)
    ax_bar.xaxis.set_visible(False)
    ax_bar.yaxis.tick_right()

        
def main(args):
    device = torch.device(args.device)
    
    whole_img_dataset = HSI_Drive_V1_visual(data_path=args.data_path,
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
        0: "Road",  # gray
        1: "Road marks",    # yellow
        2: "Vegatation",    # green
        3: "Painted metal", # red
        4: "Sky",        # blue
        5: "Concrete/Stone/Brick",  # brown
        6: "Pedestrian/Cyclist",    # pink
        7: "Unpainted metal",   # purple
        8: "Glass/Transparent Plastic", # light blue
        255: "Unknown"
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
    
    num_classes = args.num_classes
    
    model = create_model(num_classes=num_classes, in_chans=in_chans, large=args.use_large_mlp)
    # model.to(device)
    
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    
    # export to onnx
    model.eval()
    dummy_input = torch.randn(1, 1, 1, 10)
    torch.onnx.export(model, dummy_input,
                    "./cnn_mnist_raw.onnx",
                    export_params=True, opset_version=16, dynamic_axes={'input': {0: 'batch_size'}, 'sensor_out': {0: 'batch_size'}},
                    input_names=['input'], output_names=['sensor_out'])
    model.to(device)
    
    confmat = utils.ConfusionMatrix(num_classes, device)
    
    # add a gaussian noise to the model each layer
    # for name, param in model.named_parameters():
    #     noise = torch.randn_like(param) * 0.001  # Adjust the noise scale as needed
    #     param.data += noise.to(device)
        
    save_data, save_label, save_rgb, save_pred = [], [], [], []
    for image, target, img_pos, rgb_img, name in val_data_loader:
        if not ("nf3231_102" in name or "nf4332_167" in name):
            continue
        # transform img to tensor, use model to predict in gpu
        image, target = image.to(device), target.to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            pred = model(image) # pred: (-1, 10, dim)
            
        pred = torch.argmax(pred, dim=-1)

        confmat.update(target, pred)

        # accuracy print
        acc = (pred[target != 255] == target[target != 255]).float().mean().item()
        print(f"Image: {name[0]}, Accuracy: {acc:.4f}")
        
        # append the data to save_data, save_label, save_rgb
        save_data.append(image.cpu().numpy())
        save_label.append(target.cpu().numpy())
        save_rgb.append(rgb_img)
        save_pred.append(pred.cpu().numpy())
        
        img_pos = np.reshape(img_pos, (216, 409, 2))
        img_original_label = target.view(216, 409).cpu().numpy()
        pred = pred.view(216, 409).cpu().numpy()
        rgb_img = rgb_img.reshape(216, 409, 3)  # np.array, shape (H, W, C)
        
        # visualize the prediction, use the rgb image as background, with transparency factor adjustable
        # plot 2 results in one figure, subplot one is the "img_original_label" (ground truth), the other subplot is the "pred" (prediction)
        # the color of the prediction is the same as the ground truth according to the "endmember_label_color"
        # the position of the prediction is the same as the ground truth according to the "img_pos"
        transparency = 0.3
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        
        # subplot 1, use rgb image as background and plot the ground truth on it, label is in shape (219, 406) with position in shape (219, 406, 2)
        # add the color according to the "endmember_label_color" to the rgb image at the position of the label, ignore the "Unknown" (255) ground truth label
        # make the rgb image transparent by the "transparency"
        img_to_plot = overlay_labels(rgb_img.copy(), img_original_label, img_original_label, endmember_label_color, hsi_drive_label, transparency)
        ax[0].imshow(img_to_plot)
        ax[0].set_title("Ground Truth")
        ax[0].axis('off')
        
        # subplot 2, use rgb image as background and plot the prediction on it, label is in shape (-1,) woth position in shape (-1, 2)
        # add the color according to the "endmember_label_color" to the rgb image at the position of the label
        # make the rgb image transparent by the "transparency"
        img_to_plot = overlay_labels(rgb_img.copy(), pred, img_original_label, endmember_label_color, hsi_drive_label, transparency)
        ax[1].imshow(img_to_plot)
        ax[1].set_title("Prediction")
        ax[1].axis('off')
        
        # add color's label name to the figure, add a color bar under the figure filled with the color of each label
        # annotate the color's label name under the color bar
        draw_vertical_color_bar(fig, endmember_label_color)
        
        # for i, (label, color) in enumerate(endmember_label_color.items()):
            # ax[0].text(10, 150 + 13 * i, label, color=color / 255, fontsize=15)
            # ax[1].text(10, 150 + 13 * i, label, color=color / 255, fontsize=15)
        
        # save the figure
        plt.subplots_adjust(left=0.05, right=0.88, top=0.9, bottom=0.1, wspace=0.05) 
        plt.savefig(os.path.join(args.output_dir, name[0] + ".png"))
        plt.close(fig)
        # save the pred as uint8 image in .png format in the same folder's "pred" subfolder
        if not os.path.exists(os.path.join(args.output_dir, "pred")):
            os.makedirs(os.path.join(args.output_dir, "pred"))
        Image.fromarray(pred.astype(np.uint8)).save(os.path.join(args.output_dir, "pred", name[0] + ".png"))
    
    # save the data, label, rgb as npz 
    save_data = np.concatenate(save_data, axis=0)
    save_label = np.concatenate(save_label, axis=0)
    save_rgb = np.concatenate(save_rgb, axis=0)
    save_pred = np.concatenate(save_pred, axis=0)
    
    # generate the confusion matrix and calculate mean IoU
    val_info = str(confmat)
    print(val_info)
    # save cfmtx as a svg picture
    confusion_matrix_total = confusion_matrix(save_label[save_label != 255], save_pred[save_label != 255])
    classes = ["Road", "Road marks", "Vegetation", "Painted Metal", "Sky", "Concrete/Stone/Brick", "Pedestrian/Cyclist", "Unpainted Metal", "Glass/Transparent Plastic"]
    # classes = ["Sky", "Background"]
    df_cm = pd.DataFrame(confusion_matrix_total / \
                            (np.sum(confusion_matrix_total, axis=1)[:, None] + \
                                (np.sum(confusion_matrix_total, axis=1) == 0).astype(int)[:, None]), 
                         index=[i for i in classes],
                         columns=[i for i in classes])
    # save the df_cm as a .csv file
    df_cm.to_csv(os.path.join(args.output_dir, "confusion_matrix.csv"))
    plt.figure(figsize=(12, 10))
    fig = sn.heatmap(df_cm, annot=True).get_figure()
    fig.savefig(os.path.join(args.output_dir, "confusion_matrix.svg"))
    plt.close(fig)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data_path', default='/data2/chaoyi/HSI_Dataset/HSI Drive/v1/', help='dataset')    # /data2/chaoyi/HSI_Dataset/HSI Drive/v1/   /data2/chaoyi/HSI_Dataset/HSI Drive/Image_dataset/
    parser.add_argument('--img_type', default='ALL', help='image type: OSP or ALL or rgb')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    
    parser.add_argument('--use_MF', default=True, type=bool, help='use MF')
    parser.add_argument('--use_dual', default=True, type=bool, help='use dual')
    parser.add_argument('--use_OSP', default=False, type=bool, help='use OSP')
    parser.add_argument('--use_raw', default=False, type=bool, help='use raw')
    parser.add_argument('--use_cache', default=False, type=bool, help='use cache')
    parser.add_argument('--use_rgb', default=False, type=bool, help='use rgb')
    
    parser.add_argument('--use_attention', default=True, type=bool, help='use attention')
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

    parser.add_argument('--output-dir', default='./Pixel_MLP/multi_train/', help='path where to save')

    parser.add_argument('--resume', default='./Pixel_MLP/multi_train/HSI_drive/rgb/large/model_156.pth', help='resume from checkpoint')

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