import time
import os
import datetime

import torch

from src import u2net_full, u2net_lite
from train_eval_util import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
from my_dataset import HSI_Segmentation, HSI_Transformer, HSI_Transformer_all
from torch.utils.tensorboard import SummaryWriter
import transforms as T
import numpy as np
import pandas as pd


def create_model(in_chans, num_classes):
    model = u2net_lite(in_ch=in_chans, out_ch=num_classes)
    return model


def save_conv_weights(model, save_path):
    conv_layer = model.pre_process_conv.weight.data
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    for out_ch in range(conv_layer.shape[0]):
        save_mtx = conv_layer[out_ch, :, :, :].cpu().numpy()
        save_mtx = np.reshape(save_mtx, (save_mtx.shape[0] * save_mtx.shape[1], save_mtx.shape[2]))
        file_name = os.path.join(save_path, 'conv_{}.csv'.format(out_ch))
        if os.path.exists(file_name):
            os.remove(file_name)
        pd.DataFrame(save_mtx).to_csv(file_name, index=False, header=False)


def main(args):
    print(args)
    device = torch.device(args.device)

    num_classes = args.num_classes

    print("Creating model")
    # create model num_classes equal background + 20 classes
    model = create_model(in_chans=10, num_classes=num_classes)
    model.to(device)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume.endswith(".pth"):
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model.load_state_dict(checkpoint['model'])

    save_conv_weights(model, os.path.join(args.output_dir, 'conv_weights'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--train_data_path', default='/data2/chaoyi/HSI_Dataset/V2/train/', help='dataset')
    parser.add_argument('--val_data_path', default='/data2/chaoyi/HSI_Dataset/V2/test/', help='dataset')
    parser.add_argument('--label_type', default='Treelabel', help='label type: gray or viz')    # Car_white_label, Car_black_label
    parser.add_argument('--img_type', default='OSP', help='image type: OSP or PCA or rgb')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')

    parser.add_argument('--device', default='cuda', help='device')

    parser.add_argument('--num-classes', default=6, type=int, help='num_classes')
    parser.add_argument('--lambda1', default=0.4, type=float, help='lambda1')
    parser.add_argument('--lambda2', default=0.2, type=float, help='lambda2')

    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')

    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')

    parser.add_argument('--momentum', default=0.4, type=float, metavar='M',
                        help='momentum') 

    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')

    parser.add_argument('--output-dir', default='./u2net_for_atten_preprocess/multi_train/OSP/', help='path where to save')

    parser.add_argument('--resume', default='./u2net_for_atten_preprocess/multi_train/OSP/model_056.pth', help='resume from checkpoint')

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


    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
