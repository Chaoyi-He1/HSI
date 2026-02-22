import time
import os
import datetime
import random
import torch

from model.whole_model import get_model
from train import train_one_epoch, evaluate
from my_dataset import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import math
import transforms as T


class SegmentationCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.transforms = T.Compose([
            T.ToTensor(),
            T.RandomCrop(self.crop_size),
        ])
    
    def __call__(self, target_list):
        return self.transforms(target_list)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/data2/chaoyi/HSI_Dataset/HSI Drive/v1/', type=str,
                        help='dataset path')
    parser.add_argument('--img_type', default='ALL', help='image type: OSP or ALL or rgb')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    
    parser.add_argument('--use_MF', default=True, type=bool, help='use MF')
    parser.add_argument('--use_dual', default=True, type=bool, help='use dual')
    parser.add_argument('--use_cache', default=True, type=bool, help='use cache')
    
    parser.add_argument('--use_OSP', default=False, type=bool, help='use OSP')
    parser.add_argument('--use_raw', default=False, type=bool, help='use raw')
    parser.add_argument('--use_rgb', default=False, type=bool, help='use rgb')
    parser.add_argument('--use_attention', default=False, type=bool, help='use attention')
    parser.add_argument('--use_large_mlp', default=True, type=bool, help='use large mlp')
    parser.add_argument('--num_attention', default=10, type=int, help='num_attention')
    
    parser.add_argument('--use_sr', default=False, type=bool, help='use sr')
    parser.add_argument('--cal_IoU', default=True, type=bool, help='calculate IoU')

    parser.add_argument('--device', default='cuda', help='device')

    parser.add_argument('--num-classes', default=9, type=int, help='num_classes')
    
    # Training parameters
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')

    parser.add_argument('--lf', default=0.1, type=float,
                        help='learning rate decay factor')

    parser.add_argument('--print-freq', default=7, type=int, help='print frequency')

    parser.add_argument('--output-dir', default='./Up_seg/HSI_drive/atten/', help='path where to save')

    parser.add_argument('--resume', default='./Pixel_MLP/multi_train/HSI_drive/OSP/model_200', help='resume from checkpoint')
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    
    # Model parameters
    parser.add_argument('--in-channels', default=252, type=int,
                        help='input channels')
    parser.add_argument('--hidden-size', default=512, type=int)
    parser.add_argument('--in-h', default=48, type=int)
    parser.add_argument('--in-w', default=96, type=int)
    parser.add_argument('--patch-size', default=2, type=int)
    parser.add_argument('--num-heads', default=8, type=int)
    parser.add_argument('--num-layers', default=4, type=int)
    parser.add_argument('--upconv-in-ch', default=4, type=int)
    parser.add_argument('--img-h', default=192, type=int)
    parser.add_argument('--img-w', default=384, type=int)
    parser.add_argument('--transformer-scale', default=2, type=int,
                        help='scale of the transformer output size compared to the input size')
    parser.add_argument('--down-sample-rate', default=2, type=int,
                        help='down sample rate of the input image, 1 means no down sampling')
    
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    
    args = parser.parse_args()
    return args

def main(args):
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.output_dir)
    
    # Load dataset
    dataset = HSI_Drive_V1_(args.data_path, args.use_MF, args.use_dual, args.use_OSP,
                            args.use_raw, args.use_rgb, args.use_cache, args.use_attention,
                            args.use_large_mlp, args.num_attention,
                            transform=SegmentationCrop((192, 384)), down_sample_rate=args.down_sample_rate)
    
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    if args.use_rgb:
        args.in_channels = 3
    elif args.use_OSP:
        args.in_channels = args.num_attention
    elif args.use_attention:
        args.in_channels = args.num_attention
    elif not args.use_OSP and args.use_dual and not args.use_raw:
        args.in_channels = 252
    elif not args.use_OSP and not args.use_dual and not args.use_raw:
        args.in_channels = 71
    elif args.use_raw:
        args.in_channels = 25
    model = get_model(args).to(device)
    
    # Define lr scheduler and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lf) + args.lf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    start_epoch = 0
    scheduler.last_epoch = start_epoch - 1
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    
    # Resume from checkpoint if specified
    if args.resume.endswith('.pth'):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc}")
    
    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc, \
            lr, cfmtx_fig, cfmtx = train_one_epoch(model, 
                                                   optimizer, 
                                                   train_loader, 
                                                   device, 
                                                   epoch, 
                                                   print_freq=args.print_freq,
                                                   scaler=scaler,
                                                   num_classes=args.num_classes)
        _, _, iu_train = cfmtx.compute()
        train_info = str(cfmtx)
        print("Train cfmtx: \n", train_info)
        scheduler.step()
        val_loss, val_acc, \
            val_cfmtx_fig, val_cfmtx = evaluate(model, 
                                                val_loader, 
                                                device,
                                                num_classes=args.num_classes,
                                                scaler=scaler)
        _, acc_val, iu_val = val_cfmtx.compute()
        val_info = str(val_cfmtx)
        print("Val cfmtx: \n", val_info)
        if writer:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/acc', val_acc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_figure('train/confusion_matrix', cfmtx_fig, epoch)
            writer.add_figure('val/confusion_matrix', val_cfmtx_fig, epoch)

            tags = ['IoU/Road', 'IoU/Road marks', 'IoU/Vegetation', 'IoU/Painted Metal',
                    'IoU/Sky', 'IoU/Concrete or Stone or Brick', 'IoU/Pedestrian or Cyclist',
                    'IoU/Unpainted Metal', 'IoU/Glass or Transparent Plastic',
                    'mean_IoU']
            values = [i for i in (iu_val * 100).tolist()] + [iu_val.mean().item() * 100]
            for x, tag in zip(values, tags):
                writer.add_scalar(tag, x, epoch)
                
        # Save the model if it has the best accuracy so far
        save_file = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'lr_scheduler': scheduler.state_dict(),
                     'args': args,
                     'epoch': epoch}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        digits = len(str(args.epochs))
        torch.save(save_file,
                    os.path.join(args.output_dir, 'model_{}.pth'.format(str(epoch).zfill(digits))))
        
    atten_weights = model.channel_select.detach().cpu().numpy()
    np.savetxt('atten_weights_lite_mlp.csv', atten_weights, delimiter=',')
 
if __name__ == '__main__':
    args = parse_args()
    main(args)
    print("Training complete.")   