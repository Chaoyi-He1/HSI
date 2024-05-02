import time
import os
import datetime

import torch

from src import get_model
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
from my_dataset import *
import transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split


def create_model(model_name="mlp_pixel", num_classes=2, in_chans=10):
    model = get_model(model_name, num_classes=num_classes, in_channels=in_chans)
    return model

def main(args):
    init_distributed_mode(args)
    print(args)
    if args.rank in [-1, 0]:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        # Save tb_writer to runs/HSI_drive/times
        tb_writer = SummaryWriter(log_dir="runs/HSI_drive/{}".format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

    device = torch.device(args.device)

    num_classes = args.num_classes

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    print("Creating data loaders")
    # load train data set
    whole_dataset = HSI_Drive(data_path=args.data_path,
                              use_MF=args.use_MF,
                              use_dual=args.use_dual,
                              use_OSP=args.use_OSP,
                              use_raw=args.use_raw,
                              use_cache=args.use_cache)
    if not args.use_cache:
        train_dataset, val_dataset = stratified_split(whole_dataset, train_ratio=0.8)
    else:
        train_dataset, val_dataset = random_split(whole_dataset, [int(0.8*len(whole_dataset)), len(whole_dataset)-int(0.8*len(whole_dataset))])

    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=whole_dataset.collate_fn, drop_last=True)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=whole_dataset.collate_fn)

    print("Creating model")
    if args.img_type == "rgb":
        in_chans = 3
    elif args.use_OSP:
        in_chans = 10
    elif not args.use_OSP and args.use_dual and not args.use_raw:
        in_chans = 252
    elif not args.use_OSP and not args.use_dual and not args.use_raw:
        in_chans = 71
    elif args.use_raw:
        in_chans = 25
    model = create_model(num_classes=num_classes, in_chans=in_chans)
    model.to(device)
    
    num_parameters, num_layers = sum(p.numel() for p in model.parameters() if p.requires_grad), len(list(model.parameters()))
    print(f"Number of parameters: {num_parameters}, number of layers: {num_layers}")

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, 1, args.epochs, warmup=False)

    if args.resume.endswith(".pth"):
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, mean_acc, lr, confusion_mtx = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler, num_classes=num_classes)
        lr_scheduler.step()
        loss_val, acc_val, confusion_mtx_val  = evaluate(model, val_data_loader, device=device, num_classes=num_classes, scaler=scaler)

        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            if tb_writer:
                tags = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
                values = [mean_loss, mean_acc, loss_val, acc_val,]
                for x, tag in zip(values, tags):
                    tb_writer.add_scalar(tag, x, epoch)
                # add confusion matrix to tensorboard
                tb_writer.add_figure('confusion_matrix', confusion_mtx, epoch)
                tb_writer.add_figure('confusion_matrix_val', confusion_mtx_val, epoch)
                    
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n"
                f.write(train_info + "\n\n")

        if args.output_dir:
            # 只在主节点上执行保存权重操作
            save_file = {'model': model_without_ddp.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'lr_scheduler': lr_scheduler.state_dict(),
                         'args': args,
                         'epoch': epoch}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            digits = len(str(args.epochs))
            save_on_master(save_file,
                           os.path.join(args.output_dir, 'model_{}.pth'.format(
                            str(epoch).zfill(digits))))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data_path', default='/data2/chaoyi/HSI_Dataset/HSI Drive/Image_dataset/', help='dataset')
    parser.add_argument('--img_type', default='ALL', help='image type: OSP or ALL or rgb')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    
    parser.add_argument('--use_MF', default=True, type=bool, help='use MF')
    parser.add_argument('--use_dual', default=True, type=bool, help='use dual')
    parser.add_argument('--use_OSP', default=False, type=bool, help='use OSP')
    parser.add_argument('--use_raw', default=True, type=bool, help='use raw')
    parser.add_argument('--use_cache', default=True, type=bool, help='use cache')

    parser.add_argument('--device', default='cuda', help='device')

    parser.add_argument('--num-classes', default=5, type=int, help='num_classes')

    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')

    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')

    parser.add_argument('--momentum', default=0.8, type=float, metavar='M',
                        help='momentum') 

    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--print-freq', default=5, type=int, help='print frequency')

    parser.add_argument('--output-dir', default='./Pixel_MLP/multi_train/HSI_drive/OSP', help='path where to save')

    parser.add_argument('--resume', default='./Pixel_MLP/multi_train/OSP/model_030', help='resume from checkpoint')

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
