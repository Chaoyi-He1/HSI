import time
import os
import datetime
from typing import Union, List

import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from src import *
from train_utils import (train_one_epoch, evaluate, init_distributed_mode, save_on_master, mkdir,
                         create_lr_scheduler, get_params_groups)
from my_dataset import HSI_Segmentation
import transforms as T


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.8 * base_size)
        max_size = int(1.05 * base_size)

        trans = [T.ToTensor()]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            # T.RandomResize(min_size, max_size), 
            # T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            # T.RandomResize(base_size, base_size),
            # T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 1400
    crop_size = 1400

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


def main(args):
    init_distributed_mode(args)
    print(args)
    if args.rank in [-1, 0]:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(comment=os.path.join("runs", args.img_type, args.name))
    device = torch.device(args.device)

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = HSI_Segmentation(data_path=args.train_data_path,
                                    label_type=args.label_type,
                                    img_type=args.img_type,
                                    transforms=get_transform(train=True))
    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = HSI_Segmentation(data_path=args.val_data_path,
                                   label_type=args.label_type,
                                   img_type=args.img_type,
                                   transforms=get_transform(train=False))

    print("Creating data loaders")
    if args.distributed:
        train_sampler = data.distributed.DistributedSampler(train_dataset)
        test_sampler = data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = data.RandomSampler(train_dataset)
        test_sampler = data.SequentialSampler(val_dataset)

    train_data_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=train_dataset.collate_fn, drop_last=True)

    val_data_loader = data.DataLoader(
        val_dataset, batch_size=1,  # batch_size must be 1
        sampler=test_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=train_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    in_ch = 10 if args.img_type != "rgb" else 3
    model = u2net_lite(in_ch=in_ch, out_ch=20)
    model.to(device)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=2)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        mae_metric, f1_metric = evaluate(model, val_data_loader, device=device)
        print(mae_metric, f1_metric)
        return

    print("Start training")
    current_mae, current_f1 = 1.0, 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        save_file = {'model': model_without_ddp.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     'args': args,
                     'epoch': epoch}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            mae_metric, confmat = evaluate(model, val_data_loader, num_classes=20, device=device, scaler=scaler)
            mae_info, (acc_global, acc, iu) = mae_metric.compute(), confmat.compute()
            print(f"[epoch: {epoch}] val_MAE: {mae_info:.3f} ", str(confmat))

            # 只在主进程上进行写操作
            if args.rank in [-1, 0]:
                if tb_writer:
                    tags = ['global_correct', 
                            'average_class_correct/background', 'average_class_correct/car', 'average_class_correct/human', 
                            'average_class_correct/road', 'average_class_correct/traffic_light', 'average_class_correct/traffic_sign', 
                            'average_class_correct/tree', 'average_class_correct/building', 'average_class_correct/sky', 
                            'average_class_correct/object',
                            'IoU/Background', 'IoU/Car', 'IoU/Human', 'IoU/Road', 'IoU/Traffic_light', 
                            'IoU/Traffic_sign', 'IoU/Tree', 'IoU/Building', 'IoU/Sky', 'IoU/Object',
                            'mean_IoU']
                    values = [acc_global.item() * 100] + [i for i in (acc * 100).tolist()] + \
                            [i for i in (iu * 100).tolist()] + [iu.mean().item() * 100]
                    for x, tag in zip(values, tags):
                        tb_writer.add_scalar(tag, x, epoch)
                # write into txt
                with open(results_file, "a") as f:
                    # 记录每个epoch对应的train_loss、lr以及验证集各指标
                    write_info = f"[epoch: {epoch}] train_loss: {mean_loss:.4f} lr: {lr:.6f} " \
                                 f"MAE: {mae_info:.3f} \n" + str(confmat)
                    f.write(write_info + "\n\n")

                # save_best
                if current_mae >= mae_info:
                    if args.output_dir:
                        # 只在主节点上执行保存权重操作
                        save_on_master(save_file,
                                       os.path.join(args.output_dir, 'model_best.pth'))

        if args.output_dir:
            if args.rank in [-1, 0]:
                # only save latest 10 epoch weights
                if os.path.exists(os.path.join(args.output_dir, f'model_{epoch - 10}.pth')):
                    os.remove(os.path.join(args.output_dir, f'model_{epoch - 10}.pth'))

            # 只在主节点上执行保存权重操作
            save_on_master(save_file,
                           os.path.join(args.output_dir, f'model_{epoch}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练文件的根目录
    parser.add_argument('--train_data_path', default='/data2/chaoyi/HSI Dataset/V2/train/', help='dataset')
    parser.add_argument('--val_data_path', default='/data2/chaoyi/HSI Dataset/V2/val/', help='dataset')
    parser.add_argument('--label_type', default='gray', help='label type: gray or viz')
    parser.add_argument('--img_type', default='OSP', help='image type: OSP or PCA or rgb')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=360, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 是否使用同步BN(在多个GPU之间同步)，默认不开启，开启后训练速度会变慢
    parser.add_argument('--sync-bn', action='store_true', help='whether using SyncBatchNorm')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 训练学习率
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')
    # 验证频率
    parser.add_argument("--eval-interval", default=1, type=int, help="validation interval default 10 Epochs")
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./u2net/multi_train/OSP', help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 不训练，仅测试
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # 分布式进程数
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # Mixed precision training parameters
    parser.add_argument("--amp", action='store_false',
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
