import time
import os
import datetime

import torch

from src import lraspp_mobilenetv3_large, lraspp_mobilenetv3_small
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
from my_dataset import HSI_Segmentation
import transforms as T
from torch.utils.tensorboard import SummaryWriter


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.8 * base_size)
        max_size = int(1.05 * base_size)

        trans = [T.ToTensor()]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.RandomResize(min_size, max_size),
            # T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.RandomResize(base_size, base_size),
            # T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 1400
    crop_size = 1400

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


def create_model(num_classes, large=True, pretrain=True, in_chans=10, freeze_bn=True):
    model = lraspp_mobilenetv3_large(num_classes=num_classes, in_channels=in_chans) \
    if large else lraspp_mobilenetv3_small(num_classes=num_classes, in_channels=in_chans)

    if pretrain:
        weights_dict = torch.load("./lraspp/lraspp_mobilenet_v3_large.pth", map_location='cpu')
        if num_classes != 21:
            # The official pre-training weights are 21 categories (including background)
            # If you train your own data set, delete the weights related to the category 
            # to prevent inconsistent weight shapes from reporting errors
            for k in list(weights_dict.keys()):
                if "low_classifier" in k or "high_classifier" in k:
                    del weights_dict[k]
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
    if freeze_bn:
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
    
    return model


def main(args):
    init_distributed_mode(args)
    print(args)
    if args.rank in [-1, 0]:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(comment=os.path.join("runs", args.img_type, args.name))

    device = torch.device(args.device)
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # load train data set
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
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn, drop_last=True)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    print("Creating model")
    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=num_classes, in_chans=3 if args.img_type == "rgb" else 10)
    model.to(device)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=False)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
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
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes, scaler=scaler)
        acc_global, acc, iu = confmat.compute()
        val_info = str(confmat)
        print(val_info)

        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            if tb_writer:
                tags = ['global_correct', 
                        'average_class_correct/road', 'average_class_correct/sidewalk', 'average_class_correct/building', 
                        'average_class_correct/wall', 'average_class_correct/fence', 'average_class_correct/pole', 
                        'average_class_correct/traffic light', 'average_class_correct/traffic sign', 'average_class_correct/vegetation', 
                        'average_class_correct/terrain', 'average_class_correct/sky', 'average_class_correct/person',
                        'average_class_correct/rider', 'average_class_correct/car', 'average_class_correct/truck',
                        'average_class_correct/bus', 'average_class_correct/train', 'average_class_correct/motorcycle',
                        'average_class_correct/bicycle', 'average_class_correct/background',
                        
                        'IoU/road', 'IoU/sidewalk', 'IoU/building', 
                        'IoU/wall', 'IoU/fence', 'IoU/pole', 
                        'IoU/traffic light', 'IoU/traffic sign', 'IoU/vegetation', 
                        'IoU/terrain', 'IoU/sky', 'IoU/person',
                        'IoU/rider', 'IoU/car', 'IoU/truck',
                        'IoU/bus', 'IoU/train', 'IoU/motorcycle',
                        'IoU/bicycle', 'IoU/background',
                        'mean_IoU']
                values = [acc_global.item() * 100] + [i for i in (acc * 100).tolist()] + \
                         [i for i in (iu * 100).tolist()] + [iu.mean().item() * 100]
                for x, tag in zip(values, tags):
                    tb_writer.add_scalar(tag, x, epoch)
                    
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n"
                f.write(train_info + val_info + "\n\n")

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

    # 训练文件的根目录(VOCdevkit)
    parser.add_argument('--train_data_path', default='/data2/chaoyi/HSI Dataset/V2/train/', help='dataset')
    parser.add_argument('--val_data_path', default='/data2/chaoyi/HSI Dataset/V2/test/', help='dataset')
    parser.add_argument('--label_type', default='gray', help='label type: gray or viz')
    parser.add_argument('--img_type', default='OSP', help='image type: OSP or PCA or rgb')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=19, type=int, help='num_classes')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    # 是否使用同步BN(在多个GPU之间同步)，默认不开启，开启后训练速度会变慢
    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 训练学习率，这里默认设置成0.0001，如果效果不好可以尝试加大学习率
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./lraspp/multi_train/OSP/', help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='./lraspp/multi_train/OSP/model_0167.pth', help='resume from checkpoint')
    # 不训练，仅测试
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # 分布式进程数
    parser.add_argument('--world-size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
