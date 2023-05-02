import argparse
import yaml
from distribute_utils import *
from torch.utils.tensorboard import SummaryWriter
from model import *
from dataset import *
import os
import numpy as np
import pandas as pd
from train_utils import *
from torchvision import transforms


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        Returns a learning rate multiplication factor based on the number of steps,
        Note that before the training starts, pytorch will call the lr_scheduler.step() method in advance
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l1_coeff', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--clip_max_norm', default=0.4, type=float, help='gradient clipping max norm')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')

    # dataset parameters
    parser.add_argument('--hyp', type=str, default='./Support_Vector_Regression/cfg.yaml', help='hyper parameters path')
    parser.add_argument('--train_data_path', default='/data2/chaoyi/HSI Dataset/V2/train/', help='dataset')
    parser.add_argument('--val_data_path', default='/data2/chaoyi/HSI Dataset/V2/train/', help='dataset')
    parser.add_argument('--filter_path', default='./Matlab Code/EC_filter.mat', help='label type: gray or viz')
    parser.add_argument('--img_type', default='ALL', help='image type: OSP or PCA or rgb or raw')
    parser.add_argument('--output_dir', default='./SVR_weights/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='./SVR_weights/model_30.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    # distributed training parameters
    parser.add_argument('--savebest', type=bool, default=False, help='only save best checkpoint')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")
    return parser


def main(args, cfg):
    init_distributed_mode(args)
    print(args)
    if args.rank in [-1, 0]:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(comment=os.path.join("runs", "svr", args.img_type, args.name))

    device = torch.device(args.device)
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # load train data set
    train_dataset = HSI_Segmentation(data_path=args.train_data_path,
                                     filter_path=args.filter_path,
                                     img_type=args.img_type,
                                     transforms=transforms.ToTensor())
    # load validation data set
    val_dataset = HSI_Segmentation(data_path=args.val_data_path,
                                   filter_path=args.filter_path,
                                   img_type=args.img_type,
                                   transforms=transforms.ToTensor(), train=False)
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn, drop_last=True)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        sampler=test_sampler, num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn)

    print("Creating model")
    # create model num_classes equal background + 20 classes
    model = SVR_net(cfg)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.Adam(
        params_to_optimize, lr=args.lr)
    criterion = nn.MSELoss().to(device)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=True)

    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

        for params in model_without_ddp.state_dict():
            weight = model_without_ddp.state_dict()[params].cpu().numpy()
            file_name = args.resume.replace('.pth', '') + '_%s.csv' % (params.replace(".", "_"))
            pd.DataFrame(weight).to_csv(file_name, header=False, index=False)

    print("Start training")
    start_time = time.time()
    best_loss = np.float('inf')
    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = train_one_epoch(model=model, criterion=criterion, lr_scheduler=lr_scheduler,
                                        data_loader=train_data_loader, optimizer=optimizer, device=device,
                                        epoch=epoch, print_freq=args.print_freq, l1_coeff=args.l1_coeff, scaler=scaler)

        mean_loss_eval = evaluate(model=model, criterion=criterion, device=device, data_loader=val_data_loader,
                                  print_freq=args.print_freq, scaler=scaler)

        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            if tb_writer:
                tags = ['Learning Rate', 'train mean loss', 'val mean loss']
                values = [lr, mean_loss, mean_loss_eval]
                for x, tag in zip(values, tags):
                    tb_writer.add_scalar(tag, x, epoch)
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n"
                f.write(train_info + 'val mean loss' + str(mean_loss_eval) + "\n\n")

        if args.output_dir:
            save_file = {'model': model_without_ddp.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'lr_scheduler': lr_scheduler.state_dict(),
                         'args': args,
                         'epoch': epoch}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            if args.savebest and mean_loss_eval <= best_loss:
                save_on_master(save_file,
                               os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
                best_loss = mean_loss_eval
            elif not args.savebest:
                save_on_master(save_file,
                               os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SVR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    main(args, hyp)
