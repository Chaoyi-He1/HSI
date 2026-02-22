import argparse
import datetime
import os
import torch
from torch.utils.data import DataLoader, random_split

from src import get_model
from train_utils import evaluate, evaluate_sr, mkdir
from my_dataset import HSI_Drive_V1_, HSI_Drive_V1, stratified_split


def create_model(model_name="mlp_pixel", num_classes=2, in_chans=10, large=True):
    return get_model(model_name, num_classes=num_classes, in_channels=in_chans, large=large)


def build_data_loader(args):
    whole_dataset = HSI_Drive_V1_(data_path=args.data_path,
                                  use_MF=args.use_MF,
                                  use_dual=args.use_dual,
                                  use_OSP=args.use_OSP,
                                  use_raw=args.use_raw,
                                  use_cache=args.use_cache,
                                  use_rgb=args.use_rgb,
                                  use_attention=args.use_attention,
                                  use_large_mlp=args.use_large_mlp,
                                  num_attention=args.num_attention)
    
    if not args.use_cache:
        _, val_dataset = stratified_split(whole_dataset, train_ratio=0.8)
    else:
        _, val_dataset = random_split(whole_dataset, [int(0.8 * len(whole_dataset)),
                                                      len(whole_dataset) - int(0.8 * len(whole_dataset))])
    
    collate_fn = whole_dataset.collate_fn
    
    if args.use_sr:
        whole_img_dataset = HSI_Drive_V1(data_path=args.data_path,
                                         use_MF=args.use_MF,
                                         use_dual=args.use_dual,
                                         use_OSP=args.use_OSP,
                                         use_raw=args.use_raw,
                                         use_cache=False,
                                         use_rgb=args.use_rgb,
                                         use_attention=args.use_attention,
                                         use_large_mlp=args.use_large_mlp,
                                         num_attention=args.num_attention)
        _, val_dataset = stratified_split(whole_img_dataset, train_ratio=0.8)
        collate_fn = whole_img_dataset.collate_fn
    
    sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=args.batch_size if not args.use_sr else 1,
                                 sampler=sampler,
                                 num_workers=args.workers,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    return val_data_loader


def resolve_in_chans(args):
    if args.use_rgb:
        return 3
    if args.use_OSP:
        return args.num_attention
    if args.use_attention:
        return args.num_attention
    if not args.use_OSP and args.use_dual and not args.use_raw:
        return 252
    if not args.use_OSP and not args.use_dual and not args.use_raw:
        return 71
    if args.use_raw:
        return 25
    return 10


def main(args):
    device = torch.device(args.device)
    if args.output_dir:
        mkdir(args.output_dir)
    
    print("Creating validation data loader")
    val_data_loader = build_data_loader(args)
    
    num_classes = args.num_classes
    in_chans = resolve_in_chans(args)
    
    print("Loading model")
    model = create_model(num_classes=num_classes, in_chans=in_chans, large=args.use_large_mlp)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    
    print(f"Evaluating checkpoint: {args.resume}")
    confmat = None
    if args.use_sr:
        loss_val, acc_val, *figs = evaluate_sr(model, val_data_loader, device=device, num_classes=num_classes, scaler=None, epoch=args.eval_epoch)
    else:
        if args.cal_IoU:
            loss_val, acc_val, conf_fig, confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes, scaler=None, epoch=args.eval_epoch, IoU=True)
            figs = [conf_fig]
        else:
            loss_val, acc_val, conf_fig = evaluate(model, val_data_loader, device=device, num_classes=num_classes, scaler=None, epoch=args.eval_epoch, IoU=False)
            figs = [conf_fig]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(args.output_dir, f"test_results_{timestamp}.txt")
    with open(results_file, "w") as f:
        f.write(f"loss: {loss_val:.4f}\nacc: {acc_val:.4f}\n")
        if confmat is not None:
            acc_global, acc, iu = confmat.compute()
            f.write(f"acc_global: {acc_global.item()*100:.2f}\n")
            f.write(f"mean_IoU: {iu.mean().item()*100:.2f}\n")
            f.write(f"per_class_IoU: {iu.tolist()}\n")
    
    for idx, fig in enumerate(figs):
        fig_path = os.path.join(args.output_dir, f"confusion_matrix_{idx}.png")
        fig.savefig(fig_path)
    
    print(f"Validation loss: {loss_val:.4f}, acc: {acc_val:.4f}")
    print(f"Validation IoU: mean IoU: {iu.mean().item()*100:.2f}%")
    print(f"Results written to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained HSI Drive model")
    
    parser.add_argument('--data_path', default='/data2/chaoyi/HSI_Dataset/HSI Drive/v1/', help='dataset path')
    parser.add_argument('--output-dir', default='./Pixel_MLP/multi_train/HSI_drive/test/', help='path to save results')
    parser.add_argument('--resume', default='/data/chaoyi_he/HSI/EC_dataset/Pixel_MLP/multi_train/HSI_drive/atten/model_499.pth', help='checkpoint path')
    
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-classes', default=9, type=int, help='num_classes')
    
    parser.add_argument('--use_MF', default=True, type=bool, help='use MF')
    parser.add_argument('--use_dual', default=True, type=bool, help='use dual')
    parser.add_argument('--use_OSP', default=False, type=bool, help='use OSP')
    parser.add_argument('--use_raw', default=False, type=bool, help='use raw')
    parser.add_argument('--use_cache', default=True, type=bool, help='use cache')
    parser.add_argument('--use_rgb', default=False, type=bool, help='use rgb')
    parser.add_argument('--use_attention', default=True, type=bool, help='use attention')
    parser.add_argument('--use_large_mlp', default=True, type=bool, help='use large mlp')
    parser.add_argument('--num_attention', default=10, type=int, help='num of selected channels')
    
    parser.add_argument('--use_sr', default=False, type=bool, help='use spatial regularization evaluation')
    parser.add_argument('--cal_IoU', default=True, type=bool, help='calculate IoU during evaluation')
    parser.add_argument('--eval_epoch', default=0, type=int, help='epoch index for eval_sr path control')
    
    parser.add_argument('-b', '--batch-size', default=512, type=int, help='batch size for evaluation')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    
    args = parser.parse_args()
    
    if args.output_dir:
        mkdir(args.output_dir)
    
    main(args)
