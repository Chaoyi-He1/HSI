import argparse
import datetime
import os
import time
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from torch.utils.tensorboard import SummaryWriter

from src import get_model
from test_dataset import HSI_Drive_V1_
from train_utils import create_lr_scheduler, evaluate, evaluate_sr, train_one_epoch

CLASS_NAMES: List[str] = [
    "Road",
    "Road marks",
    "Vegetation",
    "Painted Metal",
    "Sky",
    "Concrete/Stone/Brick",
    "Pedestrian/Cyclist",
    "Unpainted Metal",
    "Glass/Transparent Plastic",
]


def create_model(model_name: str, num_classes: int, in_chans: int, large: bool) -> torch.nn.Module:
    return get_model(model_name, num_classes=num_classes, in_channels=in_chans, large=large)


def parse_in_chans(args: argparse.Namespace) -> int:
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
    raise ValueError("Could not determine input channels with the current flags.")


def make_r_list(args: argparse.Namespace) -> Iterable[int]:
    if args.R_list is not None:
        return args.R_list
    return list(range(args.R_start, args.R_end + 1))


def pure_test(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    in_chans = parse_in_chans(args)
    # in_chans = 252

    model = create_model(model_name="mlp_pixel", num_classes=args.num_classes, in_chans=in_chans, large=args.use_large_mlp)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    results = []
    r_indices = make_r_list(args)

    for r_idx in r_indices:
        dataset = HSI_Drive_V1_(
            data_path=args.data_path,
            use_MF=args.use_MF,
            use_dual=args.use_dual,
            use_OSP=args.use_OSP,
            use_raw=args.use_raw,
            use_cache=args.use_cache,
            use_rgb=args.use_rgb,
            use_attention=args.use_attention,
            use_large_mlp=args.use_large_mlp,
            num_attention=args.num_attention,
            R_idx=r_idx,
            for_train=False,
        )
        _, val_dataset = random_split(
        dataset,
        [
            int(0.8 * len(dataset)),
            len(dataset) - int(0.8 * len(dataset)),
        ],
    )

        data_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=SequentialSampler(val_dataset),
            num_workers=args.workers,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )

        loss, acc, fig, confmat = evaluate(
            model,
            data_loader,
            device=device,
            num_classes=args.num_classes,
            scaler=scaler,
            epoch=0,
            IoU=True,
        )

        # Close the confusion matrix figure to avoid accumulating open figures.
        plt.close(fig)

        acc_global, acc_class, iu = confmat.compute()
        result = {
            "R_idx": r_idx,
            "loss": loss,
            "acc": acc_global.item(),
            "mean_IoU": iu.mean().item(),
        }
        for i, name in enumerate(CLASS_NAMES):
            result[f"IoU/{name}"] = iu[i].item()
        results.append(result)
        print(f"R {r_idx:02d}: acc={acc_global.item() * 100:.2f}%, mIoU={iu.mean().item() * 100:.2f}%")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        pd.DataFrame(results).to_csv(args.output_csv, index=False)
        print(f"Saved IoU results to {args.output_csv}")
    
    # Print summary
    print("\nSummary of IoU results:")
    df = pd.DataFrame(results)
    print(df)


def train_main(args) -> None:
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = args.num_classes
    epochs = getattr(args, "epochs", 1)
    start_epoch = getattr(args, "start_epoch", 0)
    use_sr = getattr(args, "use_sr", False)
    cal_IoU = getattr(args, "cal_IoU", False)
    print_freq = getattr(args, "print_freq", 50)

    log_dir = getattr(
        args,
        "tb_log_dir",
        f"runs/HSI_drive/R_sweep/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    tb_writer = SummaryWriter(log_dir=log_dir)

    results_file = getattr(
        args,
        "results_file",
        f"results{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt",
    )

    dataset = HSI_Drive_V1_(
        data_path=args.data_path,
        use_MF=args.use_MF,
        use_dual=args.use_dual,
        use_OSP=args.use_OSP,
        use_raw=args.use_raw,
        use_cache=args.use_cache,
        use_rgb=args.use_rgb,
        use_attention=args.use_attention,
        use_large_mlp=args.use_large_mlp,
        num_attention=args.num_attention,
        R_idx=getattr(args, "R_idx", 1),
        for_train=False,
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [
            int(0.8 * len(dataset)),
            len(dataset) - int(0.8 * len(dataset)),
        ],
    )

    train_sampler = RandomSampler(train_dataset)
    test_sampler = SequentialSampler(val_dataset)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size if not use_sr else 1,
        sampler=test_sampler,
        num_workers=args.workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    in_chans = parse_in_chans(args)
    # in_chans = 252
    model = create_model(
        model_name="mlp_pixel",
        num_classes=num_classes,
        in_chans=in_chans,
        large=args.use_large_mlp,
    )
    model.to(device)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_layers = len(list(model.parameters()))
    print(f"Number of parameters: {num_parameters}, number of layers: {num_layers}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=getattr(args, "lr", 0.001),
        weight_decay=getattr(args, "weight_decay", 0.0),
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = create_lr_scheduler(optimizer, 1, epochs, warmup=False)

    resume_path = getattr(args, "resume", "")
    if isinstance(resume_path, str) and resume_path.endswith(".pth") and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint.get("epoch", start_epoch - 1) + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if getattr(args, "test_only", False):
        if cal_IoU:
            _, _, _, confmat = evaluate(
                model,
                val_data_loader,
                device=device,
                num_classes=num_classes,
                scaler=scaler,
                epoch=start_epoch,
                IoU=True,
            )
            print(confmat)
        else:
            _, _, _ = evaluate(
                model,
                val_data_loader,
                device=device,
                num_classes=num_classes,
                scaler=scaler,
                epoch=start_epoch,
                IoU=False,
            )
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, epochs + start_epoch):
        mean_loss, mean_acc, lr, confusion_mtx = train_one_epoch(
            model,
            optimizer,
            train_data_loader,
            device,
            epoch,
            lr_scheduler=lr_scheduler,
            print_freq=print_freq,
            scaler=scaler,
            num_classes=num_classes,
        )
        lr_scheduler.step()

        if use_sr:
            if epoch <= 290:
                loss_val, acc_val, confusion_mtx_val = evaluate_sr(
                    model,
                    val_data_loader,
                    device=device,
                    num_classes=num_classes,
                    scaler=scaler,
                    epoch=epoch,
                )
            else:
                loss_val, acc_val, confusion_mtx_val, confusion_mtx_val_sr = evaluate_sr(
                    model,
                    val_data_loader,
                    device=device,
                    num_classes=num_classes,
                    scaler=scaler,
                    epoch=epoch,
                )
        elif cal_IoU:
            loss_val, acc_val, confusion_mtx_val, confmat = evaluate(
                model,
                val_data_loader,
                device=device,
                num_classes=num_classes,
                scaler=scaler,
                epoch=epoch,
                IoU=True,
            )
            acc_global, acc_class, iu = confmat.compute()
            val_info = str(confmat)
            print(val_info)
            print(f"Validation Epoch {epoch}: Global Acc {acc_global.item() * 100:.2f}%, Mean IoU {iu.mean().item() * 100:.2f}%")
        else:
            loss_val, acc_val, confusion_mtx_val = evaluate(
                model,
                val_data_loader,
                device=device,
                num_classes=num_classes,
                scaler=scaler,
                epoch=epoch,
                IoU=False,
            )

        if tb_writer:
            tags = ["train_loss", "train_acc", "val_loss", "val_acc"]
            values = [mean_loss, mean_acc, loss_val, acc_val]

            if cal_IoU:
                tags += [
                    "IoU/Road",
                    "IoU/Road marks",
                    "IoU/Vegetation",
                    "IoU/Painted Metal",
                    "IoU/Sky",
                    "IoU/Concrete or Stone or Brick",
                    "IoU/Pedestrian or Cyclist",
                    "IoU/Unpainted Metal",
                    "IoU/Glass or Transparent Plastic",
                    "mean_IoU",
                ]
                values += [i for i in (iu * 100).tolist()] + [iu.mean().item() * 100]

            for x, tag in zip(values, tags):
                tb_writer.add_scalar(tag, x, epoch)
            tb_writer.add_figure("confusion_matrix", confusion_mtx, epoch)
            tb_writer.add_figure("confusion_matrix_val", confusion_mtx_val, epoch)
            if use_sr and epoch > 290:
                tb_writer.add_figure("confusion_matrix_val_sr", confusion_mtx_val_sr, epoch)

        with open(results_file, "a") as f:
            train_info = (
                f"[epoch: {epoch}]\n"
                f"train_loss: {mean_loss:.4f}\n"
                f"lr: {lr:.6f}\n"
            )
            f.write(train_info + "\n\n")

        output_dir = getattr(args, "output_dir", "")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            digits = len(str(epochs))
            torch.save(
                save_file,
                os.path.join(output_dir, f"model_{str(epoch).zfill(digits)}.pth"),
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    params = [p.detach().cpu().numpy().flatten() for p in model.parameters() if p.requires_grad]
    plt.figure(figsize=(10, 5))
    plt.hist(np.hstack(params), bins=100)
    plt.title("Model Parameters Histogram")
    plt.xlabel("Parameter Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(os.path.dirname(__file__), "model_parameters_histogram.png"))


def test_args():
    parser = argparse.ArgumentParser(description="Evaluate IoU on HSI Drive test set for different R matrices.")

    parser.add_argument("--data_path", default="/data2/chaoyi/HSI_Dataset/HSI Drive/v1/", help="HSI Drive dataset root.")
    parser.add_argument("--resume", default="./Pixel_MLP/multi_train/HSI_drive/atten/large/R_1/model_499.pth", help="Checkpoint to evaluate.")
    parser.add_argument("--device", default="cuda", help="Computation device.")
    parser.add_argument("--num_classes", default=9, type=int, help="Number of classes.")
    parser.add_argument("-b", "--batch-size", default=512, type=int, dest="batch_size", help="Batch size for evaluation.")
    parser.add_argument("-j", "--workers", default=4, type=int, dest="workers", help="Dataloader workers.")
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP during inference.")
    parser.add_argument("--no-amp", action="store_false", dest="amp")

    parser.add_argument("--use_MF", default=True, type=bool)
    parser.add_argument("--use_dual", default=True, type=bool)
    parser.add_argument("--use_OSP", default=False, type=bool)
    parser.add_argument("--use_raw", default=False, type=bool)
    parser.add_argument("--use_cache", default=True, type=bool)
    parser.add_argument("--use_rgb", default=False, type=bool)
    parser.add_argument("--use_attention", default=True, type=bool)
    parser.add_argument("--use_large_mlp", default=True, type=bool)
    parser.add_argument("--num_attention", default=10, type=int)

    parser.add_argument("--R_start", default=1, type=int, help="Start index for R matrix sweep (inclusive).")
    parser.add_argument("--R_end", default=32, type=int, help="End index for R matrix sweep (inclusive).")
    parser.add_argument("--R_list", nargs="+", type=int, default=None, help="Explicit R index list (overrides start/end).")
    parser.add_argument("--output_csv", default="./Pixel_MLP/R_IoU.csv", help="Where to store IoU results.")

    args = parser.parse_args()
    return args

def train_args():
    parser = argparse.ArgumentParser(description="Train Pixel MLP on HSI Drive dataset.")

    parser.add_argument("--data_path", default="/data2/chaoyi/HSI_Dataset/HSI Drive/v1/", help="HSI Drive dataset root.")
    parser.add_argument("--device", default="cuda", help="Computation device.")
    parser.add_argument("--num_classes", default=9, type=int, help="Number of classes.")

    parser.add_argument("-b", "--batch-size", default=512, type=int, dest="batch_size", help="Batch size per epoch.")
    parser.add_argument("-j", "--workers", default=8, type=int, dest="workers", help="Dataloader workers.")

    parser.add_argument("--start_epoch", default=0, type=int, help="Starting epoch.")
    parser.add_argument("--epochs", default=500, type=int, help="Total epochs.")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument("--wd", "--weight-decay", dest="weight_decay", default=0.0, type=float, help="Weight decay.")
    parser.add_argument("--print_freq", default=50, type=int, help="Logging frequency.")

    parser.add_argument("--output_dir", default="./Pixel_MLP/multi_train/HSI_drive/atten/large/R_1/", help="Path to save checkpoints.")
    parser.add_argument("--resume", default="", help="Checkpoint to resume from.")
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP during training.")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Only run evaluation.")

    parser.add_argument("--use_MF", default=True, type=bool)
    parser.add_argument("--use_dual", default=True, type=bool)
    parser.add_argument("--use_OSP", default=False, type=bool)
    parser.add_argument("--use_raw", default=False, type=bool)
    parser.add_argument("--use_cache", default=True, type=bool)
    parser.add_argument("--use_rgb", default=False, type=bool)
    parser.add_argument("--use_attention", default=True, type=bool)
    parser.add_argument("--use_large_mlp", default=True, type=bool)
    parser.add_argument("--num_attention", default=10, type=int)
    parser.add_argument("--R_idx", default=1, type=int, help="R matrix index to use for training/validation.")

    parser.add_argument("--use_sr", default=False, type=bool, help="Use super-resolution evaluation path.")
    parser.add_argument("--cal_IoU", default=True, type=bool, help="Calculate IoU during validation.")

    parser.add_argument(
        "--tb_log_dir",
        default=None,
        help="TensorBoard log directory (default uses timestamped runs/HSI_drive/R_sweep/ path).",
    )
    parser.add_argument(
        "--results_file",
        default=f"results{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt",
        help="Optional path to write training metrics.",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    test = True
    if test:
        args = test_args()
    else:
        args = train_args()
    
    if not test:
        train_main(args)
    else:
        pure_test(args)
    
    # # Print out the ./Pixel_MLP/multi_train/HSI_drive/atten/large/R_1/model_499.pth model's atten's top 10 weights' indices
    # checkpoint = torch.load("./Pixel_MLP/multi_train/HSI_drive/atten/large/R_1/model_499.pth", map_location="cpu", weights_only=False)
    # model_state = checkpoint["model"]
    # atten_weights = model_state["atten"].numpy()    # (1, 252)
    # print("Attention weights:", atten_weights.shape)
    # top10_indices = np.argsort(atten_weights[0])[-10:][::-1]
    # print("Top 10 attention weight indices:", top10_indices)
