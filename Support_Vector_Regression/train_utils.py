import sys
from typing import Iterable
import torch
import distribute_utils as utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, lr_scheduler,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, print_freq: int, scaler=None):
    lr = None
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = targets.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        if not torch.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        # backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, device: torch.device,
             data_loader: Iterable, print_freq: int, scaler=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)

            metric_logger.update(loss=loss.item())
    return metric_logger.meters["loss"].global_avg
