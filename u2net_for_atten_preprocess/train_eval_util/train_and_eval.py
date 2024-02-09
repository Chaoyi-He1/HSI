import math
import torch
from torch.nn import functional as F
import train_eval_util.distributed_utils as utils
import torch.nn as nn
from typing import Iterable


def custom_loss(output, target, model, lambda1, lambda2):
    basic_loss = [F.cross_entropy(output[i], target.squeeze(1)) for i in range(len(output))]
    basic_loss_sum = sum(basic_loss)
    
    # L1 regularization term to encourage sparsity
    l1_regularization = lambda1 * torch.norm(model.module.pre_process_conv.weight, p=1)
    
    # Custom penalty term to encourage weights to be close to 0 or 1
    penalty = lambda2 * torch.mean(torch.abs(torch.abs(model.module.pre_process_conv.weight - 0.5) - 0.5))
    
    # Calculate the accuracy of each pixel
    accuracy = (output[0].argmax(1) == target).float().mean()
    
    return basic_loss_sum + l1_regularization + penalty, accuracy


def train_one_epoch(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    data_loader: Iterable,
                    device: torch.device,
                    epoch: int,
                    lr_scheduler: torch.optim.lr_scheduler,
                    print_freq: int = 10,
                    scaler=None,
                    lambda1: float = 0.1,
                    lambda2: float = 0.1):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss, accuracy = custom_loss(output, target, model, lambda1, lambda2)
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=accuracy.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    metric_logger.synchronize_between_processes()
    return metric_logger.meters["loss"].global_avg, metric_logger.meters["acc"].global_avg, optimizer.param_groups[0]["lr"]


def evaluate(model: nn.Module,
             data_loader: Iterable,
             num_classes: int,
             device: torch.device,
             scaler=None):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes, device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Test:'
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            
            _, accuracy = custom_loss(output, targets, model, 0.1, 0.1)
            basic_loss = F.cross_entropy(output, targets)

            metric_logger.update(loss=basic_loss.item())
            metric_logger.update(acc=accuracy.item())
            confmat.update(targets.flatten(), output.argmax(1).flatten())
        
        metric_logger.synchronize_between_processes()
        confmat.reduce_from_all_processes()
    return metric_logger.meters['loss'].global_avg, metric_logger.meters['acc'].global_avg, confmat
        


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=False,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
