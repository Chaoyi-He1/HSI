import math
import torch
from torch.nn import functional as F
import train_eval_util.distributed_utils as utils
import torch.nn as nn
from types import Iteratable


def custom_loss(output, target, model, lambda1, lambda2):
    basic_loss = F.cross_entropy(output, target)
    
    # L1 regularization term to encourage sparsity
    l1_regularization = lambda1 * sum(torch.abs(param).sum() for param in model.parameters())
    
    # Custom penalty term to encourage weights to be close to 0 or 1
    penalty = lambda2 * sum(((param - 0.5).abs() - 0.5).abs().sum() for param in model.parameters())
    
    # Calculate the accuracy of each pixel
    accuracy = (output.argmax(1) == target).float().mean()
    
    return basic_loss + l1_regularization + penalty, accuracy


def train_one_epoch(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    data_loader: Iteratable,
                    device: torch.device,
                    epoch: int,
                    lr_scheduler: torch.optim.lr_scheduler,
                    print_freq: int = 10,
                    scaler=None):
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
            loss, accuracy = custom_loss(output, target, model, 0.1, 0.1)
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=accuracy(output, target))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
