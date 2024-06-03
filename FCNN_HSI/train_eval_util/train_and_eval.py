import math
import torch
from torch.nn import functional as F
import train_eval_util.distributed_utils as utils
import torch.nn as nn
from typing import Iterable
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def custom_loss(output, target, model, lambda1, lambda2, is_train=True):
    basic_loss = F.cross_entropy(output, target, ignore_index=255)
    
    pre_conv_trainable = any(param.requires_grad for param in model.pre_process_conv.parameters())
    if pre_conv_trainable:
        # L1 regularization term to encourage sparsity 
        # and subtract the max value among the in_channels to make sure the rest of the values among the in_channels stay at 0
        l1_regularization = torch.sum(torch.abs(model.pre_process_conv.weight))
        # max_value = torch.sum(torch.max(torch.abs(model.module.pre_process_conv), dim=1)[0])
        # l1_regularization -= max_value
        l1_regularization *= lambda1 / (model.pre_process_conv.weight.numel())
        
        # # Get the max value position of each in_channels, keep the same 4D shape as model.module.pre_process_conv.weight
        # Channel_max_value_index = torch.max(torch.abs(model.pre_process_conv.weight), dim=1, keepdim=True)[1]
        # # get a boolean tensor shape same as model.module.pre_process_conv.weight, where the max value position is 0, otherwise 1
        # non_max_value_index = torch.ones_like(model.pre_process_conv.weight, dtype=torch.bool, device=model.pre_process_conv.weight.device)
        # non_max_value_index.scatter_(1, Channel_max_value_index, 0)
        
        # # Custom penalty term to encourage weights to be close to 0 if is not the max value position, otherwise 1
        # penalty = lambda2 * (torch.mean(torch.abs(model.pre_process_conv.weight) * non_max_value_index) + 
        #                     torch.mean(torch.abs(1 - model.pre_process_conv.weight) * (~non_max_value_index)))
        
        basic_loss = basic_loss + l1_regularization
    
    # Calculate the accuracy of each pixel
    accuracy = (output.argmax(1)[target != 255] == target[target != 255]).float().mean()
    
    return basic_loss, accuracy


def train_one_epoch(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    data_loader: Iterable,
                    device: torch.device,
                    epoch: int,
                    lr_scheduler: torch.optim.lr_scheduler,
                    print_freq: int = 10,
                    scaler=None,
                    lambda1: float = 0.1,
                    lambda2: float = 0.1,
                    num_classes: int = 8):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    all_preds, all_labels = [], []
    
    for image, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss, accuracy = custom_loss(output, target, model, lambda1, lambda2)

            all_labels.append(target.view(-1, 1).cpu().numpy())
            all_preds.append(output.argmax(1).view(-1, 1).cpu().numpy())
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=accuracy.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    all_preds, all_labels = np.vstack(all_preds), np.vstack(all_labels)
    missing_labels = [i for i in range(num_classes) if i not in all_labels]
    if len(missing_labels) > 0:
        print(f"Missing labels: {missing_labels}")
    # remove the 255 ground truth label
    all_preds, all_labels = all_preds[all_labels != 255], all_labels[all_labels != 255]
    confusion_matrix_total = confusion_matrix(all_labels, all_preds)
    classes = ["Road", "Road marks", "Vegetation", "Painted Metal", "Sky", "Concrete/Stone/Brick", "Pedestrian/Cyclist", "Unpainted Metal", "Glass/Transparent Plastic"]
    # classes = ["Sky", "Background"]
    df_cm = pd.DataFrame(confusion_matrix_total / \
                            (np.sum(confusion_matrix_total, axis=1)[:, None] + \
                                (np.sum(confusion_matrix_total, axis=1) == 0).astype(int)[:, None]), 
                         index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 10))
    fig = sn.heatmap(df_cm, annot=True).get_figure()
    
    return metric_logger.meters["loss"].global_avg, metric_logger.meters["acc"].global_avg, \
           optimizer.param_groups[0]["lr"], fig


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
    all_preds, all_labels = [], []
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
        for images, targets, _ in metric_logger.log_every(data_loader, 10, header):
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            
            _, accuracy = custom_loss(output, targets, model, 0.1, 0.1, False)
            basic_loss = F.cross_entropy(output, targets, ignore_index=255)
            
            # store the predictions and labels
            all_preds.append(output.argmax(1).view(-1, 1).cpu().numpy())
            all_labels.append(targets.view(-1, 1).cpu().numpy())

            metric_logger.update(loss=basic_loss.item())
            metric_logger.update(acc=accuracy.item())
            confmat.update(targets[targets != 255].flatten(), output.argmax(1)[targets != 255].flatten())
        
    metric_logger.synchronize_between_processes()
    confmat.reduce_from_all_processes()
    print("Averaged stats:", metric_logger)
    
    all_preds, all_labels = np.vstack(all_preds), np.vstack(all_labels)
    # remove the 255 ground truth label
    all_preds, all_labels = all_preds[all_labels != 255], all_labels[all_labels != 255]
    # Check which label is missing in all_preds
    missing_labels = [i for i in range(num_classes) if i not in all_labels]
    if len(missing_labels) > 0:
        print(f"Missing labels: {missing_labels}")
    confusion_matrix_total = confusion_matrix(all_labels, all_preds)
    classes = ["Road", "Road marks", "Vegetation", "Painted Metal", "Sky", "Concrete/Stone/Brick", "Pedestrian/Cyclist", "Unpainted Metal", "Glass/Transparent Plastic"]
    # classes = ["Sky", "Background"]
    df_cm = pd.DataFrame(confusion_matrix_total / \
                            (np.sum(confusion_matrix_total, axis=1)[:, None] + \
                                (np.sum(confusion_matrix_total, axis=1) == 0).astype(int)[:, None]), 
                         index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 10))
    fig = sn.heatmap(df_cm, annot=True).get_figure()
    
    return metric_logger.meters['loss'].global_avg, metric_logger.meters['acc'].global_avg, confmat, fig
        


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=False,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
