import torch
from torch import nn
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import misc as utils
from scipy.ndimage import generic_filter


def criterion(inputs, target, model):
    losses = nn.functional.binary_cross_entropy_with_logits(inputs.permute(0, 2, 3, 1).as_contiguous(),
                                                            target, ignore_index=255)
    accuracy = torch.mean((inputs.argmax(1) == target).float())
    
    # # L1 norm for model.atten
    # L1_norm = 0.6 * torch.mean(torch.abs(model.module.atten))
    
    # Return losses with L1_norm if model is in training mode and atten exists
    if model.training and model.channel_select is not None:
        L1_norm = 0.6 * torch.mean(torch.abs(model.channel_select))
        # num_ones = torch.sum(torch.abs(model.atten))
        # deviation = torch.abs(num_ones - 10)
        return losses + L1_norm, accuracy
    return losses, accuracy


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, scaler=None, num_classes=6):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=10, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    all_preds, all_labels = [], []
    
    for image, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            output = model(image)
            loss, acc = criterion(output, target, model)
        
        metric_logger.update(loss=loss.item(), acc=acc.item())
        all_labels.append(target.view(-1, num_classes).cpu().numpy())
        all_preds.append(output.permute(0, 2, 3, 1).view(-1, num_classes).cpu().numpy())
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    metric_logger.synchronize_between_processes()
    
    print("Averaged stats:", metric_logger)
    
    all_preds, all_labels = np.vstack(all_preds), np.vstack(all_labels)
    missing_labels = [i for i in range(num_classes) if i not in all_labels]
    if len(missing_labels) > 0:
        print(f"Missing labels: {missing_labels}")
    
    # For target (B, H, W, C) and output (B, H, W, C), get confusion matrix for each class using the sklearn library
    cfm_figs = []
    for i in range(num_classes):
        target_i = all_labels[:, i]
        output_i = all_preds[:, i]
        # remove the 255 ground truth label
        target_i, output_i = target_i[target_i != 255], output_i[target_i != 255]
        
        confmat = confusion_matrix(target_i, output_i)
        df_cm = pd.DataFrame(confmat / \
                                (np.sum(confmat, axis=1)[:, None] + \
                                (np.sum(confmat, axis=1) == 0).astype(int)[:, None]), 
                                index=[0, 1],
                                columns=[0, 1])
        plt.figure(figsize=(4, 3))
        fig = sn.heatmap(df_cm, annot=True).get_figure()
        cfm_figs.append(fig)
    
    return metric_logger.meters["loss"].global_avg, \
           metric_logger.meters["acc"].global_avg, \
               optimizer.param_groups[0]["lr"], cfm_figs

def evaluate(model, data_loader, device, num_classes=6, scaler=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=10, fmt='{value:.6f}'))
    header = 'Test:'
    all_preds, all_labels = [], []
    
    for image, target, _ in metric_logger.log_every(data_loader, 2, header):
        image, target = image.to(device), target.to(device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                output = model(image)
                loss, acc = criterion(output, target, model)
        
        metric_logger.update(loss=loss.item(), acc=acc.item())
        all_labels.append(target.view(-1, num_classes).cpu().numpy())
        all_preds.append(output.permute(0, 2, 3, 1).view(-1, num_classes).cpu().numpy())
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    all_preds, all_labels = np.vstack(all_preds), np.vstack(all_labels)
    
    # Check which label is missing in all_preds
    missing_labels = [i for i in range(num_classes) if i not in all_labels]
    if len(missing_labels) > 0:
        print(f"Missing labels: {missing_labels}")
        
    # For target (B, H, W, C) and output (B, H, W, C), get confusion matrix for each class using the sklearn library
    cfm_figs = []
    for i in range(num_classes):
        target_i = all_labels[:, i]
        output_i = all_preds[:, i]
        # remove the 255 ground truth label
        target_i, output_i = target_i[target_i != 255], output_i[target_i != 255]
        
        confmat = confusion_matrix(target_i, output_i)
        df_cm = pd.DataFrame(confmat / \
                                (np.sum(confmat, axis=1)[:, None] + \
                                (np.sum(confmat, axis=1) == 0).astype(int)[:, None]), 
                                index=[0, 1],
                                columns=[0, 1])
        plt.figure(figsize=(4, 3))
        fig = sn.heatmap(df_cm, annot=True).get_figure()
        cfm_figs.append(fig)
    
    return metric_logger.meters["loss"].global_avg, \
           metric_logger.meters["acc"].global_avg, \
           cfm_figs