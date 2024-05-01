import torch
from torch import nn
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import train_utils.distributed_utils as utils


def criterion(inputs, target, model, num_classes=6):
    losses = nn.functional.binary_cross_entropy_with_logits(inputs, target) if torch.max(target) <= 1 \
        else nn.functional.cross_entropy(inputs, target, ignore_index=255)
    accuracy = torch.mean(((inputs > 0) == target.byte()).float()) if torch.max(target) <= 1 \
        else torch.mean((inputs.argmax(-1) == target).float())
    
    # # L1 norm for model.atten
    # L1_norm = 0.8 * torch.mean(torch.abs(model.module.atten))
    
    # # Return losses with L1_norm if model is in training mode
    # if model.module.training:
    #     if model.module.atten.requires_grad:
    #         return losses + L1_norm, accuracy
    #     else:
    #         return losses, accuracy
    # else:
    #     return losses, accuracy
    return losses, accuracy


def evaluate(model, data_loader, device, num_classes, scaler=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    header = 'Test:'
    all_preds, all_labels = [], []
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
        for image, target, _ in metric_logger.log_every(data_loader, 10, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            # output = output['out']
            loss, acc = criterion(output, target, model)
            
            # store the predictions and labels
            all_preds.append(output.argmax(-1).view(-1, 1).cpu().numpy().astype(int))
            all_labels.append(target.view(-1, 1).cpu().numpy())
            
            metric_logger.update(loss=loss.item(), acc=acc.item())
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    all_preds, all_labels = np.vstack(all_preds), np.vstack(all_labels)
    # remove the 255 ground truth label
    all_preds, all_labels = all_preds[all_labels != 255], all_labels[all_labels != 255]
    # Check which label is missing in all_preds
    missing_labels = [i for i in range(num_classes) if i not in all_labels]
    if len(missing_labels) > 0:
        print(f"Missing labels: {missing_labels}")
    confusion_matrix_total = confusion_matrix(all_labels, all_preds)
    classes = ["Unlabeled", "Road", "Road marks", "Painted Metal", "Pedestrian/Cyclist"]
    # classes = ["Sky", "Background"]
    df_cm = pd.DataFrame(confusion_matrix_total / \
                            (np.sum(confusion_matrix_total, axis=1)[:, None] + \
                                (np.sum(confusion_matrix_total, axis=1) == 0).astype(int)[:, None]), 
                         index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 10))
    fig = sn.heatmap(df_cm, annot=True).get_figure()

    return metric_logger.meters['loss'].global_avg, metric_logger.meters['acc'].global_avg, fig


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None, num_classes=6):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    all_preds, all_labels = [], []

    for image, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        # target = torch.squeeze(target, dim=1)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss, acc = criterion(output, target, model)
            assert not torch.isnan(loss), 'Model diverged with loss = NaN'
            
            all_labels.append(target.view(-1, 1).cpu().numpy())
            all_preds.append(output.argmax(-1).view(-1, 1).cpu().numpy())

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
        metric_logger.update(loss=loss.item(), lr=lr, acc=acc.item())
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    all_preds, all_labels = np.vstack(all_preds), np.vstack(all_labels)
    missing_labels = [i for i in range(num_classes) if i not in all_labels]
    if len(missing_labels) > 0:
        print(f"Missing labels: {missing_labels}")
    # remove the 255 ground truth label
    all_preds, all_labels = all_preds[all_labels != 255], all_labels[all_labels != 255]
    confusion_matrix_total = confusion_matrix(all_labels, all_preds)
    classes = ["Unlabeled", "Road", "Road marks", "Painted Metal", "Pedestrian/Cyclist"]
    # classes = ["Sky", "Background"]
    df_cm = pd.DataFrame(confusion_matrix_total / \
                            (np.sum(confusion_matrix_total, axis=1)[:, None] + \
                                (np.sum(confusion_matrix_total, axis=1) == 0).astype(int)[:, None]), 
                         index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 10))
    fig = sn.heatmap(df_cm, annot=True).get_figure()
    
    return metric_logger.meters["loss"].global_avg, \
           metric_logger.meters["acc"].global_avg, lr, fig


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
