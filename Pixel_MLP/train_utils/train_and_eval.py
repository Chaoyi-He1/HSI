import torch
from torch import nn
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import train_utils.distributed_utils as utils
from scipy.ndimage import generic_filter


def criterion(inputs, target, model, num_classes=6):
    losses = nn.functional.binary_cross_entropy_with_logits(inputs, target) if inputs.shape[-1] <= 1 \
        else nn.functional.cross_entropy(inputs, target, ignore_index=255)
    accuracy = torch.mean(((inputs > 0) == target.byte()).float()) if inputs.shape[-1] <= 1 \
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


def evaluate(model, data_loader, device, num_classes, scaler=None, epoch=0):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    header = 'Test:'
    all_preds, all_preds_sr, all_labels = [], [], []
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
        for image, target, img_pos in metric_logger.log_every(data_loader, 10, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            
            loss, acc = criterion(output, target, model)
            
            if epoch > 290:
                pred_labels = Spacial_Regularization_v2(output, img_pos)
                acc = torch.mean((pred_labels.view(-1) == target.view(-1)).float())
                
                # store the predictions and labels
                all_preds.append(output.argmax(-1).view(-1, 1).cpu().numpy().astype(int))
                all_labels.append(target.view(-1, 1).cpu().numpy())
                all_preds_sr.append(pred_labels.view(-1, 1).cpu().numpy().astype(int))
            else:
                all_preds.append(output.argmax(-1).view(-1, 1).cpu().numpy())
                all_labels.append(target.view(-1, 1).cpu().numpy())
            
            metric_logger.update(loss=loss.item(), acc=acc.item())
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    all_preds, all_labels = np.vstack(all_preds), np.vstack(all_labels)
    all_labels_ = all_labels.copy()
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
    
    # confusion_matrix_total_sr
    if epoch > 290:
        all_preds_sr = np.vstack(all_preds_sr)
        all_preds_sr = all_preds_sr[all_labels_ != 255]
        
        confusion_matrix_total_sr = confusion_matrix(all_labels, all_preds_sr)
        df_cm_sr = pd.DataFrame(confusion_matrix_total_sr / \
                                (np.sum(confusion_matrix_total_sr, axis=1)[:, None] + \
                                    (np.sum(confusion_matrix_total_sr, axis=1) == 0).astype(int)[:, None]), 
                            index=[i for i in classes],
                            columns=[i for i in classes])
        plt.figure(figsize=(12, 10))
        fig_sr = sn.heatmap(df_cm_sr, annot=True).get_figure()

        return metric_logger.meters['loss'].global_avg, metric_logger.meters['acc'].global_avg, fig, fig_sr
    
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
    classes = ["Road", "Road marks", "Vegetation", "Painted Metal", "Sky", "Concrete/Stone/Brick", "Pedestrian/Cyclist", "Unpainted Metal", "Glass/Transparent Plastic"]
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


def Spacial_Regularization(pred: torch.Tensor, pos_idx: np.array):
    '''
    first, the pixels classified with a low confidence (ANN output values below 0.8) 
    were re-labeled as "don't know", 
    and then the SR process assigned a new label to those pixels 
    by a majority voting criterion over 5x5 pixel windows
    
    pred: torch.Tensor, shape (-1, num_classes), unnormalized prediction
    pos_idx: np.array, shape (-1, 2), the position of the pixels
    '''
    # Softmax the prediction
    pred = torch.softmax(pred, dim=-1)
    # Get the prediction label
    pred_label = pred.argmax(dim=-1)
    # Get the prediction value
    pred_value = pred.max(dim=-1)[0]
    # Get the position of the pixels
    pos_idx = pos_idx.astype(int)
    # Pick the pixels with low confidence
    low_confidence = pred_value < 0.8
    # Get the position of the low confidence pixels
    low_confidence_pos = pos_idx[low_confidence.cpu().numpy()]
    # Re-label the low confidence pixels with a 5x5 window majority voting
    for i in range(len(low_confidence_pos)):
        x, y = low_confidence_pos[i]
        # find the pixels index in the 5x5 window
        window_idx = []
        for j in range(-2, 3):
            for k in range(-2, 3):
                # find all 5x5 window pixels index from pos_idx
                idx = np.where((pos_idx == [x+j, y+k]).all(axis=1))[0]
                if idx.size > 0:
                    window_idx.append(torch.from_numpy(idx))
        # Get the majority voting label from the 5x5 window
        window_idx = torch.stack(window_idx).view(-1)
        majority_voting = pred_label[window_idx].mode().values.item()
        # Re-label the low confidence pixel
        pred_label[np.where((pos_idx == [x, y]).all(axis=1))[0]] = majority_voting
    return pred_label
    

def Spacial_Regularization_v2(pred: torch.Tensor, pos_idx: np.array):
    '''
    first, the pixels classified with a low confidence (ANN output values below 0.8) 
    were re-labeled as "don't know", 
    and then the SR process assigned a new label to those pixels 
    by a majority voting criterion over 5x5 pixel windows
    
    pred: torch.Tensor, shape (-1, num_classes), unnormalized prediction
    pos_idx: np.array, shape (-1, 2), the position of the pixels
    '''
    # Softmax the prediction
    pred = torch.softmax(pred, dim=-1)
    # Get the prediction label
    pred_label = pred.argmax(dim=-1)
    # Get the prediction value
    pred_value = pred.max(dim=-1)[0]
    
    # Get the position of the pixels
    pos_idx = pos_idx.astype(int)
    
    # Pick the pixels with low confidence
    low_confidence = pred_value < 0.8
    # Get the position of the low confidence pixels
    low_confidence_pos = pos_idx[low_confidence.cpu().numpy()]

    # Create a 2D array to hold the labels
    label_map = -1 * np.ones((pos_idx[:, 0].max() + 1, pos_idx[:, 1].max() + 1), dtype=int)
    label_map[pos_idx[:, 0], pos_idx[:, 1]] = pred_label.cpu().numpy()

    # Define the function to apply for majority voting
    def majority_voting(window):
        window = window[window >= 0]  # Remove the -1 padding
        if len(window) == 0:
            return -1
        values, counts = np.unique(window, return_counts=True)
        return values[np.argmax(counts)]
    
    # Apply majority voting filter
    filtered_labels = generic_filter(label_map, majority_voting, size=5, mode='constant', cval=-1)
    
    # Re-label low confidence pixels
    for x, y in low_confidence_pos:
        majority_label = filtered_labels[x, y]
        if majority_label != -1:
            pred_label[(pos_idx[:, 0] == x) & (pos_idx[:, 1] == y)] = majority_label
    
    return pred_label