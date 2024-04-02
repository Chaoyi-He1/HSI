import torch
from torch import nn
import train_utils.distributed_utils as utils


def criterion(inputs, target, model):
    losses = nn.functional.binary_cross_entropy_with_logits(inputs, target) if torch.max(target) <= 1 \
        else nn.functional.cross_entropy(inputs.transpose(1, 2), target.squeeze(-1))
    accuracy = torch.mean(((inputs > 0) == target.byte()).float()) if torch.max(target) <= 1 \
        else torch.mean((inputs.argmax(-1) == target.squeeze(-1)).float())
    
    # L1 norm for model.atten
    L1_norm = 0.8 * torch.mean(torch.abs(model.module.atten))
    
    # Return losses with L1_norm if model is in training mode
    if model.module.training:
        if model.module.atten.grad is not None:
            return losses + L1_norm, accuracy
        else:
            return losses, accuracy
    else:
        return losses, accuracy


def evaluate(model, data_loader, device, num_classes, scaler=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Test:'
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
        for image, target in metric_logger.log_every(data_loader, 10, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            # output = output['out']
            loss, acc = criterion(output, target.unsqueeze(-1), model)
            
            metric_logger.update(loss=loss.item(), acc=acc.item())

    return metric_logger.meters['loss'].global_avg, metric_logger.meters['acc'].global_avg


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device).unsqueeze(-1)
        # target = torch.squeeze(target, dim=1)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss, acc = criterion(output, target, model)
            assert not torch.isnan(loss), 'Model diverged with loss = NaN'

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

    return metric_logger.meters["loss"].global_avg, \
           metric_logger.meters["acc"].global_avg, lr


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
