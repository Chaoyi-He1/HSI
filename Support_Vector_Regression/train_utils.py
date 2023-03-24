import datetime
import sys
from typing import Iterable
import time
import torch


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, print_freq: int, scaler=None):
    model.train()
    criterion.train()
    total_time = time.time()
    ave_loss = torch.tensor(0.).to(device)
    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = targets.to(device)
        iter_time = time.time()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(samples)
            loss = criterion(outputs, targets)
            ave_loss += loss

        if not torch.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_freq == 0 or i == len(data_loader) - 1:
            iter_time = time.time() - iter_time
            log_msg = "; ".join([
                'Epoch: [{}]'.format(epoch),
                '[{0' + ':' + str(len(str(len(data_loader)))) + 'd' + '}/{1}]',
                '{meters}',
                'time: {time}',
                'max mem: {memory:.0f}'
            ])
            if torch.cuda.is_available():
                print(log_msg.format(
                    i, len(data_loader),
                    meters=str(loss),
                    time=str(iter_time),
                    memory=torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)))
    total_time = time.time() - total_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(
        'Epoch: [{}]'.format(epoch), total_time_str, total_time / len(data_loader)))
    return ave_loss.item() / len(data_loader)


def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, device: torch.device,
             data_loader: Iterable, print_freq: int, scaler=None):
    model.eval()
    ave_loss = torch.tensor(0.).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
        for i, (image, target) in enumerate(data_loader):
            image, target = image.to(device), target.to(device)
            iter_time = time.time()
            output = model(image)
            loss = criterion(output, target)
            ave_loss += loss

            if i % print_freq == 0 or i == len(data_loader) - 1:
                iter_time = time.time() - iter_time
                log_msg = "; ".join([
                    'Test:',
                    '[{0' + ':' + str(len(str(len(data_loader)))) + 'd' + '}/{1}]',
                    '{meters}',
                    'time: {time}',
                    'max mem: {memory:.0f}'
                ])
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(data_loader),
                        meters=str(loss),
                        time=str(iter_time),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)))
    return ave_loss.item() / len(data_loader)
