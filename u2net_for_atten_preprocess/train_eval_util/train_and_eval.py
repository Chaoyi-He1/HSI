import math
import torch
from torch.nn import functional as F
import train_eval_util.distributed_utils as utils
import torch.nn as nn


def custom_loss(output, target, model, lambda1, lambda2):
    criterion = nn.CrossEntropyLoss()  # Or any other criterion
    basic_loss = criterion(output, target)
    
    # L1 regularization term to encourage sparsity
    l1_regularization = lambda1 * sum(torch.abs(param).sum() for param in model.parameters())
    
    # Custom penalty term to encourage weights to be close to 0 or 1
    penalty = lambda2 * sum(((param - 0.5).abs() - 0.5).abs().sum() for param in model.parameters())
    
    return basic_loss + l1_regularization + penalty


def train_one_epoch():
    pass