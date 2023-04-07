import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SVR_net(nn.Module):
    def __init__(self, cfg: dict):
        super(SVR_net, self).__init__()
        self.linear = nn.Linear(in_features=cfg["num_in_feature"], out_features=cfg["num_out_feature"], bias=cfg["bias"])

    def forward(self, inputs: Tensor):
        return self.linear(inputs)
