from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .transformer_model import build_Transformer_Encoder


class MLP_Pixel(nn.Module):
    def __init__(self, in_nodes=3, num_class=18, large=True) -> None:
        super(MLP_Pixel, self).__init__()
        if large:
            self.layers = nn.Sequential(
                nn.Linear(in_nodes, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_class)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_nodes, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_class)
            )
        
        # self.atten = nn.Parameter(torch.ones(1, in_nodes))
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        # x = x * self.atten
        return self.layers(x)


def get_model(model_name: str = "mlp_pixel", num_classes: int = 18, in_channels: int = 3, large: bool = True) -> nn.Module:
    if model_name == "mlp_pixel":
        model = MLP_Pixel(in_channels, num_classes, large)
    elif model_name == "transformer":
        model = build_Transformer_Encoder(num_cls=num_classes, d_model=in_channels)
    else:
        raise NotImplementedError(f"model {model_name} is not implemented in `get_model`")
    return model
