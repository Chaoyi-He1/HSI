from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()

        self.in_ch = in_ch
        self.num_classes = num_classes
        
        self.model = nn.Sequential(
            ConvBNReLU(in_ch, in_ch, kernel_size=3),
            ConvBNReLU(in_ch, 300, kernel_size=3),
            nn.Conv2d(300, num_classes, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def FCNN_lite(in_ch: int, num_classes: int) -> FCNN:
    return FCNN(in_ch, num_classes)


def convert_onnx(m, save_path):
    m.eval()
    x = torch.rand(1, 3, 288, 288, requires_grad=True)

    # export the model
    torch.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      opset_version=11)


if __name__ == '__main__':
    # n_m = RSU(height=7, in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU7.onnx")
    #
    # n_m = RSU4F(in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU4F.onnx")

    u2net = FCNN_lite()
    convert_onnx(u2net, "u2net_full.onnx")
