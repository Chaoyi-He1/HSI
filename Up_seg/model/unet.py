from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self,  in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = self.up(x1)
        return self.relu(self.bn(self.conv(x1)))


class UpConvUnet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, 
                 in_size: tuple[int, int], 
                 out_size: tuple[int, int],):
        '''
        Args:
            in_ch (int): Input channels.
            out_ch (int): Output channels.
            in_size (int): Input size.
            out_size (int): Output size.
        Use this class to upsample the input feature map to the output size.
        For increasing the resolution of the input feature map.
        Final output will behave as the segmentation map.
        Each upconv layer will increase the resolution by 2x and 
            followed by 3 ConvBNReLU layers with same in and out channels & H & W.
        '''
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.in_size = in_size
        self.out_size = out_size
        
        self.num_upconv_h = int(math.log2(out_size[0] / in_size[0]))
        self.num_upconv_w = int(math.log2(out_size[1] / in_size[1]))
        assert self.num_upconv_h == self.num_upconv_w, \
            f"Input size {in_size} and output size {out_size} must have same ratio for height and width."
        self.num_upconv = self.num_upconv_h
        assert self.num_upconv > 0, "Input size must be smaller than output size."
        assert in_size[0] * (2 ** self.num_upconv) == out_size[0] and \
               in_size[1] * (2 ** self.num_upconv) == out_size[1], \
            f"Input size {in_size} and output size {out_size} must have same ratio for height and width."
        
        self.upsample_list = nn.ModuleList()
        mid_ch = in_ch
        for i in range(self.num_upconv):
            self.upsample_list.append(UpConvBNReLU(mid_ch, 2 * mid_ch, flag=True))
            mid_ch *= 2
            for _ in range(3):
                self.upsample_list.append(ConvBNReLU(mid_ch, mid_ch))
        self.upsample_list.append(ConvBNReLU(mid_ch, out_ch, kernel_size=1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.upsample_list:
            x = layer(x)
        return x
        