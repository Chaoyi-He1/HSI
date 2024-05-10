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
            ConvBNReLU(in_ch, 25, kernel_size=3),
            ConvBNReLU(25, 300, kernel_size=3),
            # ConvBNReLU(128, 512, kernel_size=3),
            nn.Conv2d(300, num_classes, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class FCNN_4lv(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()

        self.in_ch = in_ch
        self.num_classes = num_classes

        self.model = nn.Sequential(
                ConvBNReLU(in_ch, 64, kernel_size=3),
                ConvBNReLU(64, 128, kernel_size=3),
                ConvBNReLU(128, 512, kernel_size=3),
                nn.Conv2d(512, num_classes, kernel_size=3, padding=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def FCNN_lite(in_ch: int, num_classes: int) -> FCNN:
    return FCNN(in_ch, num_classes)

def FCNN_4(in_ch: int, num_classes: int) -> FCNN_4lv:
    return FCNN_4lv(in_ch, num_classes)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits


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
