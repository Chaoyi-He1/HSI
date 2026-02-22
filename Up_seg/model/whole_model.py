from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import ViT
from .unet import UpConvUnet


class whole_model(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, 
                 in_hw: tuple[int, int], patch_size: int, out_channels: int,
                 num_heads: int, l_query: int, num_layers: int,
                 upconv_in_ch: int, upconv_out_ch: int,
                 upconv_in_size: tuple[int, int], upconv_out_size: tuple[int, int],
                 scale_factor: int = 2):
        super().__init__()
        self.vit = ViT(in_channels=in_channels, hidden_size=hidden_size,
                       in_hw=in_hw, patch_size=patch_size,
                       out_channels=out_channels,
                       num_heads=num_heads, l_query=l_query,
                       num_layers=num_layers, scale_factor=scale_factor)
        self.upconv = UpConvUnet(upconv_in_ch, upconv_out_ch,
                                 upconv_in_size, upconv_out_size)
        self.channel_select = None
        if in_channels != 10 and in_channels != 3:
            self.channel_select = nn.Parameter(torch.randn(1, in_channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W)
        """
        if self.channel_select is not None:
            x = x * self.channel_select
        x = self.vit(x)
        x = self.upconv(x)
        
        return x


def get_model(args):
    """
    Get the model.
    Args:
        args: Arguments.
    Returns:
        model: Model.
    """
    model = whole_model(in_channels=args.in_channels,
                        hidden_size=args.hidden_size,
                        in_hw=(args.in_h, args.in_w),
                        patch_size=args.patch_size,
                        out_channels=args.upconv_in_ch,
                        num_heads=8,
                        l_query=(args.in_h // args.patch_size) * args.transformer_scale * (args.in_w // args.patch_size) * args.transformer_scale,
                        num_layers=args.num_layers,
                        upconv_in_ch=args.upconv_in_ch,
                        upconv_out_ch=args.num_classes,
                        upconv_in_size=(args.in_h * args.transformer_scale, args.in_w * args.transformer_scale),
                        upconv_out_size=(args.img_h, args.img_w),
                        scale_factor=args.transformer_scale)
    return model