import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


class ViTBlock(nn.Module):
    """
    A ViT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, l_query):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.TransformerDecoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            batch_first=True, 
            activation='gelu',
            )

    def forward(self, x, tgt):
        N, L, D = x.shape
        if tgt.shape[0] != N:
            tgt = self.tgt.data.expand(N, -1, -1)
        x = self.norm1(x)
        x = self.attn(tgt, x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x

class ViT(nn.Module):
    """
    Vision Transformer (ViT) model for segmentation upsampling.
    """
    def __init__(self, in_channels, hidden_size, 
                 in_hw, patch_size, out_channels,
                 num_heads, l_query, num_layers, scale_factor=2):
        super().__init__()
        self.in_hw = in_hw
        self.scale_factor = scale_factor
        self.patch_embed = PatchEmbed(in_chans=hidden_size, 
                                      embed_dim=hidden_size,
                                      img_size=in_hw,
                                      patch_size=patch_size)
        self.num_patches = self.patch_embed.num_patches
        
        self.blocks = nn.ModuleList()
        self.in_proj = nn.Conv2d(in_channels, hidden_size, 
                                  kernel_size=1, stride=1, padding=0)
        for i in range(num_layers):
            self.blocks.append(
                ViTBlock(hidden_size=hidden_size, 
                         num_heads=num_heads, 
                         l_query=l_query)
            )
        self.final_layer = FinalLayer(hidden_size=hidden_size, 
                                      patch_size=patch_size, 
                                      out_channels=out_channels)
        self.out_channels = out_channels
        self.tgt = nn.Parameter(torch.zeros(1, l_query, hidden_size))
    
    def unpatchify(self, x):
        """
        x: (N, H * W // patch_size**2, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_embed.patch_size[0]
        h, w = self.in_hw[0] // p * self.scale_factor, self.in_hw[1] // p * self.scale_factor
        assert h * w == x.shape[1], f"Invalid number of patches: {h * w} != {x.shape[1]}"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def forward(self, x):
        """
        x: (N, C, H, W)
        """
        x = self.in_proj(x)
        x = self.patch_embed(x)
        N, L, D = x.shape
        tgt = self.tgt.data.expand(N, -1, -1)
        for block in self.blocks:
            x = block(x, tgt)
        x = self.final_layer(x)
        x = self.unpatchify(x)
        
        return x
        