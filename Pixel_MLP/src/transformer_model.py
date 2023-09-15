import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerEncoder, TransformerDecoder
from torch import Tensor
from positional_embedding import *


class Transformer_Encoder_cls(nn.Module):
    def __init__(self, num_layers=8, norm=None, d_model=512, 
                 nhead=8, dim_feedforward=2048, dropout=0.1,
                 drop_path=0.4, activation="relu", 
                 normalize_before=True, num_cls=18) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(num_layers, norm, d_model, 
                                          nhead, dim_feedforward, dropout, 
                                          drop_path, activation, 
                                          normalize_before)
        self.classify_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(inplace=True),
            nn.Linear(2 * d_model, num_cls)
        )
        
        self.pos_embed = PositionEmbeddingSine(d_model)
    
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        # src: (B, N, C)
        # src_mask: (B, N)
        pos = self.pos_embed(src)
        src = self.encoder(src, src_mask, pos=pos)
        src = self.classify_head(src)
        return src


def build_Transformer_Encoder(num_layers=8, norm=None, d_model=3, 
                              nhead=8, dim_feedforward=2048, dropout=0.1,
                              drop_path=0.4, activation="relu", 
                              normalize_before=True, num_cls=18):
    
    return Transformer_Encoder_cls(num_layers, norm, d_model, nhead, 
                                   dim_feedforward, dropout, drop_path, activation, 
                                   normalize_before, num_cls)
    