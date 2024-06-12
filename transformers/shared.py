# PositionEncoding
# LayerNorm
# TransformerEmbedding
# Scaled Dot-Product Attention
# MultiHeadAttention
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, sequence_length:int, d_k: int, masked: bool) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        if masked:
            mask = torch.full((sequence_length, sequence_length),
                              -torch.inf)
            self.mask = torch.triu(mask, diagonal=1)
        else:
            self.mask = torch.zeros((sequence_length, sequence_length))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores += self.mask
        attention = F.softmax(scores, dim=-1)
        return torch.matmul(attention, v), attention
