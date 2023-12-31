import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, with_masks: bool, sequence_length: int, d_model: int,
                 heads_count: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sequence_length = sequence_length
        self._embed_size = d_model
        self._with_masks = with_masks

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        pass
