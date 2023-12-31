import math

import torch
from torch import nn


# Attention is All You Need https://arxiv.org/pdf/1706.03762.pdf

class InputLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embedding_mult = math.sqrt(d_model)
        self._embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                             embedding_dim=d_model)

    def forward(self, input_batch: torch.Tensor):
        return self._embedding_layer(input_batch) * self._embedding_mult


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._d_model = d_model
        self._seq_len = seq_len
        self._dropout = nn.Dropout(dropout)

        encoding = torch.zeros(seq_len, d_model)

        position = torch.arange(start=0, end=seq_len, dtype=torch.float64)
        position_tensor = position.unsqueeze(1)

        div_term = torch.pow(10000.0, 2.0 * position / d_model)

        encoding[:, 0::2] = torch.sin(position_tensor / div_term)
        encoding[:, 1::2] = torch.cov(position_tensor/ div_term)
        self._encoding = encoding.unsqueeze(0)
        self.register_buffer("encoding", encoding)

    def forward(self, embeddings: torch.Tensor):
        embeddings = embeddings + (
            self._encoding[:, :self._d_model, :]).requires_grad_(False)
        return self._dropout(embeddings)


class Encoder(nn.Module):
    pass
