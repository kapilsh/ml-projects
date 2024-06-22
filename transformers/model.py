import math

import torch
from torch import nn
from torch.nn import functional as F


class InputLayer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 sequence_length: int,
                 embedding_dimension: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_dimension = embedding_dimension
        self.input_embeddings = nn.Embedding(num_embeddings=num_embeddings,
                                             embedding_dim=embedding_dimension)
        # we can also change this to a nn.Parameter
        self.position_embeddings = self._init_position_embedding(
            sequence_length, embedding_dimension)

    @staticmethod
    def _init_position_embedding(sequence_length: int,
                                 embedding_dimension: int) -> torch.Tensor:
        even_index = torch.arange(0, embedding_dimension, 2)
        # since we end up doing odd_index - 1, we end up not needing it, and
        # we can just use even_index
        # odd_index = torch.arange(1, embedding_dimension, 2)
        denominator = torch.pow(10000, even_index / embedding_dimension)
        positions = torch.arange(
            0, sequence_length, 1).reshape(sequence_length, 1)
        even_pe = torch.sin(positions / denominator)
        odd_pe = torch.cos(positions / denominator)
        stacked = torch.stack([even_pe, odd_pe], dim=2)
        return torch.flatten(stacked, start_dim=1, end_dim=2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        emb = self.input_embeddings(input)
        return emb + self.position_embeddings


class ScaledDotProductAttention(nn.Module):
    def __init__(self, sequence_length: int, d_k: int, masked: bool) -> None:
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


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimension: int, sequence_length: int,
                 num_heads: int, masked: bool = False) -> None:
        super(MultiHeadAttention, self).__init__()
        self.head_dimension = embedding_dimension // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(embedding_dimension, embedding_dimension)
        self.k_linear = nn.Linear(embedding_dimension, embedding_dimension)
        self.v_linear = nn.Linear(embedding_dimension, embedding_dimension)
        self.out = nn.Linear(embedding_dimension, embedding_dimension)
        self.attention_layer = ScaledDotProductAttention(
            sequence_length=sequence_length,
            d_k=self.head_dimension,
            masked=masked
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q, k, v have shape (batch_size, sequence_length, embedding_dimension)
        # next we apply the linear layer and then split the output into
        # num_heads
        batch_size = q.size(0)
        # second dimension is the sequence length
        output_shape = (batch_size, -1, self.num_heads, self.head_dimension)
        # we transpose the output to
        # (batch_size, num_heads, sequence_length, head_dimension)
        # this allows us to use num_heads as a batch dimension
        q = self.q_linear(q).view(*output_shape).transpose(1, 2)
        k = self.k_linear(k).view(*output_shape).transpose(1, 2)
        v = self.v_linear(v).view(*output_shape).transpose(1, 2)
        # we apply the scaled dot product attention
        x, attention = self.attention_layer(q, k, v)
        # we transpose the output back to
        # (batch_size, sequence_length, embedding_dimension)
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.head_dimension * self.num_heads)
        return self.out(x), attention


class MaskedMultiHeadAttention(MultiHeadAttention):
    def __init__(self, embedding_dimension: int, sequence_length: int,
                 num_heads: int) -> None:
        super(MaskedMultiHeadAttention, self).__init__(
            embedding_dimension=embedding_dimension,
            sequence_length=sequence_length,
            num_heads=num_heads,
            masked=True
        )


class FeedForwardNetwork(nn.Module):
    def __init__(self,
                 embedding_dimension: int,
                 hidden_dimension: int,
                 drop_prop: float):
        super().__init__()
        self.linear1 = nn.Linear(in_features=embedding_dimension,
                                 out_features=hidden_dimension)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prop)
        self.linear2 = nn.Linear(in_features=hidden_dimension,
                                 out_features=embedding_dimension)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self,
                 embedding_dimension: int,
                 sequence_length: int,
                 num_heads: int,
                 hidden_dimension: int,
                 drop_prop: float):
        super().__init__()
        self.attention = MultiHeadAttention(
            embedding_dimension=embedding_dimension,
            sequence_length=sequence_length,
            num_heads=num_heads
        )
        self.feed_forward = FeedForwardNetwork(
            embedding_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            drop_prop=drop_prop
        )
        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.norm2 = nn.LayerNorm(embedding_dimension)
        self.drop1 = nn.Dropout(drop_prop)
        self.drop2 = nn.Dropout(drop_prop)

    def forward(self, x: torch.Tensor):
        x = x + self.drop1(self.attention(x))
        x = self.norm1(x)
        x = x + self.drop2(self.feed_forward(x))
        x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 embedding_dimension: int,
                 sequence_length: int,
                 num_heads: int,
                 hidden_dimension: int,
                 drop_prop: float):
        super().__init__()
        self.input_layer = InputLayer(num_embeddings=sequence_length,
                                      sequence_length=sequence_length,
                                      embedding_dimension=embedding_dimension)
        self.layers = nn.Sequential(*[
            EncoderLayer(embedding_dimension=embedding_dimension,
                         sequence_length=sequence_length,
                         num_heads=num_heads,
                         hidden_dimension=hidden_dimension,
                         drop_prop=drop_prop)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        return self.layers(x)


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dimension: int, sequence_length: int,
                 num_heads: int, hidden_dimension: int, drop_prop: float):
        super().__init__()
        self.masked_attention = MaskedMultiHeadAttention(
            embedding_dimension=embedding_dimension,
            sequence_length=sequence_length,
            num_heads=num_heads
        )
        self.attention = MultiHeadAttention(
            embedding_dimension=embedding_dimension,
            sequence_length=sequence_length,
            num_heads=num_heads
        )
        self.feed_forward = FeedForwardNetwork(
            embedding_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            drop_prop=drop_prop
        )
        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.norm2 = nn.LayerNorm(embedding_dimension)
        self.norm3 = nn.LayerNorm(embedding_dimension)
        self.drop1 = nn.Dropout(drop_prop)
        self.drop2 = nn.Dropout(drop_prop)
        self.drop3 = nn.Dropout(drop_prop)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor):
        x = x + self.drop1(self.masked_attention(x, x, x))
        x = self.norm1(x)
        x = x + self.drop2(self.attention(x, enc_output, enc_output))
        x = self.norm2(x)
        x = x + self.drop3(self.feed_forward(x))
        x = self.norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 embedding_dimension: int,
                 sequence_length: int,
                 num_heads: int,
                 hidden_dimension: int,
                 drop_prop: float):
        super().__init__()
        self.input_layer = InputLayer(num_embeddings=sequence_length,
                                      sequence_length=sequence_length,
                                      embedding_dimension=embedding_dimension)
        self.layers = nn.Sequential(*[
            DecoderLayer(embedding_dimension=embedding_dimension,
                         sequence_length=sequence_length,
                         num_heads=num_heads,
                         hidden_dimension=hidden_dimension,
                         drop_prop=drop_prop)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor):
        x = self.input_layer(x)
        return self.layers(x, enc_output)


class Transformer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 embedding_dimension: int,
                 sequence_length: int,
                 num_heads: int,
                 hidden_dimension: int,
                 drop_prop: float):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers,
                               embedding_dimension=embedding_dimension,
                               sequence_length=sequence_length,
                               num_heads=num_heads,
                               hidden_dimension=hidden_dimension,
                               drop_prop=drop_prop)
        self.decoder = Decoder(num_layers=num_layers,
                               embedding_dimension=embedding_dimension,
                               sequence_length=sequence_length,
                               num_heads=num_heads,
                               hidden_dimension=hidden_dimension,
                               drop_prop=drop_prop)
        self.output_layer = nn.Linear(in_features=embedding_dimension,
                                      out_features=sequence_length)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        enc_output = self.encoder(x)
        dec_output = self.decoder(y, enc_output)
        return self.output_layer(dec_output)