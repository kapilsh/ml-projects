import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int
    context_window_size: int
    embedding_dimension: int
    num_layers: int
    num_heads: int
    dropout_embeddings: float = 0.1
    dropout_attention: float = 0.1
    dropout_residual: float = 0.1
    layer_norm_epsilon: float = 1e-5


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.embedding_dimension)
        self.attention = CausalMultiHeadAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dimension)
        self.mlp = MLP(config)

    def forward(self, embeddings: torch.Tensor):
        attention_output = embeddings + self.attention(self.layer_norm1(embeddings))
        mlp_output = attention_output + self.mlp(self.layer_norm2(attention_output))
        return mlp_output


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embedding_dimension, 4 * config.embedding_dimension)
        self.activation = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(4 * config.embedding_dimension, config.embedding_dimension)
        self.dropout = nn.Dropout(config.dropout_residual)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.embedding_dimension = config.embedding_dimension
        self.head_dim = self.embedding_dimension // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embedding_dimension
        ), "embedding_dimension must be divisible by num_heads"
        self.qkv = nn.Linear(self.embedding_dimension, 3 * self.embedding_dimension)
        self.out = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.attention_dropout_p = config.dropout_attention
        self.residual_dropout = nn.Dropout(config.dropout_residual)

    def forward(self, x: torch.Tensor):
        batch_size, sequence_length, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(
            self.embedding_dimension, dim=2
        )  # split along the third dimension
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        weights = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
        )
        output = (
            weights.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.embedding_dimension)
        )
        output = self.out(output)
        output = self.residual_dropout(output)
        return output


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.embedding_dimension
        )
        self.positional_embedding = nn.Embedding(
            config.context_window_size, config.embedding_dimension
        )
        positional_ids = torch.arange(config.context_window_size).unsqueeze(0)
        self.register_buffer("positional_ids", positional_ids)
        self.embedding_dropout = nn.Dropout(config.dropout_embeddings)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)

    def forward(self, input_ids: torch.Tensor):
        _, current_sequence_length = input_ids.size()
        positions = self.positional_ids[:, :current_sequence_length]
        embeddings = self.token_embedding(input_ids) + self.positional_embedding(
            positions
        )
        embeddings = self.embedding_dropout(embeddings)
        for layer in self.layers:
            embeddings = layer(embeddings)
        embeddings = self.layer_norm(embeddings)
        return embeddings
