import math
from dataclasses import dataclass
from transformers import GPT2LMHeadModel

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    context_window_size: int = 1024
    embedding_dimension: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout_embeddings: float = 0.1
    dropout_attention: float = 0.1
    dropout_residual: float = 0.1
    layer_norm_epsilon: float = 1e-5


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config, device: str = 'cpu'):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size,
                                            config.embedding_dimension)
        self.positional_embedding = nn.Embedding(config.context_window_size,
                                                 config.embedding_dimension)
        self.positional_ids = torch.arange(
            config.context_window_size).unsqueeze(0).to(device)
        self.embedding_dropout = nn.Dropout(config.dropout_embeddings)
        self.layers = nn.ModuleList(
            [GPT2Layer(config) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)
        # gpt2 does not use a bias in the output layer
        self.output_layer = nn.Linear(config.embedding_dimension,
                                      config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        _, current_sequence_length = input_ids.size()
        positions = self.positional_ids[:, :current_sequence_length]
        embeddings = self.token_embedding(
            input_ids) + self.positional_embedding(positions)
        embeddings = self.embedding_dropout(embeddings)
        for layer in self.layers:
            embeddings = layer(embeddings)
        embeddings = self.layer_norm(embeddings)
        return self.output_layer(embeddings)  # returns logits

    @classmethod
    def from_pretrained(cls) -> 'GPT2':
        model_hf = GPT2LMHeadModel.from_pretrained(
            'gpt2', resume_download=None)
        config = GPT2Config()
        model = cls(config)
        with torch.no_grad():
            model.token_embedding.weight.copy_(model_hf.transformer.wte.weight)
            model.positional_embedding.weight.copy_(
                model_hf.transformer.wpe.weight)
            for layer_idx in range(len(model.layers)):
                model.layers[layer_idx].layer_norm1.weight.copy_(
                    model_hf.transformer.h[layer_idx].ln_1.weight)
                model.layers[layer_idx].layer_norm1.bias.copy_(
                    model_hf.transformer.h[layer_idx].ln_1.bias)
                # HF model uses Conv1d for qkv projection, we use linear.
                # hence the transpose
                model.layers[layer_idx].attention.qkv.weight.copy_(
                    model_hf.transformer.h[layer_idx].attn.c_attn.weight.t())
                model.layers[layer_idx].attention.qkv.bias.copy_(
                    model_hf.transformer.h[layer_idx].attn.c_attn.bias)
                model.layers[layer_idx].attention.out.weight.copy_(
                    model_hf.transformer.h[layer_idx].attn.c_proj.weight.t())
                model.layers[layer_idx].attention.out.bias.copy_(
                    model_hf.transformer.h[layer_idx].attn.c_proj.bias)
                model.layers[layer_idx].layer_norm2.weight.copy_(
                    model_hf.transformer.h[layer_idx].ln_2.weight)
                model.layers[layer_idx].layer_norm2.bias.copy_(
                    model_hf.transformer.h[layer_idx].ln_2.bias)
                model.layers[layer_idx].mlp.fc1.weight.copy_(
                    model_hf.transformer.h[layer_idx].mlp.c_fc.weight.t())
                model.layers[layer_idx].mlp.fc1.bias.copy_(
                    model_hf.transformer.h[layer_idx].mlp.c_fc.bias)
                model.layers[layer_idx].mlp.fc2.weight.copy_(
                    model_hf.transformer.h[layer_idx].mlp.c_proj.weight.t())
                model.layers[layer_idx].mlp.fc2.bias.copy_(
                    model_hf.transformer.h[layer_idx].mlp.c_proj.bias)
            model.layer_norm.weight.copy_(model_hf.transformer.ln_f.weight)
            model.layer_norm.bias.copy_(model_hf.transformer.ln_f.bias)
            model.output_layer.weight.copy_(model_hf.lm_head.weight)
            # output_layer does not have a bias in gpt2
            return model


class GPT2Layer(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.embedding_dimension)
        self.attention = CausalMultiHeadAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dimension)
        self.residual_dropout = nn.Dropout(config.dropout_residual)
        self.mlp = MLP(config)
        self.dropout_mlp = nn.Dropout(config.dropout_residual)

    def forward(self, embeddings: torch.Tensor):
        attention_output = embeddings + self.attention(
            self.layer_norm1(embeddings))
        attention_output = self.residual_dropout(attention_output)
        mlp_output = attention_output + self.mlp(
            self.layer_norm2(attention_output))
        return mlp_output


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.fc1 = nn.Linear(config.embedding_dimension,
                             4 * config.embedding_dimension)
        self.activation = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(4 * config.embedding_dimension,
                             config.embedding_dimension)
        self.dropout = nn.Dropout(config.dropout_residual)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.embedding_dimension = config.embedding_dimension
        self.head_dim = self.embedding_dimension // self.num_heads
        assert self.head_dim * self.num_heads == self.embedding_dimension, \
            "embedding_dimension must be divisible by num_heads"
        self.qkv = nn.Linear(self.embedding_dimension,
                             3 * self.embedding_dimension)
        self.out = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.prob_dropout = config.dropout_attention

    def forward(self, x: torch.Tensor):
        batch_size, sequence_length, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.embedding_dimension,
                            dim=2)  # split along the third dimension

        k = k.view(batch_size, sequence_length, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, sequence_length, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        q = q.view(batch_size, sequence_length, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        weights = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.prob_dropout if self.training else 0)

        output = weights.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.embedding_dimension)
        output = self.out(output)
        return output
