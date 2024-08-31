from dataclasses import dataclass
from transformer import TransformerConfig, Transformer
import torch
import torch.nn as nn
from typing import List


@dataclass
class MovieLensTransformerConfig:
    movie_transformer_config: TransformerConfig
    user_embedding_dimension: int
    num_users: int
    interaction_mlp_hidden_sizes: List[int]


class InteractionMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        hidden_sizes: List[int],
        output_size: int,
    ):
        super(InteractionMLP, self).__init__()
        actual_input_size = input_size * sequence_length
        fc_layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                fc_layers.append(nn.Linear(actual_input_size, hidden_size))
            else:
                fc_layers.append(nn.Linear(hidden_sizes[i - 1], hidden_size))
            fc_layers.append(nn.ReLU())
        fc_layers.append(
            nn.Linear(
                hidden_sizes[-1] if hidden_sizes else actual_input_size,
                output_size,
                bias=False,
            )
        )
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class MovieLensTransformer(nn.Module):
    def __init__(self, config: MovieLensTransformerConfig):
        super().__init__()
        self.movie_transformer = Transformer(config.movie_transformer_config)
        self.user_embedding = nn.Embedding(
            config.num_users, config.user_embedding_dimension
        )
        self.output_layer = InteractionMLP(
            (
                config.movie_transformer_config.embedding_dimension
                + config.user_embedding_dimension
            ),
            config.movie_transformer_config.context_window_size,
            config.interaction_mlp_hidden_sizes,
            config.movie_transformer_config.vocab_size,
        )

    def forward(self, movie_ids: torch.Tensor, user_ids: torch.Tensor):
        movie_embeddings = self.movie_transformer(movie_ids)
        user_embeddings = self.user_embedding(user_ids)
        user_embeddings = user_embeddings.unsqueeze(1).expand(
            -1, movie_embeddings.shape[1], -1
        )
        embeddings = torch.cat([movie_embeddings, user_embeddings], dim=-1)
        return self.output_layer(embeddings)  # returns logits
