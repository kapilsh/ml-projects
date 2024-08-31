import pytest
import torch
from movielens_transformer import (
    MovieLensTransformer,
    MovieLensTransformerConfig,
    TransformerConfig,
)


torch.manual_seed(42)


@pytest.fixture
def sample_config():
    transformer_config = TransformerConfig(
        vocab_size=10,
        context_window_size=4,
        embedding_dimension=8,
        num_layers=2,
        num_heads=2,
        dropout_embeddings=0.1,
        dropout_attention=0.1,
        dropout_residual=0.1,
        layer_norm_epsilon=1e-5,
    )
    return MovieLensTransformerConfig(
        movie_transformer_config=transformer_config,
        user_embedding_dimension=4,
        num_users=5,
        interaction_mlp_hidden_sizes=[],
    )


@pytest.fixture
def model(sample_config):
    return MovieLensTransformer(sample_config)


def test_forward_pass(model):
    movie_ids = torch.randint(0, 10, (2, 4))
    user_ids = torch.randint(0, 4, (2,))
    output = model(movie_ids, user_ids)
    assert output.shape == (2, 10)  # batch_size, sequence_length, vocab_size


if __name__ == "__main__":
    pytest.main()
