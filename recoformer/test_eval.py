from eval import get_model_predictions
import torch
import torch.nn as nn
import pytest

torch.random.manual_seed(42)


class ModelToTest(nn.Module):
    def __init__(self, token_count: int) -> None:
        super(ModelToTest, self).__init__()
        self._token_count = token_count

    def forward(self, movies: torch.Tensor, users: torch.Tensor):
        return torch.rand(movies.shape[0], self._token_count)


def test_get_model_predictions():
    # Create dummy inputs
    model = ModelToTest(20)
    movie_id_tokens = torch.tensor([[1, 12, 3, 8, 6], [14, 5, 11, 19, 9]])
    user_id_tokens = torch.tensor([1, 2])
    n = 5

    # Call the function
    predictions = get_model_predictions(model, movie_id_tokens, user_id_tokens, n)
    # Check the output shape
    assert predictions.shape == (2, n)
    # Check the output values

    expected = torch.tensor([[10, 16, 0, 7, 14], [10, 17, 12, 3, 15]])
    assert torch.equal(predictions, expected)
