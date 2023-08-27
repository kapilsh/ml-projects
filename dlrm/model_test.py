import pytest
import torch

from dlrm.model import DenseLayer, SparseLayer

torch.manual_seed(0)

def test_dense_layer():
    dense_dim = 100
    x = torch.tensor([[x for x in range(dense_dim)]], dtype=torch.float32)
    dense_layer = DenseLayer(input_dim=dense_dim, output_dim=16)
    result = dense_layer(x)
    expected_result = torch.Tensor(
        [[0.0000, 0.0000, 0.0000, 61.1469, 0.0000, 14.6846, 26.4580, 0.0000,
          33.7176, 0.0000, 4.4533, 0.0000, 23.0941, 63.5569, 10.7485, 0.0000]])
    assert torch.allclose(expected_result, result)


def test_sparse_layer():
    sparse_dim = 2
    x = torch.LongTensor([[0, 1, 0]])
    sparse_layer = SparseLayer(sparse_dim, 8, 16, 3)
    result = sparse_layer(x)
    expected_result = torch.FloatTensor([[[0.0000, 0.0000, 0.0000, 1.0355, 0.4590, 0.0298, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.6578, 1.5072, 0.5505, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 1.0355, 0.4590, 0.0298, 0.0000, 0.0000]]])
    assert torch.allclose(expected_result, result, atol=.0001, rtol=.0001)
