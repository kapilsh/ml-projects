from torch import nn

class DenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x

class SparseLayer(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, max_nnz):
        super(SparseLayer, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.act = nn.ReLU()
        self.max_nnz = max_nnz

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        x = self.act(x)
        return x