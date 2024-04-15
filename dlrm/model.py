from dataclasses import dataclass


import json
import click
import torch
from loguru import logger
from torch import nn
from typing import Mapping, Tuple, List, Dict

from torch.utils.data import DataLoader

from dlrm.criteo_dataset import CriteoParquetDataset


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DenseArch(nn.Module):
    def __init__(self, dense_feature_count: int, output_size: int) -> None:
        super(DenseArch, self).__init__()  # Call the superclass's __init__ method
        self.mlp = MLP(input_size=dense_feature_count, hidden_size=output_size * 2, output_size=output_size)  # D X O

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input : B X D # Output : B X O
        return self.mlp(inputs)


class SparseArch(nn.Module):
    def __init__(self, embedding_dimensions: Mapping[str, Tuple[int, int]], output_size: int) -> None:
        super(SparseArch, self).__init__()

        # Create Embedding layers for each sparse feature
        self.embeddings = nn.ModuleDict({
            feature_name: nn.Embedding(num_embeddings, embedding_dim)
            for feature_name, (num_embeddings, embedding_dim) in embedding_dimensions.items()
        })
        self.num_sparse_features = len(embedding_dimensions)

        # Create MLP for each sparse feature
        self.mlps = nn.ModuleDict({
            feature_name: MLP(input_size=embedding_dim, hidden_size=output_size * 2, output_size=output_size)
            for feature_name, (num_embeddings, embedding_dim) in embedding_dimensions.items()
        })

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        output_values = []
        for feature, input_values in inputs.items():
            embeddings = self.embeddings[feature](input_values)
            sparse_out = self.mlps[feature](embeddings)
            output_values.append(sparse_out)

        return output_values


class DenseSparseInteractionLayer(nn.Module):
    def forward(self, dense_out: torch.Tensor, sparse_out: List[torch.Tensor]) -> float:
        concat = torch.cat([dense_out] + sparse_out, dim=-1).unsqueeze(2)
        out = torch.bmm(concat, torch.transpose(concat, 1, 2))
        flattened = torch.flatten(out, 1)
        return flattened


class PredictionLayer(nn.Module):
    def __init__(self, dense_out_size: int, sparse_out_sizes: List[int], hidden_size: int):
        super(PredictionLayer, self).__init__()
        concat_size = sum(sparse_out_sizes) + dense_out_size
        self.mlp = MLP(input_size=concat_size * concat_size, hidden_size=hidden_size, output_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> float:
        mlp_out = self.mlp(inputs)
        result = self.sigmoid(mlp_out)
        return result


@dataclass
class Parameters:
    dense_input_feature_size: int
    sparse_embedding_dimenstions: Mapping[str, Tuple[int, int]]
    dense_output_size: int
    sparse_output_size: int
    dense_hidden_size: int
    sparse_hidden_size: int
    prediction_hidden_size: int


class DLRM(nn.Module):
    def __init__(self, parameters: Parameters):
        super(DLRM, self).__init__()
        self.dense_layer = DenseArch(dense_feature_count=parameters.dense_input_feature_size,
                                     output_size=parameters.dense_output_size)
        self.sparse_layer = SparseArch(embedding_dimensions=parameters.sparse_embedding_dimenstions,
                                       output_size=parameters.sparse_output_size)
        self.interaction_layer = DenseSparseInteractionLayer()
        self.prediction_layer = PredictionLayer(
            dense_out_size=parameters.dense_output_size,
            sparse_out_sizes=[parameters.sparse_output_size] * len(parameters.sparse_embedding_dimenstions),
            hidden_size=parameters.prediction_hidden_size
        )

    def forward(self, dense_features: torch.Tensor, sparse_features: torch.Tensor) -> float:
        dense_out = self.dense_layer(dense_features)
        sparse_out = self.sparse_layer(sparse_features)
        ds_out = self.interaction_layer(dense_out, sparse_out)
        return self.prediction_layer(ds_out)




def read_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


@click.command()
@click.option('--file_path', type=click.Path(exists=True), help='Path to the parquet file')
@click.option('--metadata_path', type=click.Path(exists=True), help='Path to the metadata file')
def dry_run_with_data(file_path, metadata_path):
    """
    Process the file specified by --file_path and use metadata from --metadata_path.
    """
    logger.info("Reading the parquet file {}...".format(file_path))
    logger.info("Reading the metadata file {}...".format(metadata_path))

    dataset = CriteoParquetDataset(file_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    labels, dense, sparse = next(iter(data_loader))
    logger.info("Labels size: {}".format(labels.size()))
    logger.info("Dense size: {}".format(dense.size()))
    logger.info("Sparse size: {}".format(sparse.size()))

    metadata = read_metadata(metadata_path)

    dense_mlp_out_size = 16
    num_dense_features = dense.size()[1]
    dense_arch = DenseArch(num_dense_features, dense_mlp_out_size)
    dense_out = dense_arch(dense)
    logger.info("Dense out size: {}".format(dense_out.size()))

    # TODO fix the sparse loader to use the metadata
    # TODO dry run the sparse arch
    # TODO dry run the interaction layer
    # TODO dry run the prediction layer
    # TODO dry run the DLRM model


if __name__ == "__main__":
    dry_run_with_data()
    # parameters = Parameters(
    #     dense_input_feature_size=13,
    #     sparse_embedding_dimenstions={
    #         "SPARSE_{}".format(i): (100, 10) for i in range(26)
    #     },
    #     dense_output_size=4,
    #     sparse_output_size=4,
    #     dense_hidden_size=4,
    #     sparse_hidden_size=4,
    #     prediction_hidden_size=4
    # )
    #
    # model = DLRM(parameters)
    # dense_features = torch.rand(32, 13)
    # sparse_features = {f"SPARSE_{i}": torch.randint(0, 100, (32,)) for i in range(26)}
    # model(dense_features, sparse_features)
    # print(model)
    # print("Model created successfully")
    # print("Model output")
    # print(model(dense_features, sparse_features))
    # print("Model output shape")
    # print(model(dense_features, sparse_features).shape)
