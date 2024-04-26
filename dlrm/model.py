import time
from dataclasses import dataclass

import json
import click
import torch
from loguru import logger
from torch import nn, Tensor
from typing import Mapping, Tuple, List, Dict, Union, Optional

from torch.utils.data import DataLoader

from criteo_dataset import CriteoParquetDataset
from torch._dynamo.utils import CompileProfiler

# Reset since we are using a different mode.
import torch._dynamo

torch._dynamo.reset()


# MDOEL ARCHITECTURE
# output:
#                     probability of a click
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]

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
        super(DenseArch, self).__init__()
        self.mlp = MLP(input_size=dense_feature_count, hidden_size=output_size * 2, output_size=output_size)  # D X O

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input : B X D # Output : B X O
        return self.mlp(inputs)


class SparseFeatureLayer(nn.Module):
    def __init__(self, cardinality: int, embedding_size: int, output_size: int):
        super(SparseFeatureLayer, self).__init__()
        self.embedding = nn.Embedding(cardinality, embedding_size)
        self.mlp = MLP(input_size=embedding_size, hidden_size=output_size * 2, output_size=output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input : B X 1 # Output : B X O
        embeddings = self.embedding(inputs)
        return self.mlp(embeddings)


class SparseArch(nn.Module):
    def __init__(self, metadata: Dict[str, Union[int, List[int]]],
                 embedding_sizes: Mapping[str, int],
                 output_size: int,
                 modulus_hash_size: Optional[int] = None) -> None:
        super(SparseArch, self).__init__()
        self.use_modulus_hash = modulus_hash_size is not None
        self.num_sparse_features = len(metadata)
        # Create Embedding layers for each sparse feature
        self._modulus_hash_sizes = [min(m["cardinality"], modulus_hash_size or m["cardinality"]) for m in
                                    metadata.values()]

        self.sparse_layers = nn.ModuleList([
            SparseFeatureLayer(cardinality=self._modulus_hash_sizes[i],
                               embedding_size=embedding_sizes[feature_name],
                               output_size=output_size) for i, feature_name in enumerate(metadata.keys())
        ])

        # Create mapping for each sparse feature
        self.mapping = [metadata[f"SPARSE_{i}"]["tokenizer_values"] for i in range(self.num_sparse_features)]

    @staticmethod
    def index_hash(tensor: torch.Tensor, tokenizer_values: List[int]):
        tensor = tensor.reshape(-1, 1)
        tokenizers = torch.tensor(tokenizer_values).reshape(1, -1)
        if tensor.is_cuda:
            tokenizers = tokenizers.cuda()
        matches = tensor == tokenizers
        indices = torch.argmax(matches.to(torch.int64), dim=1)
        return indices

    @staticmethod
    def modulus_hash(tensor: torch.Tensor, cardinality: int):
        return (tensor + 1) % cardinality

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        output_values = []
        for i in range(self.num_sparse_features):
            if self.use_modulus_hash:
                indices = self.modulus_hash(inputs[:, i], self._modulus_hash_sizes[i])
            else:
                indices = self.index_hash(inputs[:, i], self.mapping[i])
            sparse_out = self.sparse_layers[i](indices)
            output_values.append(sparse_out)
        return output_values


class DenseSparseInteractionLayer(nn.Module):
    def forward(self, dense_out: torch.Tensor, sparse_out: List[torch.Tensor]) -> Tensor:
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mlp_out = self.mlp(inputs)
        result = self.sigmoid(mlp_out)
        return result


@dataclass
class Parameters:
    dense_input_feature_size: int
    sparse_embedding_sizes: Mapping[str, int]
    dense_output_size: int
    sparse_output_size: int
    dense_hidden_size: int
    sparse_hidden_size: int
    prediction_hidden_size: int
    modulus_hash_size: Optional[int] = None


class DLRM(nn.Module):
    def __init__(self, metadata: Dict[str, Union[int, List[int]]], parameters: Parameters):
        super(DLRM, self).__init__()
        self.dense_layer = DenseArch(dense_feature_count=parameters.dense_input_feature_size,
                                     output_size=parameters.dense_output_size)
        self.sparse_layer = SparseArch(metadata=metadata,
                                       embedding_sizes=parameters.sparse_embedding_sizes,
                                       output_size=parameters.sparse_output_size,
                                       modulus_hash_size=parameters.modulus_hash_size)
        self.interaction_layer = DenseSparseInteractionLayer()
        self.prediction_layer = PredictionLayer(
            dense_out_size=parameters.dense_output_size,
            sparse_out_sizes=[parameters.sparse_output_size] * len(parameters.sparse_embedding_sizes),
            hidden_size=parameters.prediction_hidden_size
        )

    def forward(self, dense_features: torch.Tensor, sparse_features: torch.Tensor) -> float:
        dense_out = self.dense_layer(dense_features)
        sparse_out = self.sparse_layer(sparse_features)
        ds_out = self.interaction_layer(dense_out, sparse_out)
        return self.prediction_layer(ds_out).squeeze()


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

    dense_mlp_out_size = 16
    num_dense_features = dense.size()[1]
    dense_arch = DenseArch(num_dense_features, dense_mlp_out_size)
    # dense_arch_optim = torch.compile(dense_arch)
    dense_out = dense_arch(dense)
    logger.info("Dense out size: {}".format(dense_out.size()))

    metadata = read_metadata(metadata_path)
    embedding_size = 16
    embedding_sizes = {fn: embedding_size for fn in metadata.keys()}
    sparse_mlp_out_size = 16
    sparse_arch = SparseArch(metadata, embedding_sizes, output_size=sparse_mlp_out_size)
    # compiled model hangs on running with inputs
    # sparse_arch_optim = torch.compile(sparse_arch)
    sparse_out = sparse_arch(sparse)
    for v in sparse_out:
        logger.info("Sparse out size: {}".format(v.size()))

    dense_sparse_interaction_layer = DenseSparseInteractionLayer()
    # dense_sparse_interaction_layer_optim = torch.compile(dense_sparse_interaction_layer)
    ds_out = dense_sparse_interaction_layer(dense_out, sparse_out)
    logger.info("Dense sparse interaction out size: {}".format(ds_out.size()))

    prediction_layer = PredictionLayer(dense_out_size=dense_mlp_out_size,
                                       sparse_out_sizes=[sparse_mlp_out_size] * len(metadata),
                                       hidden_size=16)
    # prediction_layer_optim = torch.compile(prediction_layer)
    pred_out = prediction_layer(ds_out)
    logger.info("Prediction out size: {}".format(pred_out.size()))
    logger.info("Prediction out value: {}".format(pred_out))

    # TODO dry run the DLRM model
    parameters = Parameters(
        dense_input_feature_size=13,
        sparse_embedding_sizes={
            "SPARSE_{}".format(i): 16 for i in range(26)
        },
        dense_output_size=128,
        sparse_output_size=128,
        dense_hidden_size=64,
        sparse_hidden_size=64,
        prediction_hidden_size=32,
        modulus_hash_size=10000
    )
    import torch._dynamo
    torch._dynamo.reset()
    torch._dynamo.config.verbose = True
    dlrm = DLRM(metadata, parameters)
    # dlrm_optim = torch.compile(dlrm, backend="aot_eager")
    dlrm_optim = torch.compile(dlrm, backend="inductor")
    logger.info("Compiled DLRM model: {}".format(dlrm))
    start = time.time()
    prediction = dlrm_optim(dense, sparse)
    logger.info("DLRM prediction size: {}".format(prediction.size()))
    logger.info("DLRM prediction value: {}".format(prediction))
    logger.info("[EAGER] Time taken for prediction: {}".format(time.time() - start))

    # dlrm_compiled = torch.compile(dlrm)
    # logger.info("Compiled DLRM model: {}".format(dlrm_compiled))
    # start = time.time()
    # prediction = dlrm_compiled(dense.cuda(), sparse.cuda())
    # logger.info("DLRM prediction size: {}".format(prediction.size()))
    # logger.info("DLRM prediction value: {}".format(prediction))
    # logger.info("[COMPILED] Time taken for prediction: {}".format(time.time() - start))


if __name__ == "__main__":
    dry_run_with_data()
