import json
import time
from dataclasses import dataclass
from typing import Mapping, List, Dict, Union

import click
import torch
import torch._dynamo
from loguru import logger
from torch import nn, Tensor
from torch.utils.data import DataLoader

from criteo_dataset import CriteoParquetDataset

torch._dynamo.reset()


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int],
                 output_size: int):
        super(MLP, self).__init__()
        fc_layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                fc_layers.append(nn.Linear(input_size, hidden_size))
            else:
                fc_layers.append(nn.Linear(hidden_sizes[i - 1], hidden_size))
            fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(hidden_sizes[-1] if hidden_sizes else input_size, output_size))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor):
        return self.fc_layers(x)


class DenseArch(nn.Module):
    def __init__(self,
                 metadata: Dict[str, Union[int, List[int], Dict[str, int], float]],
                 dense_feature_count: int,
                 dense_hidden_layers_sizes: List[int],
                 output_size: int,
                 device: str = "cpu",
                 *args, **kwargs) -> None:
        super(DenseArch, self).__init__(*args, **kwargs)
        self.mlp = MLP(input_size=dense_feature_count,
                       hidden_sizes=dense_hidden_layers_sizes,
                       output_size=output_size)  # D X O
        self.mean_vector = torch.tensor([metadata[f"DENSE_{i}"]["mean"] for i in range(dense_feature_count)],
                                        device=device)
        self.std_vector = torch.tensor([metadata[f"DENSE_{i}"]["std"] for i in range(dense_feature_count)],
                                       device=device)

    def _normalize_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.mean_vector) / self.std_vector

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input : B X D # Output : B X O
        return self.mlp(self._normalize_inputs(inputs))


class SparseFeatureLayer(nn.Module):
    def __init__(self, cardinality: int, embedding_size: int):
        super(SparseFeatureLayer, self).__init__()
        self.embedding = nn.Embedding(cardinality, embedding_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input : B X 1 # Output : B X E
        embeddings = self.embedding(inputs)
        return embeddings


class SparseArch(nn.Module):
    def __init__(self,
                 metadata: Dict[str, Union[int, List[int], Dict[str, int]]],
                 sparse_feature_count: int,
                 embedding_sizes: Mapping[str, int],
                 device: str = "cpu",
                 *args, **kwargs) -> None:
        super(SparseArch, self).__init__(*args, **kwargs)
        self.num_sparse_features = sparse_feature_count
        self._modulus_hash_sizes = [metadata[f"SPARSE_{i}"]["cardinality"] for i in
                                    range(self.num_sparse_features)]
        self.sparse_layers = nn.ModuleList([
            SparseFeatureLayer(cardinality=self._modulus_hash_sizes[i],
                               embedding_size=embedding_sizes[f"SPARSE_{i}"]) for
            i in range(self.num_sparse_features)
        ])
        self.total_embedding_size = sum(embedding_sizes.values())
        self.mapping = [
            torch.tensor(metadata[f"SPARSE_{i}"]["tokenizer_values"],
                         device=device) for i in
            range(self.num_sparse_features)]
        self.cardinality_tensor = torch.tensor(self._modulus_hash_sizes).to(
            device)

    @staticmethod
    def index_hash(tensor: torch.Tensor,
                   tokenizer_values: Union[List[int], torch.Tensor]):
        tensor = tensor.view(-1, 1)
        matches = tensor == tokenizer_values.view(1, -1)
        indices = torch.argmax(matches.to(torch.int64), dim=1)
        return indices

    @staticmethod
    def modulus_hash(tensor: torch.Tensor, cardinality: torch.Tensor):
        return (tensor + 1) % cardinality

    def _forward_index_hash(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        output_values = []
        for i in range(self.num_sparse_features):
            indices = self.index_hash(inputs[:, i], self.mapping[i])
            sparse_out = self.sparse_layers[i](indices)
            output_values.append(sparse_out)
        return output_values

    def _forward_modulus_hash(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        sparse_hashed = self.modulus_hash(inputs, self.cardinality_tensor)
        return [sparse_layer(sparse_hashed[:, i]) for i, sparse_layer in enumerate(self.sparse_layers)]

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        return self._forward_modulus_hash(inputs)


class DenseSparseInteractionLayer(nn.Module):
    SUPPORTED_INTERACTION_TYPES = ["dot", "cat"]

    def __init__(self, interaction_type: str = "dot"):
        super(DenseSparseInteractionLayer, self).__init__()
        if interaction_type not in self.SUPPORTED_INTERACTION_TYPES:
            raise ValueError(
                f"Interaction type {interaction_type} not supported. "
                f"Supported types are {self.SUPPORTED_INTERACTION_TYPES}")
        self.interaction_type = interaction_type

    def forward(self, dense_out: torch.Tensor,
                sparse_out: List[torch.Tensor]) -> Tensor:
        concat = torch.cat([dense_out] + sparse_out, dim=-1)
        if self.interaction_type == "dot":
            concat = concat.unsqueeze(2)
            out = torch.bmm(concat, torch.transpose(concat, 1, 2))
        else:
            out = concat
        flattened = torch.flatten(out, 1)
        return flattened


class PredictionLayer(nn.Module):
    def __init__(self,
                 dense_out_size: int,
                 sparse_out_sizes: List[int],
                 interaction_type: str,
                 hidden_sizes: List[int], *wargs, **kwargs):
        super(PredictionLayer, self).__init__(*wargs, **kwargs)
        concat_size = sum(sparse_out_sizes) + dense_out_size
        input_size = concat_size * concat_size if interaction_type == "dot" else concat_size
        self.mlp = MLP(input_size=input_size,
                       hidden_sizes=hidden_sizes, output_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mlp_out = self.mlp(inputs)
        result = self.sigmoid(mlp_out)
        return result


@dataclass
class Parameters:
    dense_input_feature_size: int
    sparse_input_feature_size: int
    sparse_embedding_sizes: Mapping[str, int]
    dense_mlp: Dict[str, Union[List[int], int]]
    prediction_hidden_sizes: List[int]
    interaction_type: str
    use_modulus_hash: bool = False


class DLRM(nn.Module):
    def __init__(self, metadata: Dict[str, Union[int, List[int]]],
                 parameters: Parameters,
                 device: str = "cpu"):
        super(DLRM, self).__init__()
        self.dense_layer = DenseArch(
            metadata=metadata,
            dense_feature_count=parameters.dense_input_feature_size,
            dense_hidden_layers_sizes=parameters.dense_mlp[
                "hidden_layer_sizes"],
            output_size=parameters.dense_mlp["output_size"],
            device=device)
        self.sparse_layer = SparseArch(
            metadata=metadata,
            sparse_feature_count=parameters.sparse_input_feature_size,
            embedding_sizes=parameters.sparse_embedding_sizes,
            device=device
        )
        self.interaction_layer = DenseSparseInteractionLayer(interaction_type=parameters.interaction_type)
        self.prediction_layer = PredictionLayer(
            dense_out_size=parameters.dense_mlp["output_size"],
            sparse_out_sizes=[parameters.sparse_embedding_sizes[f"SPARSE_{i}"]
                              for i in
                              range(len(parameters.sparse_embedding_sizes))],
            interaction_type=parameters.interaction_type,
            hidden_sizes=parameters.prediction_hidden_sizes,
        )

    def forward(self, dense_features: torch.Tensor,
                sparse_features: torch.Tensor) -> float:
        dense_out = self.dense_layer(dense_features)
        sparse_out = self.sparse_layer(sparse_features)
        ds_out = self.interaction_layer(dense_out, sparse_out)
        return self.prediction_layer(ds_out).squeeze()


def read_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


@click.command()
@click.option('--file_path',
              type=click.Path(exists=True), help='Path to the parquet file',
              default="data/sample_criteo_data.parquet")
@click.option('--metadata_path',
              type=click.Path(exists=True), help='Path to the metadata file',
              default="data/sample_criteo_metadata.json")
def dry_run_with_data(file_path, metadata_path):
    """
    Process the file specified by --file_path and use metadata from --metadata_path.
    """
    logger.info("Reading the parquet file {}...".format(file_path))
    logger.info("Reading the metadata file {}...".format(metadata_path))

    dataset = CriteoParquetDataset(file_path=file_path)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    metadata = read_metadata(metadata_path)

    labels, dense, sparse = next(iter(data_loader))
    logger.info("Labels size: {}".format(labels.size()))
    logger.info("Dense size: {}".format(dense.size()))
    logger.info("Sparse size: {}".format(sparse.size()))

    dense_mlp_out_size = 16
    num_dense_features = dense.size()[1]
    dense_arch = DenseArch(metadata=metadata,
                           dense_feature_count=num_dense_features,
                           dense_hidden_layers_sizes=[32],
                           output_size=dense_mlp_out_size, )
    # dense_arch_optim = torch.compile(dense_arch)
    dense_out = dense_arch(dense)
    logger.info("Dense out size: {}".format(dense_out.size()))

    embedding_size = 16
    embedding_sizes = {fn: embedding_size for fn in metadata.keys()}
    sparse_mlp_out_size = 16
    sparse_arch = SparseArch(metadata=metadata,
                             sparse_feature_count=sparse.size()[1],
                             embedding_sizes=embedding_sizes)
    # compiled model hangs on running with inputs
    # sparse_arch_optim = torch.compile(sparse_arch)
    sparse_out = sparse_arch(sparse)
    logger.info("Sparse out size: {}".format(sparse_out[0].size()))

    dense_sparse_interaction_layer = DenseSparseInteractionLayer()
    # dense_sparse_interaction_layer_optim = torch.compile(dense_sparse_interaction_layer)
    ds_out = dense_sparse_interaction_layer(dense_out, sparse_out)
    logger.info("Dense sparse interaction out size: {}".format(ds_out.size()))

    prediction_layer = PredictionLayer(dense_out_size=dense_mlp_out_size,
                                       sparse_out_sizes=[sparse_mlp_out_size] * len(sparse_out),
                                       hidden_sizes=[16])
    # prediction_layer_optim = torch.compile(prediction_layer)
    pred_out = prediction_layer(ds_out)
    logger.info("Prediction out size: {}".format(pred_out.size()))
    logger.info("Prediction out value: {}".format(pred_out))

    # TODO dry run the DLRM model
    with open("model_hyperparameters.json", 'r') as f:
        hyperparameters = json.load(f)
    parameters = Parameters(
        dense_input_feature_size=hyperparameters['dense_input_feature_size'],
        sparse_input_feature_size=hyperparameters['sparse_input_feature_size'],
        sparse_embedding_sizes=hyperparameters['sparse_embedding_sizes'],
        dense_mlp=hyperparameters['dense_mlp'],
        prediction_hidden_sizes=hyperparameters['prediction_hidden_sizes'],
        interaction_type=hyperparameters['dense_sparse_interaction_type'],
        use_modulus_hash=hyperparameters['use_modulus_hash'],
    )
    import torch._dynamo
    torch._dynamo.reset()
    torch._dynamo.config.verbose = True
    dlrm = DLRM(metadata, parameters)
    _ = dlrm(dense, sparse)

    # dlrm_optim = torch.compile(dlrm, backend="aot_eager")
    dlrm_optim = torch.compile(dlrm, backend="inductor", fullgraph=True, mode="max-autotune")
    # logger.info("Compiled DLRM model: {}".format(dlrm))
    start = time.time()
    prediction = dlrm_optim(dense, sparse)
    logger.info("DLRM prediction size: {}".format(prediction.size()))
    logger.info(
        "[COMPILED] Time taken for prediction: {}".format(time.time() - start))


if __name__ == "__main__":
    dry_run_with_data()
