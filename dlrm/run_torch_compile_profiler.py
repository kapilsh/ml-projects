import click
from loguru import logger
from torch._dynamo.utils import CompileProfiler
from torch.utils.data import DataLoader

from criteo_dataset import CriteoParquetDataset
from dlrm.model import DenseArch, read_metadata, SparseArch, DenseSparseInteractionLayer, PredictionLayer, Parameters, \
    DLRM


@click.command()
@click.option('--file_path', type=click.Path(exists=True), help='Path to the parquet file')
@click.option('--metadata_path', type=click.Path(exists=True), help='Path to the metadata file')
def profile_full_model(file_path, metadata_path):
    logger.info("Reading the parquet file {}...".format(file_path))
    logger.info("Reading the metadata file {}...".format(metadata_path))

    dataset = CriteoParquetDataset(file_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    labels, dense, sparse = next(iter(data_loader))
    logger.info("Labels size: {}".format(labels.size()))
    logger.info("Dense size: {}".format(dense.size()))
    logger.info("Sparse size: {}".format(sparse.size()))

    metadata = read_metadata(metadata_path)
    parameters = Parameters(
        dense_input_feature_size=13,
        sparse_embedding_sizes={
            "SPARSE_{}".format(i): 16 for i in range(26)
        },
        dense_output_size=16,
        sparse_output_size=16,
        dense_hidden_size=16,
        sparse_hidden_size=16,
        prediction_hidden_size=16
    )
    import torch._dynamo
    torch._dynamo.reset()
    torch._dynamo.config.verbose = True
    dlrm = DLRM(metadata, parameters)
    with CompileProfiler() as prof:
        profiler_model = torch.compile(dlrm, backend=prof)
        profiler_model(dense, sparse)
        print(prof.report())


if __name__ == "__main__":
    profile_full_model()
