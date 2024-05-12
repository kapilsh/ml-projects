import os
from typing import Tuple, List
import pyarrow.parquet as pq

import torch
import pandas as pd
import click
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from loguru import logger


class CriteoParquetDataset(Dataset):
    def __init__(self, path: str):
        self._all_file_paths = [f for f in os.listdir(path)]

    def _read_one_file(self, file_path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        df = pd.read_parquet(file_path)
        label_tensor = torch.from_numpy(df["labels"].values).to(torch.float32)
        dense_columns = [f for f in df.columns if f.startswith("DENSE")]
        sparse_columns = [f for f in df.columns if f.startswith("SPARSE")]
        dense_tensor = torch.from_numpy(df[dense_columns].values)
        sparse_tensor = torch.from_numpy(df[sparse_columns].values)
        return (label_tensor, dense_tensor, sparse_tensor)

    def _get_file_sizes(self, file_paths) -> List[int]:
        file_sizes = []
        for file_path in file_paths:
            # Open the Parquet file
            parquet_file = pq.ParquetFile(file_path)

            # Get the number of rows in the Parquet file
            num_rows = parquet_file.metadata.num_rows
            logger.info(f"Number of rows in {parquet_file} = {num_rows}")
            file_sizes.append(num_rows)
        return file_sizes

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        return self.label_tensor[idx], self.dense_tensor[idx], \
        self.sparse_tensor[idx]


@click.command()
@click.option('--file_path', type=click.Path(exists=True),
              help='Path to the parquet file')
def process_file(file_path):
    """
    Process the file specified by --file_path.
    """
    logger.info("Reading the parquet file {}...".format(file_path))

    dataset = CriteoParquetDataset(file_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    for labels, dense, sparse in data_loader:
        logger.info("Labels: {}".format(labels))
        logger.info("Dense: {}".format(dense))
        logger.info("Sparse: {}".format(sparse))

        logger.info("Labels size and dtype: {}, {}".format(labels.size(), labels.dtype))
        logger.info("Dense size and dtype: {}, {}".format(dense.size(), dense.dtype))
        logger.info("Sparse size and dtype: {}, {}".format(sparse.size(), sparse.dtype))
        break


if __name__ == "__main__":
    process_file()
