import itertools
import os
from typing import Tuple, List, Optional, Union

import numpy as np
import pyarrow.parquet as pq

import torch
import pandas as pd
import click
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from loguru import logger


class CriteoParquetDataset(Dataset):
    def __init__(self, path: Optional[Union[str, List[str]]] = None,
                 file_path: Optional[str] = None):
        if path is not None and isinstance(path, str):
            path = [path]
        if path is None and file_path is None:
            raise ValueError("Either path or file_path must be provided")
        if path is not None and file_path is not None:
            raise ValueError("Only one of path or file_path must be provided")
        if path is not None:
            self._all_file_paths = list(itertools.chain.from_iterable([
                [os.path.join(p, f) for f in sorted(os.listdir(p),
                                                    key=lambda x: int(
                                                        x.split("_")[3])) if
                 f.endswith(".parquet")] for p in path]))
        else:
            self._all_file_paths = [file_path]
        self._all_sizes = self._get_file_sizes(self._all_file_paths)
        self._cumulative_size = np.cumsum([0] + self._all_sizes)
        self.total_rows = sum(self._all_sizes)
        self.current_reading_index = -1
        self.label_tensor = None
        self.dense_tensor = None
        self.sparse_tensor = None

    @staticmethod
    def _read_one_file(file_path) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        logger.info(f"Reading file {file_path}")
        df = pd.read_parquet(file_path)
        label_tensor = torch.from_numpy(df["labels"].values).to(torch.float32)
        dense_columns = [f for f in df.columns if f.startswith("DENSE")]
        sparse_columns = [f for f in df.columns if f.startswith("SPARSE")]
        dense_tensor = torch.from_numpy(df[dense_columns].values)
        sparse_tensor = torch.from_numpy(df[sparse_columns].values)
        logger.info(f"Read {len(label_tensor)} rows")
        return label_tensor, dense_tensor, sparse_tensor

    @staticmethod
    def _get_file_sizes(file_paths) -> List[int]:
        file_sizes = []
        for file_path in file_paths:
            # Get the number of rows in the Parquet file
            num_rows = pq.ParquetFile(file_path).metadata.num_rows
            logger.info(f"Number of rows in {file_path} = {num_rows}")
            file_sizes.append(num_rows)
        return file_sizes

    def _get_file_index(self, idx):
        for i, size in enumerate(self._all_sizes):
            if idx < size:
                return i
            idx -= size
        return -1

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        file_index = self._get_file_index(idx)
        if file_index == -1:
            raise IndexError("Index out of bounds")
        if self.current_reading_index != file_index:
            (self.label_tensor, self.dense_tensor,
             self.sparse_tensor) = self._read_one_file(
                self._all_file_paths[file_index])
            self.current_reading_index = file_index
        file_read_idx = idx - self._cumulative_size[file_index]
        return (self.label_tensor[file_read_idx],
                self.dense_tensor[file_read_idx],
                self.sparse_tensor[file_read_idx])


@click.command()
@click.option('--dir_path', type=click.Path(exists=True),
              required=True,
              help='Path to the day directory')
def process_file(dir_path: str):
    """
    Process the file specified by --file_path.
    """
    logger.info("Reading the dir {}...".format(dir_path))

    dataset = CriteoParquetDataset(["/Volumes/nas-drive/criteo_converted/day_0",
                                    "/Volumes/nas-drive/criteo_converted/day_1"])
    data_loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    num_batches_read = 0
    for labels, dense, sparse in data_loader:
        num_batches_read += 1
        if num_batches_read % 1000 == 0:
            print(f"Batch {num_batches_read}")
        break

    dataset = CriteoParquetDataset(file_path=os.path.join(dir_path,
                                                          "day_0_gz_0_converted.parquet"))
    data_loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    num_batches_read = 0
    for labels, dense, sparse in data_loader:
        num_batches_read += 1
        if num_batches_read % 1000 == 0:
            print(f"Batch {num_batches_read}")
        break


if __name__ == "__main__":
    process_file()
