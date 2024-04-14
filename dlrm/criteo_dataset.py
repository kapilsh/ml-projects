import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CriteoParquetDataset(Dataset):
    def __init__(self, file_name: str):
        df = pd.read_parquet(file_name)
        self.total_rows = len(df)
        self.label_tensor = torch.from_numpy(df["labels"].values)
        dense_columns = [f for f in df.columns if f.startswith("DENSE")]
        sparse_columns = [f for f in df.columns if f.startswith("SPARSE")]
        self.dense_tensor = torch.from_numpy(df[dense_columns].values)
        self.sparse_tensor = torch.from_numpy(df[sparse_columns].values)

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        return self.label_tensor[idx], self.dense_tensor[idx], self.sparse_tensor[idx]


if __name__ == "__main__":
    dataset = CriteoParquetDataset("/Volumes/nas-drive/day_0_gz_converted.parquet")
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    for labels, dense, sparse in data_loader:
        print(labels, dense, sparse)
        break
