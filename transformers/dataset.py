import os
from typing import Tuple

import torch
from torch import nn
import math
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
import altair as alt
import numpy as np
import pandas as pd


class TextFileDataset:
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    def __init__(self, file_path: str, sequence_length: int) -> None:
        self._text = self._read_text_file(file_path)
        self._sequence_length = sequence_length
        self._tokens = self._tokenize(self._text)
        self._next_token = torch.cat(
            [self._tokens[1:, -1],
             torch.tensor([self.tokenizer.eos_token_id], dtype=torch.int32)]
        )

    @staticmethod
    def _read_text_file(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    def _tokenize(self, text: str) -> torch.Tensor:
        encoded_input = self.tokenizer(text, return_tensors='pt')
        tokens = encoded_input["input_ids"].ravel().unfold(0, self._sequence_length, 1).to(torch.int32)
        return tokens

    def __len__(self) -> int:
        return len(self._next_token)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._tokens[idx], self._next_token[idx]


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))
    print(file_dir)
    dataset = TextFileDataset(os.path.join(file_dir, "data/1984.txt"), 5)
    dataloader = DataLoader(dataset, batch_size=7, shuffle=False)
    for batch in dataloader:
        print(batch)
        break