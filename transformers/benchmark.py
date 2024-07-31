import time
from typing import List

from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

from dataset import TextFileDataset

import click
import torch

from gpt import GPT2


def plot_results(results: List[float]) -> None:
    mean_tokens_processed = sum(results) / len(results)
    x = np.array(results)
    y = np.arange(len(x))
    fig, ax = plt.subplots()
    ax.plot(y, x, '--', linewidth=2, label='Tokens/s')
    ax.set(xlabel='Iteration', ylabel='Tokens processed per second')
    ax.hlines(sum(results) / len(results), 0, len(x), colors='r', label=f'Mean={mean_tokens_processed:.0f}')
    ax.legend()
    plt.show()


def benchmark_model(model: nn.Module, dl: DataLoader) -> None:
    results = []
    skip = 5
    for _ in range(100):
        time_start = time.time()
        input_tokens, next_token = next(dl)
        _ = model(input_tokens.cuda())
        tokens_processed = input_tokens.size(1) * input_tokens.size(0) / (time.time() - time_start)
        logger.info(f"Tokens processed per second: {tokens_processed:.0f}")
        if skip == 0:
            results.append(tokens_processed)
        else:
            skip -= 1

    mean_tokens_processed = sum(results) / len(results)
    logger.info(f"Mean tokens processed per second: {mean_tokens_processed:.0f}")
    plot_results(results=results)


@click.command()
@click.option('--torch-compile', is_flag=True, help='Compile the model using torch.compile')
@click.option('--use-hf', is_flag=True, help='Use HuggingFace model')
@click.option('--use-tensorcores', is_flag=True, help='Use CUDA Tensor Cores')
@click.option("--batch-size", type=int, default=8, help="Batch size")
def main(torch_compile: bool, use_hf: bool, use_tensorcores: bool, batch_size: int):
    if use_tensorcores:
        torch.set_float32_matmul_precision('high')

    dataset = TextFileDataset("data/1984.txt", sequence_length=1024)
    dl = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False))

    if use_hf:
        model = GPT2LMHeadModel.from_pretrained('gpt2', resume_download=None)
    else:
        model = GPT2.from_pretrained()

    if torch.cuda.is_available():
        model = model.cuda()

    if torch_compile:
        time_start = time.time()
        model = torch.compile(model, backend="inductor", fullgraph=True, mode="max-autotune")
        _ = model(next(dl)[0].cuda())
        logger.info(f"Compilation time: {time.time() - time_start:.2f}")

    benchmark_model(model=model, dl=dl)


if __name__ == "__main__":
    main()
