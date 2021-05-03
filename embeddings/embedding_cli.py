import json

import click
import numpy as np
import pandas as pd
from loguru import logger

from embeddings.loader import SGLoader, SGNSLoader
from embeddings.train import HyperParameters, SGModelTrainer, SGNSModelTrainer


def read_json_file(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)


@click.group()
def train_cli():
    pass


@train_cli.command()
@click.option("--file-path", required=True, type=str)
@click.option("--model-path", required=True, type=str)
@click.option("--config", required=True, type=str)
@click.option("--neg-sample/--no-neg-sample", default=False)
def train(file_path: str, model_path: str, config: str, neg_sample: bool):
    configuration = read_json_file(config)
    loader_class = SGNSLoader if neg_sample else SGLoader
    kwargs = {
        "file_path": file_path,
        "remove_frequency": configuration["remove_frequency"],
        "batch_size": configuration["batch_size"],
        "context_window": configuration["context_window"],
        "device": configuration["device"]
    }
    if neg_sample:
        kwargs["noise_count"] = configuration["noise_count"]
    loader = loader_class(**kwargs)
    parameters = HyperParameters(epochs=configuration["epochs"],
                                 embed_size=configuration["embed_size"],
                                 lr=configuration["lr"],
                                 snapshot_frequency=configuration[
                                     "snapshot_frequency"])
    trainer_class = SGNSModelTrainer if neg_sample else SGModelTrainer
    trainer = trainer_class(save_path=model_path,
                            loader=loader, parameters=parameters,
                            device=configuration["device"])
    trainer.train()


@click.group()
def test_cli():
    pass


@train_cli.command()
@click.option("--file-path", required=True, type=str)
@click.option("--model-path", required=True, type=str)
@click.option("--config", required=True, type=str)
@click.option("--sample-count", default=10, type=int)
@click.option("--top-k", default=6, type=int)
@click.option("--neg-sample/--no-neg-sample", default=False)
@click.option("--randomize/--no-randomize", default=True)
def test(file_path: str, model_path: str, config: str, sample_count: int,
         top_k: int, neg_sample: bool, randomize: bool):
    configuration = read_json_file(config)
    loader_class = SGNSLoader if neg_sample else SGLoader
    kwargs = {
        "file_path": file_path,
        "remove_frequency": configuration["remove_frequency"],
        "batch_size": configuration["batch_size"],
        "context_window": configuration["context_window"],
        "device": configuration["device"]
    }
    if neg_sample:
        kwargs["noise_count"] = configuration["noise_count"]
    loader = loader_class(**kwargs)
    parameters = HyperParameters(epochs=configuration["epochs"],
                                 embed_size=configuration["embed_size"],
                                 lr=configuration["lr"],
                                 snapshot_frequency=configuration[
                                     "snapshot_frequency"])
    trainer_class = SGNSModelTrainer if neg_sample else SGModelTrainer
    trainer = trainer_class(save_path=model_path,
                            loader=loader, parameters=parameters,
                            device=configuration["device"])
    result = trainer.test(count=sample_count, top_k_count=top_k,
                          random=randomize)

    index = result.words
    values = result.similar_words
    columns = np.arange(len(values[0]))

    result_df = pd.DataFrame(values, index=index, columns=columns)

    pd.options.display.max_columns = 200
    pd.options.display.width = 200

    logger.info(f"\n{result_df}")


main = click.CommandCollection(sources=[train_cli, test_cli])

if __name__ == '__main__':
    main()
