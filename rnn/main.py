import os

import click
import torch

import pandas as pd

from rnn.model import ModelRunner, LSTMModel, load_parameters, generate_sample
from rnn.loader import DataLoader


@click.group()
def train_cli():
    pass


@train_cli.command()
@click.option("--file-path", required=True, type=str)
@click.option("--param-path", required=True, type=str)
@click.option("--save-path", required=True, type=str)
@click.option("--model-results-path", required=True, type=str)
def train(file_path: str, param_path: str, save_path: str,
          model_results_path: str):
    data_loader = DataLoader(file_path=file_path)
    model = ModelRunner(data_loader=data_loader, save_path=save_path)
    parameters = load_parameters(param_path)
    training_results = model.train(parameters=parameters)
    training_results.to_csv(model_results_path)


@click.group()
def test_cli():
    pass


@test_cli.command()
@click.option("--model-path", required=True, type=str)
@click.option("--seed", required=True, type=str)
@click.option("--sample-length", required=True, type=int)
@click.option("--top-k", required=False, type=int, default=5)
@click.option("--gpu/--no-gpu", default=False)
def test(model_path: str, seed: str, sample_length: int, top_k: int,
         gpu: bool):
    base_dir = os.path.dirname(model_path)
    file_name = os.path.basename(model_path)
    file_path, file_ext = os.path.splitext(file_name)
    all_check_points = [os.path.join(base_dir, x) for x in
                        os.listdir(base_dir) if x.startswith(file_path)
                        and "final" not in x and x.endswith(file_ext)]
    all_check_points.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    final_path = os.path.join(base_dir, f"{file_path}_final{file_ext}")

    with open(final_path, "rb") as f:
        checkpoint = torch.load(f)

    lstm_model = LSTMModel(checkpoint['tokens'],
                           hidden_size=checkpoint['parameters']["hidden_size"],
                           num_layers=checkpoint['parameters']["num_layers"],
                           drop_prob=checkpoint['parameters']["drop_prob"])
    lstm_model.load_state_dict(checkpoint["model"])

    sample = generate_sample(lstm_model, sample_length, seed, top_k,
                             use_gpu=gpu)
    print(sample)

    print("------------------------------------------------------")

    print("Checkpoint Results")

    for f in all_check_points:
        print(f)

        print("------------------------------------------------------")

        with open(final_path, "rb") as f:
            checkpoint = torch.load(f)

        lstm_model = LSTMModel(checkpoint['tokens'],
                               hidden_size=checkpoint['parameters'][
                                   "hidden_size"],
                               num_layers=checkpoint['parameters'][
                                   "num_layers"],
                               drop_prob=checkpoint['parameters'][
                                   "drop_prob"])
        lstm_model.load_state_dict(checkpoint["model"])

        sample = generate_sample(lstm_model, sample_length, seed, top_k,
                                 use_gpu=gpu)
        print(sample)


main = click.CommandCollection(sources=[train_cli, test_cli])

if __name__ == '__main__':
    main()
