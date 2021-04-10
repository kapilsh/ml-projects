import click
import torch

import pandas as pd

from autoencoders.convolutional_autoencoder import DataProvider, AutoEncoder, \
    Model


@click.group()
def train_cli():
    pass


@train_cli.command()
@click.option("--data-path", required=True, type=str)
@click.option("--save-path", required=True, type=str)
@click.option("--model-results-path", required=True, type=str)
@click.option("--epochs", required=True, type=int)
@click.option("--batch-size", default=1)
@click.option("--gpu/--no-gpu", default=False)
def train(data_path: str, save_path: str, model_results_path: str, epochs: int,
          batch_size: int, gpu: bool):
    data_provider = DataProvider(root_dir=data_path, batch_size=batch_size)
    model = Model(data_provider=data_provider, use_gpu=gpu)
    training_results = model.train(n_epochs=epochs)
    torch.save(training_results.model.state_dict(), save_path)

    results_df = pd.DataFrame({"TrainingLoss": training_results.train_losses})
    results_df.to_csv(model_results_path)


@click.group()
def test_cli():
    pass


@test_cli.command()
@click.option("--data-path", required=True, type=str)
@click.option("--save-path", required=True, type=str)
@click.option("--batch-size", default=1)
@click.option("--gpu/--no-gpu", default=False)
def test(data_path: str, save_path: str, batch_size: int, gpu: bool):
    data_provider = DataProvider(root_dir=data_path, batch_size=batch_size)
    model = Model(data_provider=data_provider, use_gpu=gpu)
    auto_encoder = AutoEncoder()
    auto_encoder.load_state_dict(torch.load(save_path))
    for input_data, output_data in model.test(auto_encoder):
        continue


main = click.CommandCollection(sources=[train_cli, test_cli])

if __name__ == '__main__':
    main()
