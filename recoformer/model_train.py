import os
import click
from movielens_transformer import MovieLensTransformer, MovieLensTransformerConfig
from torch.utils.data import DataLoader, Dataset
from data import MovieLensSequenceDataset
import torch
import yaml
import torch.nn as nn
from dacite import from_dict
from loguru import logger
import numpy as np
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter


def load_config(config_file: str):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def init_weights(model: MovieLensTransformer):
    initrange = 0.1
    model.movie_transformer.token_embedding.weight.data.uniform_(-initrange, initrange)
    model.user_embedding.weight.data.uniform_(-initrange, initrange)
    for name, p in model.output_layer.named_parameters():
        if "weight" in name:
            p.data.uniform_(-initrange, initrange)
        elif "bias" in name:
            p.data.zero_()


def train_step(
    movie_ids: torch.Tensor,
    user_ids: torch.Tensor,
    movie_targets: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
):
    movie_ids, user_ids, movie_targets = (
        movie_ids.to(device),
        user_ids.to(device),
        movie_targets.to(device),
    )
    optimizer.zero_grad()
    output = model(movie_ids, user_ids)
    loss = criterion(output.view(-1, output.size(-1)), movie_targets.view(-1).long())
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return loss.item()


def validation_step(
    movie_ids: torch.Tensor,
    user_ids: torch.Tensor,
    movie_targets: torch.Tensor,
    model: nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
):
    movie_ids, user_ids, movie_targets = (
        movie_ids.to(device),
        user_ids.to(device),
        movie_targets.to(device),
    )
    with torch.no_grad():
        output = model(movie_ids, user_ids)
        loss = criterion(
            output.view(-1, output.size(-1)), movie_targets.view(-1).long()
        )
    return loss.item()


def get_dataset(config) -> Dataset:
    movies_file = os.path.join(config["trainer_config"]["data_dir"], "movies.dat")
    users_file = os.path.join(config["trainer_config"]["data_dir"], "users.dat")
    ratings_file = os.path.join(config["trainer_config"]["data_dir"], "ratings.dat")

    dataset = MovieLensSequenceDataset(
        movies_file=movies_file,
        users_file=users_file,
        ratings_file=ratings_file,
        sequence_length=config["movie_transformer_config"]["context_window_size"],
        window_size=1,  # next token prediction with sliding window of 1
    )
    return dataset


def get_model_config(config: dict, dataset: Dataset) -> MovieLensTransformerConfig:
    config["movie_transformer_config"]["vocab_size"] = len(
        dataset.metadata.unique_movie_ids
    )
    config["num_users"] = len(dataset.metadata.unique_user_ids)
    model_config = from_dict(data_class=MovieLensTransformerConfig, data=config)
    logger.info(f"Model config:\n ========== \n{model_config} \n ==========")
    return model_config


def run_model_training(config: dict):
    device = config["trainer_config"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    dataset = get_dataset(config)
    train_dataloader = DataLoader(
        dataset, batch_size=config["trainer_config"]["batch_size"], shuffle=True
    )
    validation_dataloader = DataLoader(
        dataset, batch_size=config["trainer_config"]["batch_size"], shuffle=False
    )

    model_config = get_model_config(config, dataset)

    model = MovieLensTransformer(config=model_config)

    init_weights(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["trainer_config"]["starting_learning_rate"]
    )

    best_validation_loss = np.inf

    writer = SummaryWriter(
        log_dir=config["trainer_config"]["tensorboard_dir"], flush_secs=30
    )

    for epoch in range(config["trainer_config"]["num_epochs"]):
        model.train()
        total_loss = 0.0

        pbar = trange(len(train_dataloader))
        pbar.ncols = 150
        for i, (
            movie_ids,
            rating_ids,
            user_ids,
            movie_targets,
            rating_targets,
        ) in enumerate(train_dataloader):
            loss = train_step(
                movie_ids,
                user_ids,
                movie_targets,
                model,
                optimizer,
                criterion,
                device,
            )
            total_loss += loss
            pbar.update(1)
            pbar.set_description(
                f"[Epoch = {epoch}] Current training loss (loss = {np.round(loss, 4)})"
            )
            pbar.refresh()

        pbar.close()
        train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch}, Loss: {np.round(train_loss, 4)}")
        writer.add_scalar("loss/train", train_loss, epoch)

        model.eval()
        total_loss = 0.0

        pbar = trange(len(validation_dataloader))
        pbar.ncols = 150
        for i, (
            movie_ids,
            rating_ids,
            user_ids,
            movie_targets,
            rating_targets,
        ) in enumerate(validation_dataloader):
            loss = validation_step(
                movie_ids,
                user_ids,
                movie_targets,
                model,
                criterion,
                device,
            )
            total_loss += loss
            pbar.update(1)
            pbar.set_description(
                f"[Epoch = {epoch}] Current validation loss (loss = {np.round(loss, 4)})"
            )
            pbar.refresh()

        pbar.close()

        validation_loss = total_loss / len(validation_dataloader)
        logger.info(f"Validation Loss: {np.round(validation_loss, 4)}")
        writer.add_scalar("loss/validation", validation_loss, epoch)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            if not os.path.exists(config["trainer_config"]["model_dir"]):
                os.makedirs(config["trainer_config"]["model_dir"])
            torch.save(
                model.state_dict(),
                os.path.join(config["trainer_config"]["model_dir"], "model.pth"),
            )


@click.command()
@click.option("--config_file", help="config filename")
def main(config_file):
    config = load_config(config_file)
    run_model_training(config)


if __name__ == "__main__":
    main()
