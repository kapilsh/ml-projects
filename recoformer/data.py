from torch.utils.data import Dataset, DataLoader
from loguru import logger
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import torch

from dataclasses import dataclass


def read_movielens_data(movies_file: str, users_file: str, ratings_file: str):
    logger.info("Reading data from files")
    users = pd.read_csv(
        users_file,
        sep="::",
        names=["user_id", "sex", "age_group", "occupation", "zip_code"],
        encoding="ISO-8859-1",
        engine="python",
        dtype={
            "user_id": np.int32,
            "sex": "category",
            "age_group": "category",
            "occupation": "category",
            "zip_code": str,
        },
    )

    ratings = pd.read_csv(
        ratings_file,
        sep="::",
        names=["user_id", "movie_id", "rating", "unix_timestamp"],
        encoding="ISO-8859-1",
        engine="python",
        dtype={
            "user_id": np.int32,
            "movie_id": np.int32,
            "rating": np.int8,
            "unix_timestamp": np.int32,
        },
    )

    movies = pd.read_csv(
        movies_file,
        sep="::",
        names=["movie_id", "title", "genres"],
        encoding="ISO-8859-1",
        engine="python",
        dtype={"movie_id": np.int32, "title": str, "genres": str},
    )

    return users, movies, ratings


@dataclass
class MovieLensMetadata:
    unique_user_ids: List[int]
    unique_movie_ids: List[int]
    user_id_tokens: Dict[int, int]
    movie_id_tokens: Dict[int, int]
    sos_id: int
    eos_id: int


@dataclass
class MovieLensSequenceData:
    movie_token_inputs: torch.Tensor
    rating_inputs: torch.Tensor
    user_id_tokens: torch.Tensor
    movie_token_outputs: torch.Tensor
    rating_outputs: torch.Tensor
    length: int


class MovieLensSequenceDataset(Dataset):
    SOS_RATING = 0
    EOS_RATING = 3

    def __init__(
        self,
        movies_file: str,
        users_file: str,
        ratings_file: str,
        sequence_length: int,
        window_size: int = 1,
        is_validation: bool = False,
        validation_fraction: float = 0.1,
    ):
        self.is_validation = is_validation
        logger.info(
            "Creating MovieLensSequenceDataset with validation set: %s", is_validation
        )
        users, movies, ratings = read_movielens_data(
            movies_file, users_file, ratings_file
        )
        self.metadata = self._generate_metadata(users, movies)

        users, movies, ratings = self._add_tokens(users, movies, ratings)
        sos_token = self.metadata.movie_id_tokens[self.metadata.sos_id]
        eos_token = self.metadata.movie_id_tokens[self.metadata.eos_id]
        data = self._generate_sequences(
            ratings,
            sequence_length,
            window_size,
            sos_token=sos_token,
            eos_token=eos_token,
            sos_rating=self.SOS_RATING,
            eos_rating=self.EOS_RATING,
        )
        train_data, validation_data = self._split_data(data, validation_fraction)
        logger.info(f"Train data length: {train_data.length}")
        logger.info(f"Validation data length: {validation_data.length}")
        self.data = validation_data if is_validation else train_data

    def _generate_metadata(self, users, movies) -> MovieLensMetadata:
        # user ids
        unique_user_ids = users["user_id"].unique()
        unique_user_ids.sort()

        # movie ids
        unique_movie_ids = movies["movie_id"].unique()
        sos_id = max(unique_movie_ids) + 1
        eos_id = sos_id + 1
        unique_movie_ids = np.append(unique_movie_ids, [sos_id, eos_id])
        unique_movie_ids.sort()

        # tokenization
        user_id_tokens = {user_id: i for i, user_id in enumerate(unique_user_ids)}
        movie_id_tokens = {movie_id: i for i, movie_id in enumerate(unique_movie_ids)}

        return MovieLensMetadata(
            unique_user_ids=unique_user_ids,
            unique_movie_ids=unique_movie_ids,
            user_id_tokens=user_id_tokens,
            movie_id_tokens=movie_id_tokens,
            sos_id=sos_id,
            eos_id=eos_id,
        )

    def _add_tokens(self, users, movies, ratings):
        logger.info("Adding tokens to data")
        users["user_id_token"] = users["user_id"].map(self.metadata.user_id_tokens)
        movies["movie_id_token"] = movies["movie_id"].map(self.metadata.movie_id_tokens)
        ratings["user_id_token"] = ratings["user_id"].map(self.metadata.user_id_tokens)
        ratings["movie_id_token"] = ratings["movie_id"].map(
            self.metadata.movie_id_tokens
        )
        return users, movies, ratings

    def _generate_sequences(
        self,
        ratings_data: pd.DataFrame,
        sequence_length: int,
        window_size: int,
        sos_token: int,
        eos_token: int,
        sos_rating: int,
        eos_rating: int,
    ):
        logger.info("Generating sequences")

        ratings_ordered = (
            ratings_data[
                ["user_id_token", "movie_id_token", "unix_timestamp", "rating"]
            ]
            .sort_values(by="unix_timestamp")
            .groupby("user_id_token")
            .agg(list)
            .reset_index()
        )

        def generate_row_sequences(
            row: pd.Series, sequence_length: int, window_size: int
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            movie_id_tokens = torch.tensor(row.movie_id_token, dtype=torch.int32)
            ratings = torch.tensor(row.rating, dtype=torch.int8)

            movie_id_tokens = torch.cat(
                [
                    torch.tensor([sos_token], dtype=torch.int32),
                    movie_id_tokens,
                    torch.tensor([eos_token], dtype=torch.int32),
                ]
            )
            ratings = torch.cat(
                [
                    torch.tensor([sos_rating], dtype=torch.int8),
                    ratings,
                    torch.tensor([eos_rating], dtype=torch.int8),
                ]
            )

            movie_input_sequences = (
                movie_id_tokens.ravel()
                .unfold(0, sequence_length, window_size)
                .to(torch.int32)
            )
            rating_input_sequences = (
                ratings.ravel().unfold(0, sequence_length, window_size).to(torch.int8)
            )

            movie_output_tokens = movie_input_sequences[1:, -1]
            rating_output_values = rating_input_sequences[1:, -1]

            movie_input_sequences = movie_input_sequences[:-1, :]
            rating_input_sequences = rating_input_sequences[:-1, :]

            return (
                movie_input_sequences,
                rating_input_sequences,
                movie_output_tokens,
                rating_output_values,
            )

        movie_token_inputs_list = []
        rating_inputs_list = []
        user_id_tokens_list = []
        movie_token_outputs_list = []
        rating_outputs_list = []

        for _, row in ratings_ordered.iterrows():
            movies_input, rating_input, movie_output, rating_output = (
                generate_row_sequences(row, sequence_length, window_size)
            )
            movie_token_inputs_list.append(movies_input)
            rating_inputs_list.append(rating_input)
            user_id_tokens_list.extend([row.user_id_token] * movies_input.shape[0])
            movie_token_outputs_list.append(movie_output)
            rating_outputs_list.append(rating_output)

        movie_token_inputs = torch.cat(movie_token_inputs_list)
        rating_inputs = torch.cat(rating_inputs_list)
        user_id_tokens = torch.tensor(user_id_tokens_list, dtype=torch.int32)
        movie_token_outputs = torch.cat(movie_token_outputs_list)
        rating_outputs = torch.cat(rating_outputs_list)

        return MovieLensSequenceData(
            movie_token_inputs=movie_token_inputs,
            rating_inputs=rating_inputs,
            user_id_tokens=user_id_tokens,
            movie_token_outputs=movie_token_outputs,
            rating_outputs=rating_outputs,
            length=movie_token_inputs.shape[0],
        )

    def _split_data(
        self, data: MovieLensSequenceData, validation_fraction: float
    ) -> Tuple[MovieLensSequenceData, MovieLensSequenceData]:
        random_selection = torch.rand(len(data.movie_token_inputs)) <= (
            1 - validation_fraction
        )
        train_data = MovieLensSequenceData(
            movie_token_inputs=data.movie_token_inputs[random_selection],
            rating_inputs=data.rating_inputs[random_selection],
            user_id_tokens=data.user_id_tokens[random_selection],
            movie_token_outputs=data.movie_token_outputs[random_selection],
            rating_outputs=data.rating_outputs[random_selection],
            length=random_selection.int().sum(),
        )
        validation_data = MovieLensSequenceData(
            movie_token_inputs=data.movie_token_inputs[~random_selection],
            rating_inputs=data.rating_inputs[~random_selection],
            user_id_tokens=data.user_id_tokens[~random_selection],
            movie_token_outputs=data.movie_token_outputs[~random_selection],
            rating_outputs=data.rating_outputs[~random_selection],
            length=(~random_selection).int().sum(),
        )
        return train_data, validation_data

    def __len__(self):
        return self.data.length

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.data.movie_token_inputs[idx],
            self.data.rating_inputs[idx],
            self.data.user_id_tokens[idx],
            self.data.movie_token_outputs[idx],
            self.data.rating_outputs[idx],
        )


if __name__ == "__main__":
    dataset = MovieLensSequenceDataset(
        movies_file="./data/ml-1m/movies.dat",
        users_file="./data/ml-1m/users.dat",
        ratings_file="./data/ml-1m/ratings.dat",
        sequence_length=5,
        window_size=1,
    )
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    batch = next(iter(dataloader))
    for t in batch:
        print(t)
        print(t.shape)
        print("-----------------------------------")
