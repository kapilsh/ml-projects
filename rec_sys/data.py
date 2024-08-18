

from torch.utils.data import Dataset, DataLoader
from loguru import logger
import pandas as pd
import numpy as np
from typing import List, Dict
import torch

from dataclasses import dataclass

@dataclass
class MovieLensMetadata:
    unique_user_ids: List[int]
    unique_movie_ids: List[int]
    user_id_tokens: Dict[int, int]
    movie_id_tokens: Dict[int, int]


@dataclass
class MovieLensSequenceData:
    movie_token_inputs: torch.Tensor
    rating_inputs: torch.Tensor
    user_id_tokens: torch.Tensor
    movie_token_outputs: torch.Tensor
    rating_outputs: torch.Tensor
    length: int


EOS_TOKEN = -1 # end of sequence token
EOS_RATING = 2 # end of sequence rating (average rating)

class MovieLensSequenceDataset(Dataset):


    def __init__(self, movies_file, users_file, ratings_file, sequence_length, window_size=1):
        users, movies, ratings = self._read_data(movies_file, users_file, ratings_file)
        self.metadata = self._generate_metadata(users, movies)
        users, movies, ratings = self._add_tokens(users, movies, ratings)
        self.data = self._generate_sequences(ratings, sequence_length, window_size)
    
    def _read_data(self, movies_file, users_file, ratings_file):
        logger.info("Reading data from files")
        users = pd.read_csv(
            "./data/ml-1m/users.dat",
            sep="::",
            names=["user_id", "sex", "age_group", "occupation", "zip_code"],
            encoding="ISO-8859-1",
            engine="python",
            dtype={"user_id": np.int32, "sex": "category", "age_group": "category", "occupation": "category", "zip_code": str},
        )

        ratings = pd.read_csv(
            "./data/ml-1m/ratings.dat",
            sep="::",
            names=["user_id", "movie_id", "rating", "unix_timestamp"],
            encoding="ISO-8859-1",
            engine="python",
            dtype={"user_id": np.int32, "movie_id": np.int32, "rating": np.int8, "unix_timestamp": np.int32},
        )

        movies = pd.read_csv(
            "./data/ml-1m/movies.dat",
            sep="::",
            names=["movie_id", "title", "genres"],
            encoding="ISO-8859-1",
            engine="python",
            dtype={"movie_id": np.int32, "title": str, "genres": str},
        )

        return users, movies, ratings

    def _generate_metadata(self, users, movies)-> MovieLensMetadata:
        # user ids
        unique_user_ids = users["user_id"].unique()
        unique_user_ids.sort()

        # movie ids
        unique_movie_ids = movies["movie_id"].unique()
        unique_movie_ids.sort()

        # tokenization
        user_id_tokens = {user_id: i for i, user_id in enumerate(unique_user_ids)}
        movie_id_tokens = {movie_id: i for i, movie_id in enumerate(unique_movie_ids)}

        return MovieLensMetadata(
            unique_user_ids=unique_user_ids,
            unique_movie_ids=unique_movie_ids,
            user_id_tokens=user_id_tokens,
            movie_id_tokens=movie_id_tokens,
        )
    
    def _add_tokens(self, users, movies, ratings):
        logger.info("Adding tokens to data")
        users["user_id_token"] = users["user_id"].map(self.metadata.user_id_tokens)
        movies["movie_id_token"] = movies["movie_id"].map(self.metadata.movie_id_tokens)
        ratings["user_id_token"] = ratings["user_id"].map(self.metadata.user_id_tokens)
        ratings["movie_id_token"] = ratings["movie_id"].map(self.metadata.movie_id_tokens)
        return users, movies, ratings

    def _generate_sequences(self, ratings, sequence_length, window_size):
        logger.info("Generating sequences")
        
        ratings_ordered = ratings[["user_id_token", "movie_id_token", "unix_timestamp", "rating"]].sort_values(
            by="unix_timestamp").groupby("user_id_token").agg(list).reset_index()
        
        def generate_row_sequences(row, sequence_length, window_size):
            movie_ids = torch.tensor(row.movie_id_token, dtype=torch.int32)
            ratings = torch.tensor(row.rating, dtype=torch.int8)
            movie_input_sequences = movie_ids.ravel().unfold(0, sequence_length, window_size).to(torch.int32)
            rating_input_sequences = ratings.ravel().unfold(0, sequence_length, window_size).to(torch.int8)
            return (movie_input_sequences, rating_input_sequences)

        movie_token_inputs_list = []
        rating_inputs_list = []
        user_id_tokens_list = []
        movie_token_outputs_list = []
        rating_outputs_list = []

        for _, row in ratings_ordered.iterrows():
            movie_sequences, rating_sequences = generate_row_sequences(row, sequence_length, window_size)
            movie_token_inputs_list.append(movie_sequences)
            rating_inputs_list.append(rating_sequences)
            user_id_tokens_list.append(row.user_id_token)
            movie_token_outputs_list.append(
                torch.cat([movie_sequences[1:, -1], torch.tensor([EOS_TOKEN], dtype=torch.int32)])
            )
            rating_outputs_list.append(
                torch.cat([rating_sequences[1:, -1], torch.tensor([EOS_RATING], dtype=torch.int8)])
            )
        
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
            length=len(user_id_tokens_list),
        )

    def __len__(self):
        return self.data.length
    
    def __getitem__(self, idx) -> MovieLensSequenceData:
        return (
            self.data.movie_token_inputs[idx:idx+1],
            self.data.rating_inputs[idx:idx+1],
            self.data.user_id_tokens[idx:idx+1],
            self.data.movie_token_outputs[idx:idx+1],
            self.data.rating_outputs[idx:idx+1]
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