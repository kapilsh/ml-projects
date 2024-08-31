from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_popular_movies(ratings_file: str, n: int):
    # Read the ratings data
    ratings = pd.read_csv(
        ratings_file,
        sep="::",
        names=["user_id", "movie_id", "rating", "unix_timestamp"],
        encoding="ISO-8859-1",
        engine="python",
        dtype={
            "user_id": np.int32,
            "movie_id": np.int32,
            "rating": np.float16,
            "unix_timestamp": np.int32,
        },
    )
    # Get the most popular movies
    rating_counts = ratings["movie_id"].value_counts().reset_index()
    rating_counts.columns = ["movie_id", "rating_count"]

    # Get the most frequently rated movies
    min_ratings_threshold = rating_counts["rating_count"].quantile(0.95)

    # Filter movies based on the minimum number of ratings
    popular_movies = ratings.merge(rating_counts, on="movie_id")
    popular_movies = popular_movies[
        popular_movies["rating_count"] >= min_ratings_threshold
    ]

    # Calculate the average rating for each movie
    average_ratings = popular_movies.groupby("movie_id")["rating"].mean().reset_index()

    # Get the top rated movies
    return average_ratings.sort_values("rating", ascending=False).head(n)


@dataclass
class ModelOutput:
    predictions: torch.Tensor
    scores: torch.Tensor


@dataclass
class Metrics:
    MAP: float
    MRR: float
    NDCG: float


def _drop_duplicates(
    output_tokens: torch.Tensor,
    scores: torch.Tensor,
    input_tokens: torch.Tensor,
    n: int,
) -> ModelOutput:
    for row in range(output_tokens.shape[0]):
        try:
            merged, counts = torch.cat((output_tokens[row], input_tokens[row])).unique(
                return_counts=True
            )
        except:
            print("Error")
            import ipdb

            ipdb.set_trace()
            print(output_tokens[row])
            print(input_tokens[row])

        intersection = merged[torch.where(counts.gt(1))]
        dedup_mask = torch.isin(output_tokens[row], intersection, invert=True)
        output_tokens[row, :n] = output_tokens[row][dedup_mask][:n]
        scores[row, :n] = scores[row][dedup_mask][:n]
    return ModelOutput(predictions=output_tokens[:, :n], scores=scores[:, :n])


def get_model_predictions(
    model: nn.Module,
    movie_id_tokens: torch.Tensor,
    user_id_tokens: torch.Tensor,
    n: int,
) -> ModelOutput:
    with torch.no_grad():
        # batch x num_tokens
        output = model(movie_id_tokens, user_id_tokens)

    output_probabilites = F.softmax(output, dim=-1)

    # get top k predictions
    # we work with the top n + movie_id_tokens.shape[-1] to ensure
    # that we do not recommended the movies that the user has already seen
    scores, top_tokens = output_probabilites.topk(n + movie_id_tokens.shape[-1], dim=-1)

    return _drop_duplicates(top_tokens, scores, movie_id_tokens, n)


def get_popular_movie_predictions(
    popular_movies: pd.DataFrame, input_tokens: torch.Tensor, n: int, batch_size: int
) -> ModelOutput:
    predictions = torch.from_numpy(popular_movies["movie_id"].values).repeat(
        batch_size, 1
    )
    scores = F.softmax(
        torch.from_numpy(popular_movies["rating"].values).repeat(batch_size, 1),
        dim=-1,
    )
    return _drop_duplicates(predictions, scores, input_tokens, n)


def calculate_relevance(predictions, targets):
    return (predictions == targets.unsqueeze(1)).float()


def calculate_metrics(relevance: torch.Tensor, scores: torch.Tensor) -> Metrics:
    relevant_index = torch.where(relevance == 1)
    reciprocal_ranks = torch.zeros(relevance.shape[0])
    reciprocal_ranks[relevant_index[0]] = 1 / (relevant_index[1].float() + 1)
    _mrr = reciprocal_ranks.mean().item()
    _map = relevance.sum(dim=-1).mean().item()
    _ndcg = ndcg_score(relevance, scores, k=relevance.shape[-1])
    return Metrics(MAP=_map, MRR=_mrr, NDCG=_ndcg)
