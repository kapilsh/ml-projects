import pandas as pd
import numpy as np


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
