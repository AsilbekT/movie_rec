import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from movie_recommender.core import MovieRecommender
from movie_recommender.config import MOVIES_CSV, CREDITS_CSV


if __name__ == '__main__':
    recommender = MovieRecommender(MOVIES_CSV, CREDITS_CSV)

    movie_title = "Avatar"
    try:
        results = recommender.recommend(movie_title)
        print(f"Top recommendations for '{movie_title}':")
        for movie in results:
            print("•", movie)
    except ValueError as e:
        print("Error:", e)
