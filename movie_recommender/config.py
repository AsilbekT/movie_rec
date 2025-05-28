import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

MOVIES_CSV = os.path.join(DATA_DIR, 'tmdb_5000_movies.csv')
CREDITS_CSV = os.path.join(DATA_DIR, 'tmdb_5000_credits.csv')
