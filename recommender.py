import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, movies_path: str, credits_path: str):
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.movies = None
        self.similarity_matrix = None
        self._load_and_process_data()

    def _load_and_process_data(self):
        movies = pd.read_csv(self.movies_path)
        credits = pd.read_csv(self.credits_path)
        movies = movies.merge(credits, on='title')
        movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

        # Drop missing values
        movies.dropna(inplace=True)

        # Convert JSON-like strings
        movies['genres'] = movies['genres'].apply(self._extract_names)
        movies['keywords'] = movies['keywords'].apply(self._extract_names)
        movies['cast'] = movies['cast'].apply(self._extract_names)
        movies['crew'] = movies['crew'].apply(self._extract_director)

        # Create tags column
        movies['tags'] = (
            movies['overview'] + ' ' +
            movies['genres'].apply(lambda x: ' '.join(x)) + ' ' +
            movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
            movies['cast'].apply(lambda x: ' '.join(x)) + ' ' +
            movies['crew']
        )

        movies['tags'] = movies['tags'].str.lower()
        self.movies = movies
        self._compute_similarity()

    def _extract_names(self, obj: str):
        try:
            return [d['name'] for d in ast.literal_eval(obj)][:5]
        except Exception:
            return []

    def _extract_director(self, obj: str):
        try:
            for d in ast.literal_eval(obj):
                if d['job'] == 'Director':
                    return d['name']
        except Exception:
            return ''
        return ''

    def _compute_similarity(self):
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(self.movies['tags'].values.astype(str)).toarray()
        self.similarity_matrix = cosine_similarity(vectors)

    def recommend(self, movie_title: str, top_n: int = 5):
        if movie_title not in self.movies['title'].values:
            raise ValueError(f"Movie '{movie_title}' not found in dataset.")
        
        idx = self.movies[self.movies['title'] == movie_title].index[0]
        distances = list(enumerate(self.similarity_matrix[idx]))
        recommendations = sorted(distances, key=lambda x: x[1], reverse=True)[1:top_n + 1]
        return [self.movies.iloc[i[0]].title for i in recommendations]
