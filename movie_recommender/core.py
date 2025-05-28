import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import extract_names, extract_director


class MovieRecommender:
    def __init__(self, movies_path: str, credits_path: str):
        """
        Initializes the movie recommender system.
        :param movies_path: Path to the TMDB movies CSV file.
        :param credits_path: Path to the TMDB credits CSV file.
        """
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.movies = None
        self.similarity = None
        self._load_data()

    def _load_data(self):
        """Loads and processes movie and credits data."""
        movies = pd.read_csv(self.movies_path)
        credits = pd.read_csv(self.credits_path)

        df = movies.merge(credits, on='title')

        df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].dropna()

        df['genres'] = df['genres'].apply(extract_names)
        df['keywords'] = df['keywords'].apply(extract_names)
        df['cast'] = df['cast'].apply(extract_names)
        df['crew'] = df['crew'].apply(extract_director)

        df['tags'] = (
            df['overview'] + ' ' +
            df['genres'].apply(lambda x: ' '.join(x)) + ' ' +
            df['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
            df['cast'].apply(lambda x: ' '.join(x)) + ' ' +
            df['crew']
        )

        df['tags'] = df['tags'].str.lower()

        self.movies = df
        self._vectorize()

    def _vectorize(self):
        """Converts tags into vector format and computes similarity matrix."""
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(self.movies['tags'].fillna("")).toarray()
        self.similarity = cosine_similarity(vectors)

    def recommend(self, title: str, n: int = 5):
        """
        Recommend similar movies.
        :param title: Movie title to base recommendations on.
        :param n: Number of recommendations to return.
        :return: List of recommended movie titles.
        """
        if title not in self.movies['title'].values:
            raise ValueError(f"'{title}' not found in dataset.")
        
        idx = self.movies[self.movies['title'] == title].index[0]
        distances = list(enumerate(self.similarity[idx]))
        movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:n + 1]
        
        return [self.movies.iloc[i[0]].title for i in movies_list]
