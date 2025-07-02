import os
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import defaultdict
from .utils import extract_names, extract_director


class SmartMovieRecommender:
    def __init__(self, movies_csv, credits_csv, model_path):
        print("[INIT] Loading SentenceTransformer model from:", model_path)
        self.model = SentenceTransformer(model_path)

        self.user_memory = []
        self.user_profile = None
        self.genre_preferences = defaultdict(float)

        self.movies = self._preprocess(movies_csv, credits_csv)
        print(f"[INIT] Loaded {len(self.movies)} movies")

        self.embeddings = self._compute_embeddings(self.movies['tags'])
        print(f"[INIT] Embedding shape: {self.embeddings.shape}")

        self.index = self._build_faiss_index(self.embeddings)
        print("[INIT] System ready ✅")

        self.load_user_state()

    def _preprocess(self, movies_csv, credits_csv):
        cache_file = "movie_recommender/cache/movies.pkl"
        if os.path.exists(cache_file):
            print("[CACHE] Loading preprocessed movies...")
            return pd.read_pickle(cache_file)

        print("[PREPROCESS] Processing raw CSV files...")
        movies = pd.read_csv(movies_csv)
        credits = pd.read_csv(credits_csv)
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

        df['tags'] = df['tags'].astype(str).str.lower()

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        df.to_pickle(cache_file)
        return df.reset_index(drop=True)

    def _compute_embeddings(self, texts):
        embed_path = "movie_recommender/cache/embeddings.npy"
        if os.path.exists(embed_path):
            print("[CACHE] Loading cached embeddings...")
            return np.load(embed_path)

        print("[EMBEDDING] Generating new embeddings...")
        embeddings = normalize(self.model.encode(texts.tolist(), show_progress_bar=True))
        np.save(embed_path, embeddings)
        return embeddings

    def _build_faiss_index(self, embeddings):
        index_path = "movie_recommender/cache/faiss.index"
        if os.path.exists(index_path):
            print("[CACHE] Loading cached FAISS index...")
            return faiss.read_index(index_path)

        print("[FAISS] Creating new FAISS index...")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, index_path)
        return index

    def _update_user_memory(self, vector):
        self.user_memory.insert(0, vector)
        self.user_memory = self.user_memory[:10]

    def _update_user_profile(self, vector, alpha=0.3):
        if self.user_profile is None:
            self.user_profile = vector.copy()
        else:
            updated = (1 - alpha) * self.user_profile + alpha * vector
            self.user_profile = normalize(updated)[0].reshape(1, -1)

    def recommend(self, title: str, top_k: int = 5):
        if title not in self.movies['title'].values:
            raise ValueError(f"'{title}' not found in dataset.")

        idx = self.movies[self.movies['title'] == title].index[0]
        vector = self.embeddings[idx].reshape(1, -1)

        self._update_user_memory(vector)
        self._update_user_profile(vector)

        _, indices = self.index.search(vector.astype(np.float32), top_k + 1)
        return [self.movies.iloc[i].title for i in indices[0] if i != idx][:top_k]

    def recommend_from_memory(self, top_k: int = 5):
        if not self.user_memory:
            raise ValueError("No recent memory.")

        avg_vec = normalize(np.mean(self.user_memory, axis=0)).reshape(1, -1)
        _, indices = self.index.search(avg_vec.astype(np.float32), top_k * 2)  # extra results

        seen = set()
        results = []
        for i in indices[0]:
            title = self.movies.iloc[i].title
            if title not in seen:
                results.append(title)
                seen.add(title)
            if len(results) == top_k:
                break

        return results


    def recommend_from_profile(self, top_k: int = 5, explain=False):
        if self.user_profile is None:
            raise ValueError("No user profile available.")

        _, indices = self.index.search(self.user_profile.astype(np.float32), 50)
        results = []

        for i in indices[0]:
            movie = self.movies.iloc[i]
            genres = movie.genres
            genre_score = sum([self.genre_preferences.get(g, 0) for g in genres])
            title = movie.title
            if explain:
                results.append({
                    "title": title,
                    "genres": genres,
                    "genre_score": round(genre_score, 2)
                })
            else:
                results.append(title)

        if explain:
            sorted_results = sorted(results, key=lambda x: x["genre_score"], reverse=True)[:top_k]
            for r in sorted_results:
                print(f"[EXPLAIN] {r['title']}: genre_score={r['genre_score']} from genres={r['genres']}")
            return [r['title'] for r in sorted_results]
        else:
            return results[:top_k]

    def reset_profile(self):
        self.user_profile = None
        self.user_memory = []
        self.genre_preferences = defaultdict(float)

    def give_feedback(self, title, score: float):
        if title not in self.movies['title'].values:
            print(f"[FEEDBACK] '{title}' not found.")
            return

        idx = self.movies[self.movies['title'] == title].index[0]
        vector = self.embeddings[idx].reshape(1, -1)
        genres = self.movies.iloc[idx]['genres']

        alpha = min(max(abs(score), 0.1), 1.0)
        sign = np.sign(score)

        if sign > 0:
            print(f"[FEEDBACK] 👍 Liked '{title}' with score {score}")
            self._update_user_profile(vector, alpha=alpha)
            for genre in genres:
                self.genre_preferences[genre] += score
        else:
            print(f"[FEEDBACK] 👎 Disliked '{title}' with score {score}")
            for genre in genres:
                self.genre_preferences[genre] -= abs(score)

    def decay_preferences(self, decay_rate=0.01):
        for g in list(self.genre_preferences):
            self.genre_preferences[g] *= (1 - decay_rate)
            if abs(self.genre_preferences[g]) < 0.01:
                del self.genre_preferences[g]

    def save_user_state(self, folder="movie_recommender/cache"):
        os.makedirs(folder, exist_ok=True)
        if self.user_profile is not None:
            np.save(os.path.join(folder, "user_profile.npy"), self.user_profile)
        if self.user_memory:
            np.save(os.path.join(folder, "user_memory.npy"), np.array(self.user_memory))
        with open(os.path.join(folder, "genre_prefs.json"), "w") as f:
            json.dump(self.genre_preferences, f)

    def load_user_state(self, folder="movie_recommender/cache"):
        try:
            self.user_profile = np.load(os.path.join(folder, "user_profile.npy")).reshape(1, -1)
            self.user_memory = list(np.load(os.path.join(folder, "user_memory.npy")))
            with open(os.path.join(folder, "genre_prefs.json")) as f:
                self.genre_preferences = defaultdict(float, json.load(f))
            print("[USER] Loaded previous user state.")
        except FileNotFoundError:
            print("[USER] No saved user state found.")
