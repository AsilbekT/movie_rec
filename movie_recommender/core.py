# movie_recommender/core.py (Fully Updated and Corrected)
import os
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import defaultdict
from .utils import extract_names, extract_director
from reinforcement.agent import QLearningAgent

from sqlalchemy import Connection, select, insert, update, delete
from datetime import datetime, timezone 


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import users_table, user_profiles_table, user_memory_entries_table, user_genre_preferences_table



class SmartMovieRecommender:
    def __init__(self, movies_csv, credits_csv, model_path):
        print("[INIT] Loading SentenceTransformer model from:", model_path)
        self.model = SentenceTransformer(model_path)
        self.current_user_id: int | None = None 
        self._user_profile: np.ndarray | None = None
        self._user_memory: list[np.ndarray] = []
        self._genre_preferences: defaultdict = defaultdict(float)

        self.movies = self._preprocess(movies_csv, credits_csv)
        print(f"[INIT] Loaded {len(self.movies)} movies")
        self.rl_agent = QLearningAgent(action_space=list(range(len(self.movies))))

        self.embeddings = self._compute_embeddings(self.movies['tags'])
        print(f"[INIT] Embedding shape: {self.embeddings.shape}")
        
        self.index = self._build_faiss_index(self.embeddings)
        print("[INIT] System ready ✅")

    def _user_cache_dir(self, user_id: int):
        path = f"movie_recommender/cache/users/{user_id}"
        os.makedirs(path, exist_ok=True)
        return path

    def _save_local_user_cache(self):
        if self.current_user_id is None:
            return
        path = self._user_cache_dir(self.current_user_id)
        if self._user_profile is not None:
            np.save(os.path.join(path, "profile.npy"), self._user_profile)
        if self._user_memory:
            memory_stack = np.stack(self._user_memory)
            np.save(os.path.join(path, "memory.npy"), memory_stack)

    def _load_local_user_cache(self):
        if self.current_user_id is None:
            return
        path = self._user_cache_dir(self.current_user_id)
        try:
            profile_path = os.path.join(path, "profile.npy")
            memory_path = os.path.join(path, "memory.npy")
            if os.path.exists(profile_path):
                self._user_profile = np.load(profile_path)
            if os.path.exists(memory_path):
                memory_stack = np.load(memory_path)
                self._user_memory = [vec.reshape(1, -1) for vec in memory_stack]
        except Exception as e:
            print(f"[CACHE] Failed to load user cache: {e}")
    
    def _get_user_state_key(self):
        return str(sorted(self._genre_preferences.items()))
    
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
        
        # Create cache directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        df.to_pickle(cache_file)
        return df.reset_index(drop=True)

    def _compute_embeddings(self, texts):
        embed_path = "movie_recommender/cache/embeddings.npy"
        if os.path.exists(embed_path):
            print("[CACHE] Loading cached embeddings...")
            return np.load(embed_path)
        
        print("[EMBEDDING] Generating new embeddings...")
        # Normalize embeddings for cosine similarity with dot product
        embeddings = normalize(self.model.encode(texts.tolist(), show_progress_bar=True))
        np.save(embed_path, embeddings)
        return embeddings

    def _build_faiss_index(self, embeddings):
        index_path = "movie_recommender/cache/faiss.index"
        if os.path.exists(index_path):
            print("[CACHE] Loading cached FAISS index...")
            return faiss.read_index(index_path)
        
        print("[FAISS] Creating new FAISS index...")
        # Use IndexFlatIP for Inner Product (cosine similarity after normalization)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32)) # FAISS expects float32
        faiss.write_index(index, index_path)
        return index

    # --- Methods for managing user state with a SQLAlchemy Core Connection ---

    def load_user_state(self, conn: Connection, user_id: int):
        """
        Loads the user's profile, memory, and genre preferences from the database
        into the recommender's in-memory state. Falls back to local cache if DB is empty.
        """
        self.current_user_id = user_id
        cache_dir = self._user_cache_dir(user_id)

        # Load User Profile
        stmt = select(user_profiles_table.c.profile_vector).where(user_profiles_table.c.user_id == user_id)
        result = conn.execute(stmt).scalar_one_or_none()
        if result is not None:
            self._user_profile = np.array(result).reshape(1, -1)
            print(f"[USER {user_id}] Loaded profile from DB.")
        elif os.path.exists(f"{cache_dir}/profile.npy"):
            self._user_profile = np.load(f"{cache_dir}/profile.npy").reshape(1, -1)
            print(f"[USER {user_id}] Loaded profile from local cache.")
        else:
            self._user_profile = None
            print(f"[USER {user_id}] No profile found in DB or cache, starting fresh.")

        # Load User Memory
        stmt = select(user_memory_entries_table.c.movie_embedding)\
            .where(user_memory_entries_table.c.user_id == user_id)\
            .order_by(user_memory_entries_table.c.timestamp.desc())\
            .limit(10)
        memory_results = conn.execute(stmt).fetchall()
        if memory_results:
            self._user_memory = [np.array(entry[0]).reshape(1, -1) for entry in memory_results]
            print(f"[USER {user_id}] Loaded {len(self._user_memory)} memory entries from DB.")
        elif os.path.exists(f"{cache_dir}/memory.npy"):
            self._user_memory = list(np.load(f"{cache_dir}/memory.npy", allow_pickle=True))
            print(f"[USER {user_id}] Loaded memory from local cache ({len(self._user_memory)} entries).")
        else:
            self._user_memory = []
            print(f"[USER {user_id}] No memory found in DB or cache.")

        # Load User Genre Preferences
        stmt = select(user_genre_preferences_table.c.preferences_data).where(user_genre_preferences_table.c.user_id == user_id)
        genre_prefs_data = conn.execute(stmt).scalar_one_or_none()
        if genre_prefs_data is not None:
            self._genre_preferences = defaultdict(float, genre_prefs_data)
            print(f"[USER {user_id}] Loaded genre preferences from DB.")
        else:
            self._genre_preferences = defaultdict(float)
            print(f"[USER {user_id}] No genre preferences found in DB, starting fresh.")


    def add_external_movie(self, movie_dict: dict):
        """
        Adds an external movie (from API) into the system temporarily for feedback tracking.
        """
        # You can optionally enrich this if you store external movies
        title = movie_dict["title"]
        vector = self.model.encode([title])
        self.embeddings = np.vstack([self.embeddings, normalize(vector)])
        
        # Append to movies DataFrame
        new_row = {
            "movie_id": movie_dict["id"],
            "title": movie_dict["title"],
            "genres": [],
            "keywords": [],
            "cast": [],
            "crew": "",
            "tags": movie_dict["title"].lower()
        }
        self.movies.loc[len(self.movies)] = new_row
        self.index.add(normalize(vector).astype(np.float32))  # Update FAISS
        self.rl_agent.actions.append(len(self.movies) - 1)  # Add to RL action space

    def save_user_state(self, conn: Connection):
        """
        Saves the current in-memory user state (profile, memory, genre preferences)
        to the database for the current_user_id.
        This method also commits the transaction.
        """
        if self.current_user_id is None:
            print("[USER] No current user to save state for.")
            return

        # Save User Profile (UPSERT logic: Update if exists, Insert if not)
        profile_data = self._user_profile.flatten().tolist() if self._user_profile is not None else None
        
        if profile_data is not None:
            # Check if profile exists for this user
            existing_profile_stmt = select(user_profiles_table.c.id).where(user_profiles_table.c.user_id == self.current_user_id)
            existing_profile_id = conn.execute(existing_profile_stmt).scalar_one_or_none()

            if existing_profile_id:
                # Update existing profile
                stmt = update(user_profiles_table).where(user_profiles_table.c.user_id == self.current_user_id).values(profile_vector=profile_data)
                conn.execute(stmt)
                print(f"[USER {self.current_user_id}] Profile updated.")
            else:
                # Insert new profile
                stmt = insert(user_profiles_table).values(user_id=self.current_user_id, profile_vector=profile_data)
                conn.execute(stmt)
                print(f"[USER {self.current_user_id}] Profile inserted.")
        else: # If _user_profile is None, delete it from DB if it exists (e.g., after a reset)
            stmt = delete(user_profiles_table).where(user_profiles_table.c.user_id == self.current_user_id)
            conn.execute(stmt)
            print(f"[USER {self.current_user_id}] Profile deleted (reset).")


        # Save User Memory (Delete all existing entries and insert new top 10)
        # It's cleaner to delete and re-insert for memory, given it's a small, rotating list.
        delete_stmt = delete(user_memory_entries_table).where(user_memory_entries_table.c.user_id == self.current_user_id)
        conn.execute(delete_stmt)
        
        memory_entries_to_insert = []
        for vec_np in self._user_memory:
            # Try to find the movie title from the main movie DataFrame based on embedding.
            # This is a best-effort, not strictly necessary for memory functionality but good for DB readability.
            movie_title_for_memory = "Unknown Title"
            # Find the closest movie title for the memory entry for logging/debugging purposes
            if not self.movies.empty and vec_np.shape == (1, self.embeddings.shape[1]):
                # Reshape to 1D for comparison, then find closest index
                idx = np.argmin(np.linalg.norm(self.embeddings - vec_np, axis=1))
                if idx < len(self.movies):
                    movie_title_for_memory = self.movies.iloc[idx].title

            memory_entries_to_insert.append({
                "user_id": self.current_user_id,
                "movie_title": movie_title_for_memory,
                "movie_embedding": vec_np.flatten().tolist(), # Convert NumPy array to list for DB storage
                "timestamp": datetime.now(timezone.utc) # Use timezone-aware datetime
            })
        
        if memory_entries_to_insert:
            insert_stmt = insert(user_memory_entries_table).values(memory_entries_to_insert)
            conn.execute(insert_stmt)
            print(f"[USER {self.current_user_id}] Memory entries saved/updated ({len(memory_entries_to_insert)}).")
        else:
            print(f"[USER {self.current_user_id}] No memory entries to save.")


        # Save User Genre Preferences (UPSERT logic)
        genre_prefs_data = dict(self._genre_preferences) # Convert defaultdict to regular dict for JSONB storage
        
        # Check if genre preferences exist for this user
        existing_genre_prefs_stmt = select(user_genre_preferences_table.c.id).where(user_genre_preferences_table.c.user_id == self.current_user_id)
        existing_genre_prefs_id = conn.execute(existing_genre_prefs_stmt).scalar_one_or_none()

        if existing_genre_prefs_id:
            # Update existing genre preferences
            stmt = update(user_genre_preferences_table).where(user_genre_preferences_table.c.user_id == self.current_user_id).values(preferences_data=genre_prefs_data)
            conn.execute(stmt)
            print(f"[USER {self.current_user_id}] Genre preferences updated.")
        else:
            # Insert new genre preferences
            stmt = insert(user_genre_preferences_table).values(user_id=self.current_user_id, preferences_data=genre_prefs_data)
            conn.execute(stmt)
            print(f"[USER {self.current_user_id}] Genre preferences inserted.")

        # Crucial: Commit the transaction after all DML operations for this request
        conn.commit() 
        print(f"[USER {self.current_user_id}] All state changes committed to DB.")
        self._save_local_user_cache()

    def _update_user_memory(self, vector: np.ndarray):
        """Internal method to update the in-memory user memory."""
        self._user_memory.insert(0, vector)
        self._user_memory = self._user_memory[:10] # Keep only the 10 most recent entries

    def _update_user_profile(self, vector: np.ndarray, alpha=0.3):
        """Internal method to update the in-memory user profile using exponential moving average."""
        if self._user_profile is None:
            self._user_profile = vector.copy()
        else:
            updated = (1 - alpha) * self._user_profile + alpha * vector
            self._user_profile = normalize(updated)[0].reshape(1, -1) # Re-normalize after update

    # Public methods now explicitly take 'conn: Connection' as the first argument after self
    # and call save_user_state(conn) after state modification.

    def recommend(self, conn: Connection, title: str, top_k: int = 5):
        """
        Recommends movies based on a given title, updates user memory and profile,
        and saves the state to the database.
        """
        if title not in self.movies['title'].values:
            raise ValueError(f"'{title}' not found in dataset.")
        
        idx = self.movies[self.movies['title'] == title].index[0]
        vector = self.embeddings[idx].reshape(1, -1)
        
        self._update_user_memory(vector)
        self._update_user_profile(vector)
        self.save_user_state(conn) # Save state after updating memory/profile

        # Perform FAISS search for similar movies
        # top_k + 1 to exclude the movie itself if it's the top result
        _, indices = self.index.search(vector.astype(np.float32), top_k + 1)
        
        # Filter out the input movie itself and return top_k titles
        return [self.movies.iloc[i].title for i in indices[0] if i != idx][:top_k]

    def recommend_from_memory(self, conn: Connection, top_k: int = 5):
        """
        Recommends movies based on the user's recent memory.
        Does NOT modify user state, so no save_user_state() call here.
        """
        if not self._user_memory:
            raise ValueError(f"No recent memory available for user {self.current_user_id}.")
        
        # Calculate average vector from memory
        avg_vec = normalize(np.mean(self._user_memory, axis=0)).reshape(1, -1)
        
        # Search for similar movies using the average memory vector
        _, indices = self.index.search(avg_vec.astype(np.float32), top_k * 2)  # Search extra results to ensure top_k unique
        
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

    def recommend_from_profile(self, conn: Connection, top_k: int = 5, explain=False):
        """
        Recommends movies based on the user's long-term profile.
        Optionally provides an explanation based on genre preferences.
        Does NOT modify user state.
        """
        if self._user_profile is None:
            raise ValueError(f"No user profile available for user {self.current_user_id}.")
        
        # Search for more than top_k to account for filtering and deduplication
        _, indices = self.index.search(self._user_profile.astype(np.float32), 50)

        results = []
        seen_titles = set()
        for i in indices[0]:
            movie = self.movies.iloc[i]
            title = movie.title

            if title in seen_titles:
                continue
            seen_titles.add(title)

            genres = movie.genres
            genre_score = sum([self._genre_preferences.get(g, 0) for g in genres])

            if explain:
                results.append({
                    "title": title,
                    "genres": genres,
                    "genre_score": round(genre_score, 2)
                })
            else:
                results.append({
                    "id": movie.movie_id,
                    "title": title,
                    "reason": [
                        f"Genres matched: {', '.join(genres)}" if genres else "Similar to your profile vector"
                    ]
                })

            if len(results) >= top_k:
                break

        if explain:
            sorted_results = sorted(results, key=lambda x: x["genre_score"], reverse=True)[:top_k]
            for r in sorted_results:
                print(f"[EXPLAIN] {r['title']}: genre_score={r['genre_score']} from genres={r['genres']}")
            return [r['title'] for r in sorted_results]
        else:
            return results



    def reset_profile(self, conn: Connection):
        """
        Resets the user's in-memory profile, memory, and genre preferences,
        then saves this reset state to the database.
        """
        self._user_profile = None
        self._user_memory = []
        self._genre_preferences = defaultdict(float)
        self.save_user_state(conn) # Save the reset state to DB

    def give_feedback(self, conn: Connection, title: str, score: float):
        """
        Processes user feedback for a movie, updating the user's profile,
        genre preferences, and reinforcement agent, then saving to DB.
        """
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
                self._genre_preferences[genre] += score
        else:
            print(f"[FEEDBACK] 👎 Disliked '{title}' with score {score}")
            for genre in genres:
                self._genre_preferences[genre] -= abs(score)

        # ➕ Reinforcement Learning integration
        try:
            state_key = self._get_user_state_key()
            action = idx
            reward = float(score)
            
            # Temporary profile update for next state
            self._update_user_profile(vector, alpha=alpha)
            next_state_key = self._get_user_state_key()

            # Restore profile back (optional depending on your logic)
            # Or you can clone state before modifying
            self.rl_agent.update(state_key, action, reward, next_state_key)
            self.rl_agent.decay_epsilon()
            print(f"[RL] Updated Q-table for action={action} with reward={reward}")
        except Exception as e:
            print(f"[RL] ❌ Failed to update Q-learning agent: {e}")

        self.save_user_state(conn)

    def decay_preferences(self, conn: Connection, decay_rate=0.01):
        """
        Applies a decay factor to the user's genre preferences,
        then saves the updated state to the database.
        """
        # Iterate over a copy of keys, as dictionary might change size during iteration
        for g in list(self._genre_preferences.keys()):
            self._genre_preferences[g] *= (1 - decay_rate)
            # Remove genres with very low preference to keep the dict clean
            if abs(self._genre_preferences[g]) < 0.01:
                del self._genre_preferences[g]
        self.save_user_state(conn) # Save the decayed state to DB


    def recommend_from_rl(self, top_k=5):
        """
        Recommend movies based on the Q-learning policy.
        It uses current genre preferences as the state.
        """
        state_key = self._get_user_state_key()

        # Rank actions (movie indices) by Q-value for current state
        q_values = [(a, self.rl_agent.q_table[(state_key, a)]) for a in self.rl_agent.actions]
        ranked = sorted(q_values, key=lambda x: x[1], reverse=True)

        recommendations = []
        seen_titles = set()
        for action_idx, q in ranked:
            movie = self.movies.iloc[action_idx]
            title = movie.title

            if title in seen_titles:
                continue  # Avoid duplicates
            seen_titles.add(title)

            genres = movie.genres
            genre_score = sum([self._genre_preferences.get(g, 0) for g in genres])

            # Optional genre filter (disable this if too strict)
            if genre_score <= -2:
                continue

            recommendations.append({
                "id": movie.movie_id,
                "title": title,
                "reason": [
                    f"Recommended based on Q-learning (score={round(q, 2)})",
                    f"Genres: {genres}"
                ]
            })

            if len(recommendations) >= top_k:
                break

        print(f"[RL] Final recommendations: {[r['title'] for r in recommendations]}")
        return recommendations


