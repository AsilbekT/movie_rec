import os
import sys
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import difflib
import requests
from sklearn.preprocessing import normalize

# Add module path manually
sys.path.append("/Users/asilbekturgunboev/Desktop/movie_rec")  # Adjust if needed

from database import engine, users_table, user_profiles_table, user_memory_entries_table, user_genre_preferences_table
from movie_recommender.core import SmartMovieRecommender
from movie_recommender.config import MOVIES_CSV, CREDITS_CSV
from sqlalchemy import select, insert, delete

MODEL_PATH = "/Users/asilbekturgunboev/Desktop/movie_rec/movie_recommender/models/sentence-transformers--paraphrase-MiniLM-L6-v2"
EXTERNAL_MOVIE_API = "https://gateway.kinoplus.uz/catalogservice/movies/"

def fetch_external_movies():
    try:
        response = requests.get(EXTERNAL_MOVIE_API, timeout=10)
        response.raise_for_status()
        return response.json()["data"]
    except Exception as e:
        print(f"❌ Failed to fetch from API: {e}")
        return []

def add_movies_to_local_dataset(recommender, external_movies, movie_titles_to_add):
    added = 0
    for title in movie_titles_to_add:
        match = difflib.get_close_matches(title, [m["title"] for m in external_movies], n=1)
        if not match:
            continue
        matched = next((m for m in external_movies if m["title"] == match[0]), None)
        if matched:
            recommender.movies.loc[len(recommender.movies)] = {
                "movie_id": matched["id"],
                "title": matched["title"],
                "overview": matched.get("title", ""),
                "genres": [g["name"] for g in matched.get("genre", [])] if matched.get("genre") else [],
                "keywords": "",
                "cast": "",
                "crew": "",
                "tags": matched.get("title", "").lower()
            }
            added += 1
    if added:
        recommender.movies.reset_index(drop=True, inplace=True)  # 🧠 FIX: reset index before embedding
        recommender.embeddings = recommender._compute_embeddings(recommender.movies['tags'])
        recommender.index = recommender._build_faiss_index(recommender.embeddings)
        print(f"✅ {added} movies added from API and embedded.")


def find_movie_index_by_title(recommender, title):
    all_titles = recommender.movies['title'].tolist()
    closest = difflib.get_close_matches(title, all_titles, n=1)
    if closest:
        idx = recommender.movies[recommender.movies['title'] == closest[0]].index[0]
        return idx, closest[0]
    return None, None

def populate_user_data(username: str = "john_doe"):
    print(f"\n--- Populating data for user '{username}' ---")
    recommender = SmartMovieRecommender(MOVIES_CSV, CREDITS_CSV, model_path=MODEL_PATH)

    with engine.connect() as conn:
        stmt = select(users_table.c.id).where(users_table.c.username == username)
        user_id = conn.execute(stmt).scalar_one_or_none()

        if user_id is None:
            user_id = conn.execute(insert(users_table).values(username=username).returning(users_table.c.id)).scalar_one()
            conn.commit()
            print(f"   Created user ID: {user_id}")
        else:
            print(f"   User ID: {user_id}")

        movie_id_to_title = {
            1: "Sarvinoz",
            2: "Marvel",
            3: "Kalmar o'yini",
            4: "Venom 3: So'nggi raqs",
            5: "O'rgimchak odam: Uyga yo'l yo'q",
            6: "Salaar: 1",
            7: "Maktab",
            8: "Do'zaxdagi notanishlar",
        }

        liked_titles = [movie_id_to_title[5], movie_id_to_title[6], movie_id_to_title[7], movie_id_to_title[8]]
        disliked_titles = [movie_id_to_title[1], movie_id_to_title[2], movie_id_to_title[3], movie_id_to_title[4]]

        external_movies = fetch_external_movies()
        add_movies_to_local_dataset(recommender, external_movies, liked_titles + disliked_titles)

        liked_movie_embeddings, disliked_movie_embeddings = [], []

        print("2. Generating embeddings...")
        for title in liked_titles:
            idx, matched = find_movie_index_by_title(recommender, title)
            if idx is not None:
                liked_movie_embeddings.append(recommender.embeddings[idx].reshape(1, -1))
                print(f"   ✅ Liked: '{title}' -> '{matched}'")
            else:
                print(f"   ⚠️ Liked movie '{title}' not found.")

        for title in disliked_titles:
            idx, matched = find_movie_index_by_title(recommender, title)
            if idx is not None:
                disliked_movie_embeddings.append(recommender.embeddings[idx].reshape(1, -1))
                print(f"   ✅ Disliked: '{title}' -> '{matched}'")
            else:
                print(f"   ⚠️ Disliked movie '{title}' not found.")

        # Clear existing data
        conn.execute(delete(user_profiles_table).where(user_profiles_table.c.user_id == user_id))
        conn.execute(delete(user_memory_entries_table).where(user_memory_entries_table.c.user_id == user_id))
        conn.execute(delete(user_genre_preferences_table).where(user_genre_preferences_table.c.user_id == user_id))
        conn.commit()

        # Profile vector
        if liked_movie_embeddings:
            profile_vector = normalize(np.mean(liked_movie_embeddings, axis=0))[0].tolist()
            conn.execute(insert(user_profiles_table).values(user_id=user_id, profile_vector=profile_vector))
            print("   ✅ Profile vector inserted.")

        # Memory entries (last 3 liked)
        memory_entries = []
        for i, title in enumerate(liked_titles[-3:]):
            idx, matched = find_movie_index_by_title(recommender, title)
            if idx is not None:
                vector = recommender.embeddings[idx].reshape(1, -1)
                memory_entries.append({
                    "user_id": user_id,
                    "movie_title": matched,
                    "movie_embedding": vector.flatten().tolist(),
                    "timestamp": datetime.now(timezone.utc) - timedelta(minutes=i * 5)
                })

        if memory_entries:
            conn.execute(insert(user_memory_entries_table).values(memory_entries))
            print("   ✅ Memory entries inserted.")

        # Genre preferences
        genre_scores = defaultdict(float)
        for title in liked_titles:
            idx, _ = find_movie_index_by_title(recommender, title)
            if idx is not None:
                for genre in recommender.movies.iloc[idx]["genres"]:
                    genre_scores[genre] += 1.0
        for title in disliked_titles:
            idx, _ = find_movie_index_by_title(recommender, title)
            if idx is not None:
                for genre in recommender.movies.iloc[idx]["genres"]:
                    genre_scores[genre] -= 0.5

        conn.execute(insert(user_genre_preferences_table).values(
            user_id=user_id,
            preferences_data=dict(genre_scores)
        ))
        conn.commit()
        print("✅ Genre preferences inserted.")
        print("--- 🎉 DONE: User data populated ---")

if __name__ == "__main__":
    populate_user_data("john_doe")
