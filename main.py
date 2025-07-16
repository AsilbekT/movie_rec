# main.py (Full Updated Version with External API Integration, Feedback, Reset, Decay)
import os
import sys
import difflib
import requests
import numpy as np
from typing import Annotated

from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import Connection, select, insert

# Add the parent directory to sys.path to import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from movie_recommender.core import SmartMovieRecommender
from movie_recommender.config import MOVIES_CSV, CREDITS_CSV
from database import (
    get_db_connection,
    create_db_tables,
    users_table
)

# --- Constants ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'movie_recommender', 'models', 'sentence-transformers--paraphrase-MiniLM-L6-v2')
EXTERNAL_MOVIE_API = "https://gateway.kinoplus.uz/catalogservice/movies/"

# --- FastAPI App ---
app = FastAPI(
    title="Smart Movie Recommender API (PostgreSQL Backend)",
    description="API for personalized movie recommendations with persistent user data in PostgreSQL.",
    version="1.0.0",
)

origins = ["http://localhost:4200", "http://127.0.0.1:4200"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender: SmartMovieRecommender = None

@app.on_event("startup")
async def startup_event():
    global recommender
    try:
        recommender = SmartMovieRecommender(MOVIES_CSV, CREDITS_CSV, model_path=MODEL_PATH)
        print("FastAPI: SmartMovieRecommender loaded successfully.")
        create_db_tables()
        print("FastAPI: Database tables checked/created.")
    except Exception as e:
        print(f"FastAPI: Error during startup: {e}")
        raise RuntimeError(f"Startup failure: {e}")

# --- External Movie Fetching & Matching ---
def fetch_external_movies():
    try:
        response = requests.get(EXTERNAL_MOVIE_API, timeout=5)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        print(f"Error fetching external movies: {e}")
        return []

def match_external_movie(title: str, external_movies: list[dict]) -> dict | None:
    titles = [movie["title"] for movie in external_movies]
    closest = difflib.get_close_matches(title, titles, n=1)
    if closest:
        match = next((m for m in external_movies if m["title"] == closest[0]), None)
        if match:
            return {"id": match["id"], "title": match["title"]}
    return None

# --- Utility ---
def find_movie_index_by_title(recommender, title):
    titles = recommender.movies['title'].tolist()
    closest = difflib.get_close_matches(title, titles, n=1)
    if closest:
        idx = recommender.movies[recommender.movies['title'] == closest[0]].index[0]
        return idx, closest[0]
    return None, None

# --- Pydantic Models ---
class UserRecommendationQuery(BaseModel):
    username: str
    top_k: int = 5

class FeedbackRequest(BaseModel):
    username: str
    title: str
    score: float  # e.g., 1.0 = like, -1.0 = dislike

class RecommendationResponse(BaseModel):
    recommendations: list[dict]  # each with id + title

# --- Dependency Injection ---
def get_user_query(request: UserRecommendationQuery = Body(...)) -> UserRecommendationQuery:
    return request

async def get_current_user_id_dependency(
    request: UserRecommendationQuery = Depends(get_user_query),
    conn: Connection = Depends(get_db_connection)
) -> int:
    global recommender
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender not initialized.")

    stmt = select(users_table.c.id).where(users_table.c.username == request.username)
    user_id = conn.execute(stmt).scalar_one_or_none()
    if user_id is None:
        user_id = conn.execute(
            insert(users_table).values(username=request.username).returning(users_table.c.id)
        ).scalar_one()
        conn.commit()
    recommender.load_user_state(conn, user_id)
    return user_id

# --- Endpoints ---
@app.post("/feedback")
async def give_feedback(
    payload: FeedbackRequest,
    conn: Connection = Depends(get_db_connection)
):
    global recommender

    # 1. Ensure user exists
    stmt = select(users_table.c.id).where(users_table.c.username == payload.username)
    user_id = conn.execute(stmt).scalar_one_or_none()
    if user_id is None:
        user_id = conn.execute(
            insert(users_table).values(username=payload.username).returning(users_table.c.id)
        ).scalar_one()
        conn.commit()

    recommender.load_user_state(conn, user_id)

    # 2. Ensure movie exists in local dataset
    idx, matched_title = find_movie_index_by_title(recommender, payload.title)
    if idx is None:
        external_movies = fetch_external_movies()
        match = match_external_movie(payload.title, external_movies)
        if match:
            # Add to recommender.movies
            recommender.movies.loc[len(recommender.movies)] = {
                "movie_id": match["id"],
                "title": match["title"],
                "overview": match["title"],
                "genres": [],
                "keywords": "",
                "cast": "",
                "crew": "",
                "tags": match["title"].lower()
            }
            # Regenerate embeddings and reset index
            recommender.movies.reset_index(drop=True, inplace=True)
            recommender.embeddings = recommender._compute_embeddings(recommender.movies["tags"])
            recommender.index = recommender._build_faiss_index(recommender.embeddings)
            matched_title = match["title"]
        else:
            raise HTTPException(status_code=404, detail="Movie not found in dataset or external API.")

    # 3. Give feedback
    recommender.give_feedback(conn, matched_title, payload.score)

    return {
        "status": "success",
        "message": f"Feedback recorded for '{matched_title}' with score {payload.score}"
    }


@app.post("/profile/reset")
async def reset_user_profile(
    payload: UserRecommendationQuery,
    conn: Connection = Depends(get_db_connection)
):
    global recommender
    stmt = select(users_table.c.id).where(users_table.c.username == payload.username)
    user_id = conn.execute(stmt).scalar_one_or_none()
    if user_id is None:
        raise HTTPException(status_code=404, detail="User not found")

    recommender.load_user_state(conn, user_id)
    recommender.reset_profile(conn)
    return {"status": "success", "message": f"Profile reset for {payload.username}"}

@app.post("/profile/decay")
async def decay_user_preferences(
    payload: UserRecommendationQuery,
    conn: Connection = Depends(get_db_connection)
):
    global recommender
    stmt = select(users_table.c.id).where(users_table.c.username == payload.username)
    user_id = conn.execute(stmt).scalar_one_or_none()
    if user_id is None:
        raise HTTPException(status_code=404, detail="User not found")

    recommender.load_user_state(conn, user_id)
    recommender.decay_preferences(conn, decay_rate=0.01)
    return {"status": "success", "message": f"Genre preferences decayed for {payload.username}"}

@app.post("/recommend/from_profile", response_model=RecommendationResponse)
async def get_recommendations_from_profile(
    current_user_id: Annotated[int, Depends(get_current_user_id_dependency)],
    request: UserRecommendationQuery = Depends(get_user_query),
    conn: Connection = Depends(get_db_connection)
):
    if not recommender or recommender.current_user_id != current_user_id:
        raise HTTPException(status_code=503, detail="Recommender context error.")

    try:
        titles = recommender.recommend_from_profile(conn, top_k=request.top_k)
        external_movies = fetch_external_movies()
        matched_results = []

        for title in titles:
            match = match_external_movie(title, external_movies)
            if match:
                matched_results.append(match)
            else:
                matched_results.append({"id": None, "title": title})  # fallback if not matched

        return RecommendationResponse(recommendations=matched_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))