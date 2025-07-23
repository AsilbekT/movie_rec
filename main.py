# main.py (Fully Corrected and Extended RL Version)
import os
import sys
import difflib
import requests
import numpy as np
from typing import Annotated

from fastapi import FastAPI, HTTPException, Body, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import Connection, select, insert

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from movie_recommender.core import SmartMovieRecommender
from movie_recommender.config import MOVIES_CSV, CREDITS_CSV
from database import (
    get_db_connection,
    create_db_tables,
    users_table
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'movie_recommender', 'models', 'sentence-transformers--paraphrase-MiniLM-L6-v2')
EXTERNAL_MOVIE_API = "https://gateway.kinoplus.uz/catalogservice/movies/"

app = FastAPI(
    title="Smart Movie Recommender API (PostgreSQL Backend)",
    description="API for personalized movie recommendations with persistent user data in PostgreSQL.",
    version="1.0.0",
    docs_url="/recommend/docs",
    redoc_url="/recommend/redoc",
    openapi_url="/recommend/openapi.json",
)

origins = ["http://localhost:4200", "http://127.0.0.1:4200"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

recommender: SmartMovieRecommender = None
Q_TABLE_PATH = os.path.join(os.path.dirname(__file__), 'movie_recommender', 'cache', 'q_table.pkl')

@app.on_event("startup")
async def startup_event():
    global recommender
    try:
        os.makedirs(os.path.dirname(Q_TABLE_PATH), exist_ok=True)
        recommender = SmartMovieRecommender(MOVIES_CSV, CREDITS_CSV, model_path=MODEL_PATH)
        recommender.rl_agent.load(Q_TABLE_PATH)  # ← Load Q-table here
        print("FastAPI: SmartMovieRecommender loaded successfully.")
        create_db_tables()
        print("FastAPI: Database tables checked/created.")
    except Exception as e:
        print(f"FastAPI: Error during startup: {e}")
        raise RuntimeError(f"Startup failure: {e}")


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

def find_movie_index_by_title(recommender, title):
    titles = recommender.movies['title'].tolist()
    closest = difflib.get_close_matches(title, titles, n=1)
    if closest:
        idx = recommender.movies[recommender.movies['title'] == closest[0]].index[0]
        return idx, closest[0]
    return None, None

class UserRecommendationQuery(BaseModel):
    username: str
    top_k: int = 5
    explore: bool = False

class FeedbackRequest(BaseModel):
    username: str
    title: str
    score: float

class UserProfileResponse(BaseModel):
    top_genres: list[str]  # sorted by preference score
    genre_scores: dict[str, float]  # full preference map
    feedback_summary: dict[str, int]  # e.g., {"liked": 12, "disliked": 5}
    liked_directors: list[str] = []  # optional; empty for now or future use

class RecommendationItem(BaseModel):
    id: int
    title: str
    reason: list[str]

class RecommendationResponse(BaseModel):
    user_profile: UserProfileResponse
    recommendations: list[RecommendationItem]


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

@router.post("/feedback")
async def give_feedback(
    payload: FeedbackRequest,
    conn: Connection = Depends(get_db_connection)
):
    global recommender

    stmt = select(users_table.c.id).where(users_table.c.username == payload.username)
    user_id = conn.execute(stmt).scalar_one_or_none()
    if user_id is None:
        user_id = conn.execute(
            insert(users_table).values(username=payload.username).returning(users_table.c.id)
        ).scalar_one()
        conn.commit()

    recommender.load_user_state(conn, user_id)

    idx, matched_title = find_movie_index_by_title(recommender, payload.title)
    if idx is None:
        external_movies = fetch_external_movies()
        match = match_external_movie(payload.title, external_movies)
        if match:
            recommender.add_external_movie(match)
            matched_title = match["title"]
        else:
            raise HTTPException(status_code=404, detail="Movie not found in dataset or external API.")

    recommender.give_feedback(conn, matched_title, payload.score)
    recommender.rl_agent.save(Q_TABLE_PATH) 

    return {
        "status": "success",
        "message": f"Feedback recorded for '{matched_title}' with score {payload.score}"
    }

@router.post("/profile/reset")
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

@router.post("/profile/decay")
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

@router.post("/from_profile", response_model=RecommendationResponse)
async def get_recommendations_from_profile(
    current_user_id: Annotated[int, Depends(get_current_user_id_dependency)],
    request: UserRecommendationQuery = Depends(get_user_query),
    conn: Connection = Depends(get_db_connection)
):
    if not recommender or recommender.current_user_id != current_user_id:
        raise HTTPException(status_code=503, detail="Recommender context error.")

    try:
        titles = recommender.recommend_from_profile(conn, top_k=request.top_k, explain=True)
        external_movies = fetch_external_movies()
        matched_results = []

        for title in titles:
            match = match_external_movie(title, external_movies)
            if not match:
                match = {"id": None, "title": title}
            matched_results.append(match)

        recommendation_items = []
        for match in matched_results:
            if match.get("id") is None:
                continue
            idx = recommender.movies[recommender.movies['title'] == match["title"]].index
            if not idx.empty:
                movie = recommender.movies.loc[idx[0]]
                vector = recommender.embeddings[idx[0]].reshape(1, -1)
                reasons = explain_reason(movie, vector, recommender, q_value=None)
            else:
                reasons = ["Recommended based on your overall profile"]

            recommendation_items.append(RecommendationItem(
                id=match["id"], title=match["title"], reason=reasons
            ))

        return RecommendationResponse(
            user_profile=summarize_user_profile(recommender),
            recommendations=recommendation_items
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def explain_reason(movie, movie_vector, recommender, q_value):
    reasons = []

    # Q-learning signal
    if q_value is not None and q_value > 0.5:
        reasons.append("Learned from your positive feedback on similar movies")

    # Genre preferences
    genres = movie.genres
    liked_genres = [
        g for g in genres if recommender._genre_preferences.get(g, 0) > 0.5
    ]
    if liked_genres:
        reasons.append(f"Matches your favorite genres: {', '.join(liked_genres)}")

    # Similarity to profile
    if recommender._user_profile is not None:
        similarity = float(np.dot(recommender._user_profile, movie_vector.T))
        if similarity > 0.7:
            reasons.append("Very similar to your preferred movie style")

    # Past director match
    director = movie.crew
    if director and isinstance(director, str):
        for mem in recommender._user_memory:
            idx = np.argmin(np.linalg.norm(recommender.embeddings - mem, axis=1))
            past_director = recommender.movies.iloc[idx].crew
            if past_director == director:
                reasons.append(f"Directed by {director}, whose work you’ve watched")

    # Novelty
    title = movie.title
    recently_seen = any(
        np.allclose(recommender.embeddings[i], movie_vector, atol=0.01)
        for i in range(len(recommender.movies))
        if recommender.movies.iloc[i].title == title
    )
    if not recently_seen:
        reasons.append("You haven’t seen it recently")

    return reasons or ["Selected based on your overall taste"]

def summarize_user_profile(recommender):
    genre_scores = dict(recommender._genre_preferences)

    # Top genres sorted by score descending
    top_genres = sorted(
        [g for g in genre_scores if genre_scores[g] > 0],
        key=lambda g: genre_scores[g],
        reverse=True
    )

    liked_directors = set()
    for vec in recommender._user_memory:
        idx = np.argmin(np.linalg.norm(recommender.embeddings - vec, axis=1))
        director = recommender.movies.iloc[idx].crew
        if director and isinstance(director, str):
            liked_directors.add(director)

    return {
        "top_genres": top_genres,
        "genre_scores": genre_scores,
        "liked_directors": sorted(liked_directors),
        "feedback_summary": {
            "total_seen": len(recommender._user_memory),
            "positives": sum(1 for g in genre_scores if genre_scores[g] > 0.5),
            "negatives": sum(1 for g in genre_scores if genre_scores[g] < 0)
        }
    }



@app.post("/recommend/rl", response_model=RecommendationResponse)
async def get_recommendation_from_rl(
    current_user_id: Annotated[int, Depends(get_current_user_id_dependency)],
    request: UserRecommendationQuery = Depends(get_user_query),
    conn: Connection = Depends(get_db_connection)
):
    global recommender

    if not recommender or recommender.current_user_id != current_user_id:
        raise HTTPException(status_code=503, detail="Recommender context error.")

    try:
        state_key = recommender._get_user_state_key()
        q_values = [
            (a, recommender.rl_agent.q_table.get((state_key, a), 0.0))
            for a in recommender.rl_agent.actions
        ]

        if request.explore and np.random.rand() < recommender.rl_agent.epsilon:
            print("[Exploration] Choosing random unseen actions due to epsilon.")
            # Shuffle all unseen actions
            unseen_actions = [
                a for a in recommender.rl_agent.actions
                if recommender.movies.iloc[a].title not in {
                    m.title for vec in recommender._user_memory
                    for i in range(len(recommender.movies))
                    if np.allclose(recommender.embeddings[i], vec, atol=0.01)
                    for m in [recommender.movies.iloc[i]]
                }
            ]
            np.random.shuffle(unseen_actions)
            ranked = [(a, recommender.rl_agent.q_table.get((state_key, a), 0.0)) for a in unseen_actions]
        else:
            # Exploitation (deterministic greedy sort)
            ranked = sorted(q_values, key=lambda x: x[1], reverse=True)

        seen_titles = set(
            m.title for vec in recommender._user_memory
            for i in range(len(recommender.movies))
            if np.allclose(recommender.embeddings[i], vec, atol=0.01)
            for m in [recommender.movies.iloc[i]]
        )

        recommendations = []
        for action_idx, q in ranked:
            movie = recommender.movies.iloc[action_idx]
            if movie.title in seen_titles:
                continue

            vector = recommender.embeddings[action_idx].reshape(1, -1)

            # Generate intelligent explanation
            reasons = explain_reason(movie, vector, recommender, q_value=q)

            # Update RL and user profile
            recommender._update_user_memory(vector)
            recommender._update_user_profile(vector)
            recommender.rl_agent.update(
                state_key=state_key,
                action=action_idx,
                reward=1.0,
                next_state_key=recommender._get_user_state_key()
            )
            recommender.rl_agent.decay_epsilon()

            recommendations.append({
                "id": int(movie.movie_id),
                "title": movie.title,
                "reason": reasons
            })

            if len(recommendations) >= request.top_k:
                break

        recommender.save_user_state(conn)
        recommender.rl_agent.save(Q_TABLE_PATH)

        return {
            "user_profile": summarize_user_profile(recommender),
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RL recommendation failed: {str(e)}")


@router.post("/rl/reset")
async def reset_q_table():
    global recommender
    recommender.rl_agent.reset()
    if os.path.exists(Q_TABLE_PATH):
        os.remove(Q_TABLE_PATH)
    return {"status": "success", "message": "Q-table reset and file deleted."}

app.include_router(router, prefix="/recommend")
