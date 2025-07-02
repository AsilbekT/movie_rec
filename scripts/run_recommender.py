import os
from movie_recommender.core import SmartMovieRecommender
from movie_recommender.config import MOVIES_CSV, CREDITS_CSV

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '../movie_recommender/models/sentence-transformers--paraphrase-MiniLM-L6-v2'
    )
)

recommender = SmartMovieRecommender(MOVIES_CSV, CREDITS_CSV, model_path=MODEL_PATH)

# Simulated feedback (title, score from 0 to 1)
feedbacks = [
    ("Avatar", 1.0),
    ("Iron Man", 0.9),
    ("Twilight", 0.1),
    ("The Dark Knight", 0.95),
    ("Batman & Robin", 0.3),
]

# Apply feedback loop
for title, score in feedbacks:
    print(f"\n>>> User watched '{title}' and rated it {score}")
    try:
        recommendations = recommender.recommend(title)
        print(f"Recommended after '{title}': {recommendations}")
        recommender.give_feedback(title, score=score)
    except ValueError as e:
        print(f"[ERROR] {e}")

# Show memory-based suggestions
print("\n>>> Memory-based Recommendations:")
try:
    print(recommender.recommend_from_memory())
except ValueError as e:
    print(f"[ERROR] {e}")

# Show profile-based suggestions with explanation
print("\n>>> Profile-based Recommendations (with explanation):")
try:
    print(recommender.recommend_from_profile(explain=True))
except ValueError as e:
    print(f"[ERROR] {e}")

# Save user state
recommender.save_user_state()
print("\n✅ User session saved.")
