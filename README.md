# 🎬 Movie Recommender System

This is a content-based movie recommendation system built using Python and the TMDB 5000 Movie Dataset. It recommends similar movies based on plot overview, cast, crew, genres, and keywords.

## 📁 Project Structure

```
movie_recommender/
├── core.py # Main recommender class
├── utils.py # Helper functions
├── config.py # Dataset paths
├── data/
│ ├── tmdb_5000_movies.csv
│ └── tmdb_5000_credits.csv
scripts/
├── run_recommender.py # Example script to run the recommender
tests/
├── test_core.py # Unit tests
setup.py
requirements.txt
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Add the dataset
Download the dataset from Kaggle
Place tmdb_5000_movies.csv and tmdb_5000_credits.csv into the movie_recommender/data/ folder.
## 4. Run the recommender
```
python scripts/run_recommender.py
```

## 🔍 Example Output
```
Top recommendations for 'Avatar':
• John Carter
• Guardians of the Galaxy
• Star Trek Into Darkness
• Star Wars: The Force Awakens
• Prometheus
```

How It Works
- Extracts relevant tags from:

- - Overview

- - Genres

- - Keywords

- - Cast

- - Director

- Combines them into a text representation

- Vectorizes using CountVectorizer

- Computes similarity using cosine similarity


## License
MIT License © 2025 AI Project
