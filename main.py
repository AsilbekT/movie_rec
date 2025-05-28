from recommender import MovieRecommender

if __name__ == '__main__':
    recommender = MovieRecommender(
        movies_path='data/tmdb_5000_movies.csv',
        credits_path='data/tmdb_5000_credits.csv'
    )

    movie_name = 'Inception'
    try:
        recommendations = recommender.recommend(movie_name)
        print(f"Top recommendations for '{movie_name}':")
        for i, title in enumerate(recommendations, 1):
            print(f"{i}. {title}")
    except ValueError as e:
        print(e)