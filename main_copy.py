import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.dropna(inplace=True)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L[:5]

def get_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return i['name']
    return ''

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
movies['crew'] = movies['crew'].apply(get_director)

movies['tags'] = movies['overview'] + movies['genres'].apply(lambda x: ' '.join(x)) + \
                 movies['keywords'].apply(lambda x: ' '.join(x)) + \
                 movies['cast'].apply(lambda x: ' '.join(x)) + \
                 movies['crew']

movies['tags'] = movies['tags'].apply(lambda x: x.lower())

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie_title):
    if movie_title not in movies['title'].values:
        print("Movie not found.")
        return
    index = movies[movies['title'] == movie_title].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    
    print(f"Top 5 recommendations for '{movie_title}':")
    for i in movies_list:
        print(movies.iloc[i[0]].title)

recommend('Avatar')
