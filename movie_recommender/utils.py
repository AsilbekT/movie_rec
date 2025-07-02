# movie_recommender/utils.py

import ast

def extract_names(obj: str):
    try:
        return [item['name'] for item in ast.literal_eval(obj)][:5]
    except Exception:
        return []

def extract_director(obj: str):
    try:
        for d in ast.literal_eval(obj):
            if d.get('job') == 'Director':
                return d.get('name', '')
    except Exception:
        return ''
