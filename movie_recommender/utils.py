import ast


def extract_names(obj: str):
    """Extracts up to 5 names from a JSON-like string field."""
    try:
        return [item['name'] for item in ast.literal_eval(obj)][:5]
    except Exception:
        return []


def extract_director(obj: str):
    """Extracts the director's name from the crew JSON string."""
    try:
        for d in ast.literal_eval(obj):
            if d.get('job') == 'Director':
                return d.get('name', '')
    except Exception:
        pass
    return ''
