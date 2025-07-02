import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

def get_local_model_path(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2", local_dir="movie_recommender/models"):
    model_id = model_name.replace("/", "--")
    local_path = os.path.join(local_dir, model_id)

    if not os.path.exists(local_path):
        print(f"[INFO] Downloading model '{model_name}' to local path '{local_path}'...")
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(repo_id=model_name, local_dir=local_path, local_dir_use_symlinks=False)
        print("[INFO] Download complete.")
    else:
        print(f"[INFO] Model already exists locally at '{local_path}'.")

    return local_path

# Usage in your script
if __name__ == "__main__":
    local_model_path = get_local_model_path()
    model = SentenceTransformer(local_model_path)
    print("[READY] Model loaded from local path.")
