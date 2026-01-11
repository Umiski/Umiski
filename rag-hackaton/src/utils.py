import os

from dotenv import load_dotenv


def get_config():
    load_dotenv()

    # Ścieżka bazowa projektu
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return {
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "chroma_path": os.path.join(BASE_DIR, "data", "chroma_db"),
        "data_path": os.path.join(BASE_DIR, "data"),
        # Modele Google
        "llm_model": os.getenv("LLM_MODEL"),  # Szybki model
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL", "models/text-embedding-004"
        ),  # Embeddingi
        # Parametry RAG
        "chunk_size": int(os.getenv("CHUNK_SIZE", 2000)),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 200)),
        "retrieval_k": int(os.getenv("RETRIEVAL_K", 5)),
        "temperature": float(os.getenv("TEMPERATURE", 0)),
        "judge_model": os.getenv("JUDGE_MODEL"),  # Model sędziego
    }
