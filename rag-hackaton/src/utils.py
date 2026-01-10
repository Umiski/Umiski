import os

from dotenv import load_dotenv


def get_config():
    load_dotenv()
    return {
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "chroma_path": "chroma_db",
        "data_path": "data",
    }
