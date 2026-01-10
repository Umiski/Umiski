import os

from dotenv import load_dotenv


def get_config():
    load_dotenv()
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "chroma_path": "chroma_db",
        "data_path": "data",
    }
