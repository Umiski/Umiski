import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # ZMIANA
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import get_config


def run_ingestion():
    config = get_config()
    print("🚀 Rozpoczynam Ingestion (Google Gemini Stack)...")

    if not config["google_api_key"]:
        print("❌ BŁĄD: Brak klucza GOOGLE_API_KEY w pliku .env!")
        return

    # 1. Sprawdzenie folderów
    if not os.path.exists(config["data_path"]):
        os.makedirs(config["data_path"])
        print(f"Stworzono folder {config['data_path']}. Wrzuć tam PDFy!")
        return

    # 2. Ładowanie PDFów
    documents = []
    for file in os.listdir(config["data_path"]):
        if file.endswith(".pdf"):
            file_path = os.path.join(config["data_path"], file)
            print(f"📄 Przetwarzam: {file}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file  # Dodaj nazwę pliku jako metadane
            documents.extend(docs)

    if not documents:
        print("❌ Folder data/ jest pusty.")
        return

    # 3. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],  # Konfigurowalne
        chunk_overlap=config["chunk_overlap"],  # Konfigurowalne
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Pocięto na {len(chunks)} fragmentów.")

    # 4. Reset starej bazy (bo zmieniamy OpenAI na Google)
    if os.path.exists(config["chroma_path"]):
        shutil.rmtree(config["chroma_path"])

    # 5. Zapis do ChromaDB
    print(f"☁️  Generowanie wektorów ({config['embedding_model']})...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config["embedding_model"], google_api_key=config["google_api_key"]
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config["chroma_path"],
    )

    print(f"🎉 Sukces! Baza gotowa w: {config['chroma_path']}")


if __name__ == "__main__":
    run_ingestion()
