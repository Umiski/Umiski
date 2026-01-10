import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from utils import get_config


def run_ingestion():
    config = get_config()
    print("📂 KROK 1: Szukam plików w folderze data...")

    documents = []
    # Upewnijmy się, że ścieżka jest absolutna dla Windowsa
    data_path = os.path.abspath(config["data_path"])

    files = [f for f in os.listdir(data_path) if f.endswith(".pdf")]
    print(f"📄 Znaleziono pliki: {files}")

    for file in files:
        print(f"📖 Ładuję plik: {file}...")
        loader = PyPDFLoader(os.path.join(data_path, file))
        documents.extend(loader.load())

    if not documents:
        print("❌ Folder data/ jest pusty lub nie ma w nim PDFów!")
        return

    print(f"✂️ KROK 2: Dzielę tekst na fragmenty (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Podzielono na {len(chunks)} fragmentów.")

    print(f"🧠 KROK 3: Wysyłam do Ollamy (to może potrwać)...")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding = OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory=config["chroma_path"],
        )
        print(f"🚀 SUKCES! Przetworzono {len(chunks)} fragmentów.")
    except Exception as e:
        print(f"❌ BŁĄD wektoryzacji: {e}")


if __name__ == "__main__":
    run_ingestion()
