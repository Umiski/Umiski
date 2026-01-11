import os
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.utils import get_config


def run_ingestion():
    config = get_config()

    # 1. Przygotowanie ścieżek
    DATA_PATH = "data/"  # Tu wrzuć PDFy od NASA, ESA, UNOOSA
    CHROMA_PATH = config["chroma_path"]

    # Czyszczenie starej bazy, żeby nie dublować danych (ważne przy testach)
    if os.path.exists(CHROMA_PATH):
        print(f"🧹 Usuwanie starej bazy w {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)

    # 2. Ładowanie dokumentów (PDFy)
    print("📂 Ładowanie dokumentów z folderu data/...")
    loader = DirectoryLoader(
        DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )

    raw_documents = loader.load()
    print(f"✅ Załadowano {len(raw_documents)} stron dokumentacji.")

    # 3. Podział tekstu na mniejsze fragmenty (Chunking)
    # Rozmiar 1000 znaków z zakładką 200, żeby nie gubić kontekstu między fragmentami
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        length_function=len,
        add_start_index=True,
    )

    print("✂️ Dzielenie dokumentów na fragmenty...")
    chunks = text_splitter.split_documents(raw_documents)
    print(f"✅ Utworzono {len(chunks)} fragmentów wiedzy.")

    # 4. Inicjalizacja modelu Embeddingów (Google)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=config["google_api_key"]
    )

    # 5. Budowa i zapis bazy wektorowej ChromaDB
    print(f"🚀 Budowanie bazy ChromaDB w {CHROMA_PATH}... (To może chwilę potrwać)")

    # Przetwarzanie w paczkach (batching), aby uniknąć błędów API przy dużej ilości danych
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"},  # Dopasowanie do brain.py
    )

    print("\n" + "=" * 30)
    print("✅ INGESTION ZAKOŃCZONE SUKCESEM!")
    print(f"Zindeksowano: {len(chunks)} fragmentów.")
    print(f"Lokalizacja bazy: {CHROMA_PATH}")
    print("=" * 30)


if __name__ == "__main__":
    # Upewnij się, że folder data istnieje
    if not os.path.exists("data"):
        os.makedirs("data")
        print(
            "📁 Utworzono folder 'data/'. Wrzuć tam swoje pliki PDF i uruchom ponownie."
        )
    else:
        run_ingestion()
