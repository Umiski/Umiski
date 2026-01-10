import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import get_config


def run_ingestion():
    config = get_config()

    # 1. Ładowanie PDFów z folderu data/
    documents = []
    for file in os.listdir(config["data_path"]):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(config["data_path"], file))
            documents.extend(loader.load())

    if not documents:
        print("Folder data/ jest pusty! Kasia musi tam wrzucić dokumenty NASA/ESA.")
        return

    # 2. Chunking - SONIA: Tutaj zmieniasz parametry (rozmiar fragmentów)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Wektoryzacja i zapis do ChromaDB
    # SONIA: Możesz tu zmienić model embeddingów na nowszy/tańszy
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(api_key=config["openai_api_key"]),
        persist_directory=config["chroma_path"],
    )

    print(f"Sukces! Przetworzono {len(chunks)} fragmentów do bazy wektorowej.")


def quick_chat():
    """Funkcja do interaktywnego testowania bota w konsoli."""
    print("\n🚀 AstroGuide CLI Test Mode (wpisz 'exit' aby wyjść)")
    print("-" * 50)

    while True:
        user_input = input("Ty: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        try:
            # Używamy modelu Flash 2.5 - jest darmowy i najszybszy
            answer = get_astro_answer(user_input)
            print(f"\nAstroGuide: {answer}\n")
            print("-" * 20)
        except Exception as e:
            print(f"❌ Błąd: {e}")


if __name__ == "__main__":
    run_ingestion()
    # quick_chat()
