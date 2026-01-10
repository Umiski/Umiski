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


if __name__ == "__main__":
    run_ingestion()
