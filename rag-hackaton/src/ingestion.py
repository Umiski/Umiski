import os
import shutil

from tqdm import tqdm

# ... (reszta importów)


def run_ingestion():
    config = get_config()
    print("🚀 ASTROGUIDE: Budowanie bazy wiedzy...")

    if not config["google_api_key"]:
        print("❌ BŁĄD: Brak GOOGLE_API_KEY!")
        return

    if not os.path.exists(config["data_path"]) or not os.listdir(config["data_path"]):
        print(f"❌ Brak plików PDF w {config['data_path']}")
        return

    # Inicjalizacja embeddingów raz
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        task_type="retrieval_document",
        google_api_key=config["google_api_key"],
    )

    # Czyścimy starą bazę
    if os.path.exists(config["chroma_path"]):
        shutil.rmtree(config["chroma_path"])
        print("🧹 Stara baza usunięta.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        length_function=len,
        add_start_index=True,
    )

    all_chunks = []
    pdf_files = [f for f in os.listdir(config["data_path"]) if f.endswith(".pdf")]

    print(f"📄 Znaleziono {len(pdf_files)} dokumentów. Rozpoczynam przetwarzanie...")

    for file in tqdm(pdf_files, desc="Przetwarzanie PDF"):
        file_path = os.path.join(config["data_path"], file)
        try:
            loader = PyPDFLoader(file_path)
            # Ładujemy i od razu tniemy plik
            pages = loader.load()

            # Dodatkowe czyszczenie metadanych
            for page in pages:
                page.metadata["source"] = file
                # PyPDFLoader dodaje 'page', więc mamy to z automatu

            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"⚠️ Błąd przy pliku {file}: {e}")

    # Zapis do Chroma z wymuszeniem Cosine Similarity
    print(f"☁️  Indeksowanie {len(all_chunks)} fragmentów w ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=config["chroma_path"],
        collection_metadata={"hnsw:space": "cosine"},  # MUSI być spójne z brain.py
    )

    print(f"🎉 Misja zakończona sukcesem! Baza gotowa.")


# ...
