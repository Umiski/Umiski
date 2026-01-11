import os
from operator import itemgetter

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser  # Dodano dla czystego tekstu
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

from src.utils import get_config

_CACHED_CHAIN = None
use_groq = True  # Ustaw na False, aby używać modeli Google zamiast Groq


def get_rag_chain():
    global _CACHED_CHAIN
    if _CACHED_CHAIN is not None:
        return _CACHED_CHAIN

    config = get_config()

    if not os.path.exists(config["chroma_path"]):
        raise FileNotFoundError("❌ Brak bazy! Uruchom najpierw ingestion.")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config["embedding_model"], google_api_key=config["google_api_key"]
    )

    vectorstore = Chroma(
        persist_directory=config["chroma_path"], embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": config["retrieval_k"]})
    if not use_groq:
        llm = ChatGoogleGenerativeAI(
            model=config["llm_model"],
            google_api_key=config["google_api_key"],
            temperature=config["temperature"],
        )
    else:
        llm = ChatGroq(model_name=config["llm_model"], temperature=0.1, max_tokens=1024)

    system_template = """Jesteś AstroGuide, ekspertem od prawa kosmicznego i regulacji.
    Odpowiadaj konkretnie na podstawie kontekstu. Jeśli nie ma informacji, powiedz to wprost.
    
    KONTEKST:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("human", "{question}")]
    )

    # KLUCZOWA ZMIANA: Zapewniamy, że retriever dostaje tylko string (tekst pytania)
    _CACHED_CHAIN = RunnableParallel(
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
    ).assign(
        answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
        )
    )

    return _CACHED_CHAIN


def get_astro_answer(query_text):
    chain = get_rag_chain()

    # Wywołujemy łańcuch
    result = chain.invoke({"question": query_text})

    answer = result["answer"]
    raw_docs = result["context"]  # To są dokumenty znalezione przez retrievera

    # Wyciągamy szczegółowe źródła: Plik + Strona
    detailed_sources = []
    for doc in raw_docs:
        source_name = doc.metadata.get("source", "Nieznany plik")
        page_num = doc.metadata.get("page", "?")  # PyPDFLoader dodaje to domyślnie

        # Tworzymy ładny opis fragmentu
        source_info = f"📄 {source_name} (str. {page_num + 1})"  # +1 bo indeksuje od 0
        if source_info not in detailed_sources:
            detailed_sources.append(source_info)

    return {
        "answer": answer,
        "sources": detailed_sources,
        "raw_fragments": [
            doc.page_content[:200] + "..." for doc in raw_docs
        ],  # Opcjonalnie dla Kamila do debugowania
    }


def quick_chat():
    print("\n🚀 AstroGuide (Expert Mode) - Test bazy i źródeł")
    print("Wpisz 'q', aby wyjść.")
    print("-" * 60)

    try:
        chain = get_rag_chain()

        while True:
            query = input("\nTy: ")
            if query.lower() in ["q", "exit"]:
                break

            # 1. Wywołanie łańcucha (invoke zamiast stream dla łatwiejszego dostępu do słownika)
            print("AstroGuide analizuje dokumentację...", end="\r")
            result = chain.invoke({"question": query})

            # 2. Wyświetlenie odpowiedzi
            print(f"\nAstroGuide: {result['answer']}")

            # 3. Wyświetlenie szczegółowych źródeł (Metadane dla Sonii i Kamila)
            print(f"\n{'=' * 20} ŹRÓDŁA (METADANE) {'=' * 20}")

            # 'context' zawiera listę obiektów Document znalezionych przez retrievera
            raw_docs = result.get("context", [])

            if not raw_docs:
                print("⚠️ Brak fragmentów w kontekście (retriever nic nie znalazł).")
            else:
                seen_sources = set()
                for i, doc in enumerate(raw_docs, 1):
                    # Wyciągamy metadane dodane podczas Ingestion
                    source_file = doc.metadata.get("source", "Nieznany plik")
                    page_num = doc.metadata.get("page", 0) + 1  # +1 bo PDFy są od 0

                    source_id = f"{source_file} (str. {page_num})"

                    if source_id not in seen_sources:
                        print(f"[{i}] {source_id}")
                        # Opcjonalnie: wyświetl fragment tekstu dla Kamila (debugowanie promptu)
                        # print(f"    Snippet: {doc.page_content[:100]}...")
                        seen_sources.add(source_id)

            print("-" * 60)

    except Exception as e:
        print(f"❌ Błąd podczas rozmowy: {e}")


if __name__ == "__main__":
    quick_chat()
