import os
from operator import itemgetter

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser  # Dodano dla czystego tekstu
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.utils import get_config

_CACHED_CHAIN = None


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

    llm = ChatGoogleGenerativeAI(
        model=config["llm_model"],
        google_api_key=config["google_api_key"],
        temperature=config["temperature"],
    )

    system_template = """Jesteś AstroGuide, ekspertem od prawa kosmicznego i regulacji.
    Odpowiadaj konkretnie na podstawie kontekstu. Jeśli nie ma informacji, powiedz to wprost.
    
    KONTEKST:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("human", "{question}")]
    )

    # KLUCZOWA ZMIANA: Zapewniamy, że retriever dostaje tylko string (tekst pytania)
    _CACHED_CHAIN = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return _CACHED_CHAIN


def get_astro_answer(query_text):
    """Główna funkcja wywoływana przez Streamlit."""
    chain = get_rag_chain()

    # ZMIANA: Przekazujemy słownik, bo tego wymaga itemgetter w chainie
    answer = chain.invoke({"question": query_text})

    # Pobieramy źródła (ponowne wyszukanie dla metadanych)
    config = get_config()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config["embedding_model"], google_api_key=config["google_api_key"]
    )
    vectorstore = Chroma(
        persist_directory=config["chroma_path"], embedding_function=embeddings
    )
    # Szukamy dokumentów, aby wyciągnąć nazwy plików
    docs = vectorstore.similarity_search(query_text, k=config["retrieval_k"])
    sources = list(set([d.metadata.get("source", "Nieznany plik") for d in docs]))

    return {"answer": answer, "sources": sources}


# Funkcja quick_chat zostaje bez zmian (używa już poprawnego słownika w stream)


def quick_chat():
    print("\n🚀 AstroGuide (Gemini 1.5 Flash) - Test Mode")
    print("Wpisz 'q' lub 'exit', aby zakończyć.")
    print("-" * 50)

    try:
        # 1. Pobieramy łańcuch (raz, dzięki cache w get_rag_chain)
        chain = get_rag_chain()

        while True:
            query = input("\nTy: ")
            if query.lower() in ["q", "exit", "quit"]:
                break

            print("\nAstroGuide: ", end="", flush=True)

            # 2. Streaming odpowiedzi (efekt pisania na żywo)
            # Przekazujemy słownik zgodnie z definicją w LCEL
            full_response = ""
            for chunk in chain.stream({"question": query}):
                # Jeśli używasz StrOutputParser() na końcu chaina, chunk to string
                # Jeśli nie używasz, chunk to AIMessageChunk (wtedy: chunk.content)
                content = chunk if isinstance(chunk, str) else chunk.content
                print(content, end="", flush=True)
                full_response += content

            # 3. Dodatkowy krok dla Sonii i Kasi: Wyświetlenie źródeł
            # Musimy ręcznie sprawdzić, co retriever podał jako kontekst
            print("\n" + "." * 20)
            try:
                # Pobieramy źródła, żeby sprawdzić czy metadane z ingestion.py działają
                ans_data = get_astro_answer(query)  # Wykorzystujemy wrapper ze źródłami
                sources = ans_data.get("sources", [])
                if sources:
                    print(f"📚 Źródła: {', '.join(sources)}")
                else:
                    print("⚠️ Brak konkretnych źródeł w kontekście.")
            except Exception as e:
                print(f"DEBUG: Nie udało się pobrać źródeł: {e}")

            print("-" * 50)

    except Exception as e:
        print(f"❌ Błąd krytyczny: {e}")


if __name__ == "__main__":
    quick_chat()
