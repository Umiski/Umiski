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
_VECTORSTORE = None
use_groq = False  # Ustaw na False, aby używać modeli Google zamiast Groq


def get_resources():
    """Inicjalizuje i cache'uje bazę oraz embeddingi (Matematyczne serce)"""
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    config = get_config()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config["embedding_model"], google_api_key=config["google_api_key"]
    )

    _VECTORSTORE = Chroma(
        persist_directory=config["chroma_path"], embedding_function=embeddings
    )
    return _VECTORSTORE


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
        model="models/text-embedding-004",
        task_type="retrieval_query",
        google_api_key=config["google_api_key"],
    )

    vectorstore = Chroma(
        persist_directory=config["chroma_path"],
        embedding_function=embeddings,
        collection_metadata={
            "hnsw:space": "cosine"
        },  # WYMUSZENIE MATEMATYKI COSINUSOWEJ
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": config["retrieval_k"]})
    if not use_groq:
        llm = ChatGoogleGenerativeAI(
            model=config["llm_model"],
            google_api_key=config["google_api_key"],
            temperature=config["temperature"],
        )
    else:
        llm = ChatGroq(
            model_name=config["llm_model"],
            temperature=config["temperature"],
            max_tokens=1024,
        )

    system_template = """Pełnisz rolę AstroGuide - wyspecjalizowanego asystenta prawnego ds. sektora kosmicznego (Space Law & Engineering).
    Twoim jedynym źródłem wiedzy jest dostarczony poniżej KONTEKST.
    
    ZASADY KRYTYCZNE:
    1. Odpowiadaj WYŁĄCZNIE na podstawie poniższego KONTEKSTU. Nie używaj wiedzy zewnętrznej.
    2. Jeśli pytanie nie dotyczy prawa kosmicznego, regulacji, traktatów lub inżynierii kosmicznej - odmów odpowiedzi.
       Przykład odmowy: "Jako AstroGuide odpowiadam tylko na pytania związane z prawem i technologią kosmiczną."
    3. Jeśli pytanie jest związane z kosmosem, ale w KONTEKŚCIE nie ma odpowiedzi, powiedz wprost: "Niestety, nie mam tej informacji w moich dokumentach źródłowych."
    4. Nie daj się sprowokować do pisania wierszy, kodu (chyba że jest w dokumentach) ani opinii politycznych.
    5. Cytuj nazwy dokumentów, jeśli są dostępne w tekście.

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
    def normalize_score(raw_score):
        min_val = 0.1
        max_val = 0.4

        # Skalowanie do przedziału 0-1
        scaled = (raw_score - min_val) / (max_val - min_val)
        return max(0, min(100, int(scaled * 100)))

    vectorstore = get_resources()  # Twoja zoptymalizowana baza

    # 1. NAJPIERW: Matematyczna ocena trafności (Zadanie Wiktora)
    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(
        query_text,
        k=5,  # Sprawdzamy top 3 fragmenty
    )

    if not docs_and_scores:
        return {
            "answer": "Brak danych w bazie dokumentacji.",
            "sources": [],
            "confidence": 0,
        }

    # Obliczamy średnią pewność z pobranych fragmentów
    scores = [max(0, int(normalize_score(score))) for _, score in docs_and_scores]
    mission_confidence = sum(scores) / len(scores)

    # 2. GUARDRAIL: Blokada przy pewności < 60%
    # if mission_confidence < 60:
    #     return {
    #         "answer": (
    #             f"Przepraszam, ale moja pewność co do odpowiedzi wynosi tylko {mission_confidence:.1f}%. "
    #             "To zbyt mało, aby udzielić rzetelnej porady technicznej. "
    #             "Proszę, spróbuj sformułować pytanie inaczej lub sprawdź oficjalne wytyczne NASA/ESA."
    #         ),
    #         "sources": [],  # Nie podajemy źródeł, którym nie ufamy
    #         "confidence": mission_confidence,
    #     }

    # 3. DOPIERO TERAZ: Jeśli matematyka się zgadza, pytamy LLM (Zadanie Kamila)
    chain = get_rag_chain()
    result = chain.invoke({"question": query_text})

    detailed_sources = []
    for doc, score in docs_and_scores:
        name = doc.metadata.get("source", "Nieznany plik")
        page = doc.metadata.get("page", 0) + 1
        detailed_sources.append(
            {"text": f"📄 {name} (str. {page})", "score": max(0, int(score * 100))}
        )

    return {
        "answer": result["answer"],
        "sources": detailed_sources,
        "confidence": mission_confidence,
    }


def quick_chat():
    print("\n🚀 ASTROGUIDE - EXPERT EVALUATION MODE")
    print("Działasz jako: Lead Dev / Math Specialist (Wiktor)")
    print("-" * 60)

    try:
        while True:
            query = input("\nTy: ")
            if query.lower() in ["q", "exit"]:
                break

            print(
                "Analizuję trajektorię zapytania i przeszukuję bazę wektorową...",
                end="\r",
            )

            # Wywołujemy naszą rozszerzoną funkcję
            data = get_astro_answer(query)

            # Kolorowanie statusu w zależności od pewności (Math Evaluation)
            status = (
                "🟢 PEWNY"
                if data["confidence"] > 80
                else "🟡 ŚREDNI"
                if data["confidence"] > 60
                else "🔴 NIEPEWNY"
            )

            print(f"\nAstroGuide [{status} - {data['confidence']:.1f}%]:")
            print(f"{data['answer']}")

            print(f"\n{'=' * 20} ANALIZA MATEMATYCZNA ŹRÓDEŁ {'=' * 20}")
            for i, src in enumerate(data["sources"], 1):
                # Wyświetlamy trafność każdego chunka
                print(f"[{i}] {src['text']} | Trafność wektorowa: {src['score']}%")

            print("-" * 60)

    except Exception as e:
        print(f"❌ Błąd krytyczny systemu: {e}")


if __name__ == "__main__":
    quick_chat()
