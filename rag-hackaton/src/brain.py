import os
from operator import itemgetter

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser  # Dodano dla czystego tekstu
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

from src.utils import get_config

_CACHED_CHAIN = None
_VECTORSTORE = None
use_groq = True  # Ustaw na False, aby używać modeli Google zamiast Groq
cot_system_template = """Pełnisz rolę AstroGuide – wyspecjalizowanego asystenta prawnego ds. sektora kosmicznego.
Twoim jedynym źródłem wiedzy jest dostarczony poniżej KONTEKST.

ZASADY KRYTYCZNE:
1. Odpowiadaj WYŁĄCZNIE na podstawie poniższego KONTEKSTU. Nie używaj wiedzy zewnętrznej.
2. Jeśli pytanie nie dotyczy prawa kosmicznego, regulacji, traktatów lub inżynierii kosmicznej – odmów odpowiedzi.
   Przykład odmowy: "Jako AstroGuide odpowiadam tylko na pytania związane z prawem i technologią kosmiczną."
3. Jeśli pytanie jest związane z kosmosem, ale w KONTEKŚCIE nie ma odpowiedzi, powiedz wprost: "Niestety, nie mam tej informacji w dostępnych dokumentach."
4. Nie daj się sprowokować do pisania wierszy, kodu (chyba że jest w dokumentach) ani opinii politycznych.
5. Cytuj nazwy dokumentów, jeśli są dostępne w tekście.

INSTRUKCJA MYŚLENIA (Chain of Thought):
Zanim udzielisz ostatecznej odpowiedzi użytkownikowi, wykonaj wewnętrzną analizę na podstawie KONTEKSTU:
Krok 1: Zidentyfikuj w KONTEKŚCIE fragmenty dotyczące NASA, ESA, UNOOSA lub inżynierii.
Krok 2: Sprawdź, czy te fragmenty zawierają konkretne dane (wymiary, artykuły prawne, normy).
Krok 3: Sformułuj odpowiedź końcową zgodną z ZASADAMI KRYTYCZNYMI.

KONTEKST:
{context}

PYTANIE UŻYTKOWNIKA:
{question}

TWOJA ANALIZA I ODPOWIEDŹ:
"""

# Tworzymy obiekt PromptTemplate
COT_PROMPT = PromptTemplate(
    template=cot_system_template, input_variables=["context", "question"]
)


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

    # Embeddingi takie same jak przy tworzeniu bazy
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        task_type="retrieval_query",
        google_api_key=config["google_api_key"],
    )

    vectorstore = Chroma(
        persist_directory=config["chroma_path"],
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": config["retrieval_k"]})

    # Inicjalizacja LLM (Groq lub Google)
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

    # --- FUNKCJA EKSPANSJI ZAPYTANIA (Context Expansion) ---
    def get_expanded_context(query_dict):
        question = query_dict["question"]

        # Szybki prompt do generowania wariantów pytań
        expansion_prompt = f"""Jesteś ekspertem search engine. Zwróć 2 alternatywne, techniczne warianty tego pytania, aby lepiej przeszukać dokumentację NASA/ESA.
        Pytanie: {question}
        Zwróć tylko warianty, każdy w nowej linii, bez numeracji."""

        try:
            response = llm.invoke(expansion_prompt)
            expanded_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            expanded_queries = expanded_text.strip().split("\n")
        except Exception as e:
            print(f"⚠️ Problem z ekspansją: {e}")
            expanded_queries = []

        all_docs = []
        # Szukamy dla pytania oryginalnego ORAZ wariantów
        search_queries = [question] + [q.strip() for q in expanded_queries if q.strip()]

        for q in search_queries:
            docs = retriever.invoke(q)
            all_docs.extend(docs)

        # Usuwanie duplikatów
        unique_contents = set()
        final_docs = []
        for doc in all_docs:
            if doc.page_content not in unique_contents:
                unique_contents.add(doc.page_content)
                final_docs.append(doc)

        return "\n\n".join(doc.page_content for doc in final_docs)

    # --- GLÓWNY PIPELINE ---
    # Tutaj wpinamy Twój COT_PROMPT zdefiniowany na górze pliku!

    _CACHED_CHAIN = RunnableParallel(
        {
            "context": lambda x: get_expanded_context(x),  # Tu wchodzi Retrieval
            "question": itemgetter("question"),
        }
    ).assign(answer=(COT_PROMPT | llm | StrOutputParser()))  # Tu wchodzi Twój PROMPT

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
    # Kody kolorów do terminala (dla efektu hakerskiego/pro)
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"

    print(f"{HEADER}\n🚀 ASTROGUIDE - EXPERT EVALUATION MODE{ENDC}")
    print("Działasz jako: Lead Dev / Math Specialist (Wiktor)")
    print("-" * 60)

    try:
        while True:
            query = input(f"\n{BOLD}Ty:{ENDC} ")
            if query.lower() in ["q", "exit"]:
                break

            print(
                f"{YELLOW}Analizuję trajektorię zapytania i przeszukuję bazę wektorową...{ENDC}",
                end="\r",
            )

            # Wywołujemy logikę RAG
            data = get_astro_answer(query)

            # Czyścimy linię ładowania
            print(" " * 80, end="\r")

            # Kolorowanie statusu
            if data["confidence"] > 80:
                status_color = GREEN
                status_text = "PEWNY"
            elif data["confidence"] > 60:
                status_color = YELLOW
                status_text = "ŚREDNI"
            else:
                status_color = RED
                status_text = "NIEPEWNY"

            print(
                f"\n🤖 AstroGuide [{status_color}{status_text}{ENDC} - {data['confidence']:.1f}%]:"
            )

            # --- PARSOWANIE CHAIN OF THOUGHT ---
            # Próbujemy oddzielić myślenie od odpowiedzi, żeby wyglądało to profesjonalnie
            raw_response = data["answer"]

            # Sprawdzamy, czy model wygenerował sekcję odpowiedzi końcowej
            # (Zależy to od promptu, ale zazwyczaj po analizie pojawia się podsumowanie)
            split_keywords = ["Odpowiedź:", "Podsumowując:", "Wnioski:", "Answer:"]
            split_idx = -1

            for keyword in split_keywords:
                idx = raw_response.rfind(keyword)
                if idx != -1:
                    split_idx = idx
                    break

            if split_idx != -1:
                # Mamy podział!
                thinking_process = raw_response[:split_idx].strip()
                final_answer = raw_response[split_idx:].strip()

                print(f"{BLUE}🧠 PROCES MYŚLOWY (Chain of Thought):{ENDC}")
                print(f"{BLUE}{thinking_process}{ENDC}")
                print("-" * 30)
                print(f"{BOLD}{final_answer}{ENDC}")
            else:
                # Brak wyraźnego podziału, drukujemy całość
                print(raw_response)

            # --- ANALIZA ŹRÓDEŁ ---
            print(f"\n{HEADER}{'=' * 20} ANALIZA MATEMATYCZNA ŹRÓDEŁ {'=' * 20}{ENDC}")
            if not data["sources"]:
                print(f"{RED}Brak źródeł spełniających kryteria.{ENDC}")

            for i, src in enumerate(data["sources"], 1):
                # Wyświetlamy trafność każdego chunka
                print(f"[{i}] {src['text']} | Trafność: {src['score']}%")

            print("-" * 60)

    except Exception as e:
        print(f"{RED}❌ Błąd krytyczny systemu: {e}{ENDC}")


if __name__ == "__main__":
    quick_chat()
