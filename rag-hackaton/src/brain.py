import os
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. KONFIGURACJA ŚCIEŻEK (Musi pasować do ingestion.py!) ---
# Pobieramy ścieżkę do folderu, w którym jest ten plik (src)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Wychodzimy jeden poziom wyżej (do rag-hackaton)
BASE_DIR = os.path.dirname(CURRENT_DIR)
# Wskazujemy folder z bazą
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

# --- 2. KONFIGURACJA MODELI (Lokalne na Maca) ---
# Ważne: Musi być ten sam model embeddingów co w ingestion!
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "mistral-nemo"

# --- 3. SYSTEM PROMPT (Osobowość Prawnika) ---
SYSTEM_TEMPLATE = """
Jesteś OrbitCounsel, zaawansowanym asystentem prawnym specjalizującym się w Prawie Kosmicznym (UNOOSA).
Twoim zadaniem jest udzielanie precyzyjnych odpowiedzi na podstawie dostarczonej dokumentacji.

ZASADY:
1. Bazuj TYLKO na poniższym KONTEKŚCIE. Nie wymyślaj faktów.
2. Jeśli nie znasz odpowiedzi na podstawie kontekstu, napisz: "Niestety, dokumentacja nie zawiera informacji na ten temat."
3. Cytuj źródła (np. "zgodnie z Artykułem IV...").
4. Utrzymuj profesjonalny ton.

KONTEKST:
{context}
"""

def get_rag_chain():
    """
    Buduje i zwraca gotowy łańcuch RAG.
    """
    
    # Sprawdzenie czy baza istnieje
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"❌ BŁĄD: Nie znaleziono bazy danych w: {DB_PATH}. \nUruchom najpierw 'python src/ingestion.py'!")

    print(f"🧠 Ładowanie bazy wiedzy z: {DB_PATH}")

    # 1. Inicjalizacja Embeddingów
    embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)

    # 2. Podłączenie do ChromaDB
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    # 3. Konfiguracja Wyszukiwarki (Retriever)
    # search_type="mmr" - Max Marginal Relevance (szuka różnorodnych fragmentów, nie tylko identycznych)
    # k=5 - pobiera 5 najlepszych fragmentów
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    # 4. Inicjalizacja LLM (Mózgu)
    llm = ChatOllama(
        model=LLM_MODEL_NAME,
        temperature=0,      # 0 = Maksymalna precyzja, zero halucynacji
        keep_alive="1h"     # Trzymaj model w RAM, żeby działał szybciej przy kolejnym pytaniu
    )

    # 5. Szablon Prompta
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ])

    # 6. Funkcja pomocnicza do sklejania dokumentów w tekst
    def format_docs(docs):
        # Łączy treść dokumentów, oddzielając je dwoma nowymi liniami
        return "\n\n".join(doc.page_content for doc in docs)

    # 7. Budowa Łańcucha (LCEL - LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# --- 4. TEST BEZPOŚREDNI (Gdy uruchamiasz plik ręcznie) ---
if __name__ == "__main__":
    try:
        print("⏳ Inicjalizacja OrbitCounsel...")
        chain = get_rag_chain()
        print("✅ System gotowy! (Wpisz 'exit' aby wyjść)")
        
        while True:
            question = input("\n⚖️  Twoje pytanie: ")
            if question.lower() in ["exit", "wyjście", "q"]:
                break
            
            print("\n📝 Generowanie odpowiedzi...\n")
            # Streamowanie (efekt pisania na żywo)
            for chunk in chain.stream(question):
                print(chunk, end="", flush=True)
            print("\n" + "-"*50)
            
    except Exception as e:
        print(f"\n❌ Wystąpił błąd: {e}")