import os
import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.brain import get_rag_chain
from src.utils import get_config

# 1. Konfiguracja Sędziego (Groq)
# Używamy tego samego modelu co w brain.py, ale z temperaturą 0 dla powtarzalności ocen.
config = get_config()
judge_llm = ChatGroq(
    model_name=config["llm_model"],  # Np. llama3-70b-8192
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


def extract_json_from_text(text):
    """Extract JSON from text that may contain additional content"""
    # Look for JSON-like structure
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try to find score and reason separately
    score_match = re.search(r'"score"\s*:\s*([01])', text)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)

    if score_match and reason_match:
        return {"score": int(score_match.group(1)), "reason": reason_match.group(1)}

    # Fallback
    return {"score": 0, "reason": "Failed to parse response"}


def evaluate_faithfulness(answer, context_text):
    """
    Sprawdza Wierność (Faithfulness): Czy odpowiedź wynika TYLKO z dostarczonych dokumentów?
    Chroni przed halucynacjami (zmyślaniem faktów).
    """
    prompt = ChatPromptTemplate.from_template("""
    Jesteś surowym sędzią AI. Oceniasz "Wierność" (Faithfulness) odpowiedzi systemu RAG.

    KONTEKST (Źródła wiedzy):
    {context}

    ODPOWIEDŹ SYSTEMU:
    {answer}

    Zadanie: Przeanalizuj czy odpowiedź wynika TYLKO z kontekstu. Jeśli zawiera informacje spoza kontekstu, daj 0. Jeśli wszystko jest oparte na kontekście, daj 1.

    Odpowiedz WYŁĄCZNIE w formacie JSON, bez żadnego dodatkowego tekstu:
    {{"score": <0 lub 1>, "reason": "<krótki powód>"}}
    """)

    chain = prompt | judge_llm | StrOutputParser()
    raw_response = chain.invoke({"answer": answer, "context": context_text})
    return extract_json_from_text(raw_response)


def evaluate_relevancy(question, answer):
    """
    Sprawdza Trafność (Relevancy): Czy odpowiedź faktycznie odpowiada na zadane pytanie?
    """
    prompt = ChatPromptTemplate.from_template("""
    Jesteś surowym sędzią AI. Oceniasz "Trafność" (Answer Relevancy).

    PYTANIE UŻYTKOWNIKA:
    {question}

    ODPOWIEDŹ SYSTEMU:
    {answer}

    Zadanie: Oceń czy odpowiedź jest na temat pytania. Daj 1 jeśli odpowiedź dotyczy pytania, 0 jeśli nie.

    Odpowiedz WYŁĄCZNIE w formacie JSON, bez żadnego dodatkowego tekstu:
    {{"score": <0 lub 1>, "reason": "<krótki powód>"}}
    """)

    chain = prompt | judge_llm | StrOutputParser()
    raw_response = chain.invoke({"question": question, "answer": answer})
    return extract_json_from_text(raw_response)


def run_evaluation():
    print("\n🚀 START EWALUACJI (Sędzia: Groq/Llama3)")
    print("-" * 50)

    # Zestaw pytań testowych ("Golden Dataset")
    test_questions = [
        "Czym jest obiekt kosmiczny w świetle prawa?",
        "Kto odpowiada za szkody wyrządzone przez satelitę na Ziemi?",
        "Czy Księżyc może należeć do prywatnej firmy?",
        "Jaki jest przepis na ciasto marchewkowe?",  # Test negatywny (Guardrails)
    ]

    rag_chain = get_rag_chain()

    total_faithfulness = 0
    total_relevancy = 0
    results_log = []

    for q in test_questions:
        print(f"🔍 Pytanie: {q}")

        # 2. Uruchomienie RAG (Twojego Braina)
        # UWAGA: brain.py wymaga klucza "question", nie "input"
        response = rag_chain.invoke({"question": q})

        answer = response["answer"]
        # Wyciągamy tekst z dokumentów źródłowych (context)
        context_docs = response["context"]
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        # 3. Ocena Sędziego
        try:
            faith_result = evaluate_faithfulness(answer, context_text)
            rel_result = evaluate_relevancy(q, answer)

            # Logowanie wyników
            print(f"   🤖 Odpowiedź: {answer[:80]}...")
            print(
                f"   🛡️  Wierność (Faithfulness): {faith_result['score']} -> {faith_result['reason']}"
            )
            print(
                f"   🎯 Trafność (Relevancy):    {rel_result['score']} -> {rel_result['reason']}"
            )

            total_faithfulness += faith_result["score"]
            total_relevancy += rel_result["score"]

        except Exception as e:
            print(f"   ⚠️ Błąd oceny dla tego pytania: {e}")

        print("-" * 50)

    # 4. Raport Końcowy
    avg_faith = total_faithfulness / len(test_questions)
    avg_rel = total_relevancy / len(test_questions)

    print("\n📊 RAPORT KOŃCOWY:")
    print(f"Średnia Wierność: {avg_faith:.2f} / 1.0")
    print(f"Średnia Trafność: {avg_rel:.2f} / 1.0")

    if avg_faith < 0.8:
        print(
            "⚠️ SUGESTIA: Model halucynuje. Zwiększ 'temperature' na 0 lub popraw Prompt Systemowy w brain.py."
        )
    if avg_rel < 0.8:
        print("⚠️ SUGESTIA: Model nie odpowiada wprost. Sprawdź retrieval_k w utils.py.")


if __name__ == "__main__":
    run_evaluation()
