import os
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)  # ZMIANA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.utils import get_config


def get_rag_chain():
    config = get_config()

    if not os.path.exists(config["chroma_path"]):
        raise FileNotFoundError(
            "❌ Brak bazy! Uruchom najpierw: python -m src.ingestion"
        )

    # 1. Embeddingi (Te same co w ingestion!)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config["embedding_model"], google_api_key=config["google_api_key"]
    )

    # 2. Baza
    vectorstore = Chroma(
        persist_directory=config["chroma_path"], embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": config["retrieval_k"]})

    # 3. LLM (Gemini 1.5 Flash)
    llm = ChatGoogleGenerativeAI(
        model=config["llm_model"],
        google_api_key=config["google_api_key"],
        temperature=config["temperature"],
    )

    # 4. Prompt
    system_template = """You are AstroGuide, an expert in space law, regulations, and rules for startups and individuals launching objects into space.
    
    Based on the provided context from space law documents, answer questions about compliance, licensing, international treaties (e.g., Outer Space Treaty), and practical steps for space missions.
    If the context doesn't cover it, say so and suggest consulting official sources like the UN Office for Outer Space Affairs.
    
    Be precise, cite relevant regulations where possible, and avoid speculation.
    
    Context: {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("human", "{question}")]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Łańcuch
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def quick_chat():
    print("🚀 AstroGuide (Google Gemini) - Test Mode")
    try:
        chain = get_rag_chain()
        while True:
            q = input("\nTy: ")
            if q.lower() in ["q", "exit"]:
                break

            print("Gemini: ", end="", flush=True)
            for chunk in chain.stream(q):
                print(chunk, end="", flush=True)
            print("\n")
    except Exception as e:
        print(f"❌ Błąd: {e}")


if __name__ == "__main__":
    quick_chat()
