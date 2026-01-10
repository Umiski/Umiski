from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.utils import get_config


def get_astro_answer(user_query):
    config = get_config()

    # 1. Sonia: Darmowe i szybkie embeddingi Google
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=config["google_api_key"]
    )

    # 2. Wiktor: Podpięcie bazy Chroma
    vectorstore = Chroma(
        persist_directory=config["chroma_path"], embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 3. Kamil: Model Gemini (używamy Flash dla szybkości demo)
    # potem zmieniamy na models/gemini-2.5-pro lub flash (pro ma mocne limity)
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-lite",
        google_api_key=config["google_api_key"],
    )

    # 4. System Prompt
    system_prompt = "Jesteś AstroGuide. Odpowiadaj na podstawie: {context}"
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # 5. Łańcuch RAG
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return rag_chain.invoke({"input": user_query})["answer"]
