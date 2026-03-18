from __future__ import annotations
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from src.ingestion import get_vector_store

load_dotenv()

SYSTEM_PROMPT = (
    "You are InsightForge, an AI business intelligence assistant. "
    "Answer questions strictly based on the retrieved context provided. "
    "Do not fabricate data or statistics. "
    "When answering numeric questions, prefer pre-aggregated summary statistics. "
    "Always cite the source of your answer (e.g., 'According to CSV sales data...' "
    "or 'According to [PDF filename]...'). "
    "If the retrieved context does not contain enough information to answer, "
    "explicitly say so rather than guessing. "
    "Acknowledge uncertainty rather than hallucinate.\n\n"
    "Context:\n{context}"
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0)


def _build_retriever():
    store = get_vector_store()
    return store.as_retriever(search_kwargs={"k": 5})


def build_rag_chain():
    """Build a stateless LCEL RAG chain.

    Input:  {"input": str, "chat_history": list[BaseMessage]}
    Output: {"answer": str, "context": list[Document]}

    Caller is responsible for maintaining and passing chat_history.
    """
    llm = _build_llm()
    retriever = _build_retriever()

    def _run(x: dict) -> dict:
        docs = retriever.invoke(x["input"])
        context = "\n\n".join(doc.page_content for doc in docs)
        messages = _PROMPT.invoke({
            "input": x["input"],
            "chat_history": x.get("chat_history", []),
            "context": context,
        })
        answer = llm.invoke(messages).content
        return {"answer": answer, "context": docs}

    return RunnableLambda(_run)
