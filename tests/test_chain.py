from unittest.mock import MagicMock, patch
from langchain_core.runnables import Runnable
from src.chain import build_rag_chain, SYSTEM_PROMPT


def _mock_store():
    store = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = []
    store.as_retriever.return_value = retriever
    return store


def _mock_llm():
    llm = MagicMock()
    response = MagicMock()
    response.content = "Mocked answer."
    llm.invoke.return_value = response
    return llm


def test_build_rag_chain_returns_runnable():
    with patch("src.chain.get_vector_store", return_value=_mock_store()), \
         patch("src.chain._build_llm", return_value=_mock_llm()):
        chain = build_rag_chain()
    assert isinstance(chain, Runnable)


def test_build_rag_chain_returns_answer_and_context():
    with patch("src.chain.get_vector_store", return_value=_mock_store()), \
         patch("src.chain._build_llm", return_value=_mock_llm()):
        chain = build_rag_chain()
        result = chain.invoke({"input": "What are total sales?", "chat_history": []})
    assert "answer" in result
    assert "context" in result
    assert isinstance(result["context"], list)


def test_system_prompt_contains_grounding_instructions():
    assert "do not fabricate" in SYSTEM_PROMPT.lower() or "retrieved context" in SYSTEM_PROMPT.lower()
    assert "source" in SYSTEM_PROMPT.lower()
    assert "uncertainty" in SYSTEM_PROMPT.lower()
