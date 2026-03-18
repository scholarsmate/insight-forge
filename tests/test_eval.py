from unittest.mock import MagicMock
from src.eval import _parse_grade, run_evaluation, QA_PAIRS


def test_parse_grade_correct():
    assert _parse_grade("CORRECT") == "CORRECT"


def test_parse_grade_correct_case_insensitive():
    assert _parse_grade("Grade: correct\nExplanation...") == "CORRECT"


def test_parse_grade_incorrect():
    assert _parse_grade("INCORRECT - the answer was wrong") == "INCORRECT"


def test_parse_grade_unknown():
    assert _parse_grade("I cannot determine") == "UNKNOWN"


def test_run_evaluation_returns_correct_schema():
    # Mock retrieval chain: returns fixed answer and empty context
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"answer": "Widget A", "context": []}

    # Mock grader LLM: returns a message with content "CORRECT"
    mock_grader_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "CORRECT"
    mock_grader_llm.invoke.return_value = mock_response

    results = run_evaluation(mock_chain, grader_llm=mock_grader_llm)

    assert isinstance(results, list)
    assert len(results) == len(QA_PAIRS)
    assert mock_chain.invoke.call_count == len(QA_PAIRS)
    for row in results:
        assert set(row.keys()) == {"question", "reference", "predicted", "grade"}
        assert row["grade"] in {"CORRECT", "INCORRECT", "UNKNOWN"}
