from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

QA_PAIRS: list[dict] = [
    {
        "question": "Which product had the highest total sales?",
        "answer": "Widget A had the highest total sales.",
    },
    {
        "question": "Which region generated the most sales?",
        "answer": "The West region generated the most sales.",
    },
    {
        "question": "What is the overall median sales value across all records?",
        "answer": "The overall median sales is 552.5.",
    },
    {
        "question": "Which age group had the highest total sales?",
        "answer": "The 46+ age group had the highest total sales.",
    },
    {
        "question": "How many total rows are in the dataset?",
        "answer": "The dataset contains 2500 rows.",
    },
]

_GRADER_SYSTEM = (
    "You are grading a predicted answer against a reference answer for a business "
    "intelligence question. Reply with exactly one word: CORRECT or INCORRECT."
)


def _parse_grade(result_text: str) -> str:
    """Extract CORRECT or INCORRECT from grader LLM output (case-insensitive)."""
    upper = result_text.upper()
    if "INCORRECT" in upper:
        return "INCORRECT"
    if "CORRECT" in upper:
        return "CORRECT"
    return "UNKNOWN"


def run_evaluation(retrieval_chain, grader_llm=None) -> list[dict]:
    """Evaluate retrieval_chain against QA_PAIRS using a direct LLM grader.

    Args:
        retrieval_chain: LCEL chain built by build_rag_chain().
        grader_llm: BaseLanguageModel; defaults to ChatOpenAI(gpt-4o, temperature=0).

    Returns:
        list of dicts with keys: question, reference, predicted, grade
    """
    if grader_llm is None:
        grader_llm = ChatOpenAI(model="gpt-4o", temperature=0)

    results = []
    for pair in QA_PAIRS:
        result = retrieval_chain.invoke({"input": pair["question"], "chat_history": []})
        predicted = result["answer"]

        grade_messages = [
            SystemMessage(content=_GRADER_SYSTEM),
            HumanMessage(content=(
                f"Question: {pair['question']}\n"
                f"Reference answer: {pair['answer']}\n"
                f"Predicted answer: {predicted}\n\n"
                "Is the predicted answer correct?"
            )),
        ]
        grade_response = grader_llm.invoke(grade_messages)
        grade = _parse_grade(grade_response.content)

        results.append({
            "question": pair["question"],
            "reference": pair["answer"],
            "predicted": predicted,
            "grade": grade,
        })

    return results
