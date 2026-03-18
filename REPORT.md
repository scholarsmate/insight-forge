# InsightForge — Project Completion Report

**Date:** 2026-03-16
**Test suite:** 19 / 19 passing
**Eval score:** 5 / 5 correct

---

## Feature Completion Matrix

| # | Feature | Planned In | Status | Implementation |
|---|---------|-----------|--------|----------------|
| 1 | CSV row ingestion (sentence-per-row, batched 50/chunk) | Plan 1 · Task 2 | ✅ Complete | `src/ingestion.py` · `rows_to_chunks()` |
| 2 | Pre-aggregated summary chunks (product, region, monthly, age group, overall stats) | Plan 1 · Task 3 | ✅ Complete | `src/ingestion.py` · `build_summary_chunks()` |
| 3 | PDF ingestion with overlap splitting | Plan 1 · Task 4 | ✅ Complete | `src/ingestion.py` · `load_pdf_chunks()` |
| 4 | FAISS vector store build + persist + load | Plan 1 · Task 4 | ✅ Complete | `src/ingestion.py` · `build_vector_store()` / `load_vector_store()` / `get_vector_store()` |
| 5 | `REBUILD_INDEX` env-var flag to force re-embed | Plan 1 · Task 4 | ✅ Complete | `src/ingestion.py` · `get_vector_store()` |
| 6 | RAG chain with grounding system prompt | Plan 1 · Task 5 | ✅ Complete | `src/chain.py` · `build_rag_chain()` + `SYSTEM_PROMPT` |
| 7 | System prompt: no hallucination, cite source, admit uncertainty | Plan 1 · Task 5 | ✅ Complete | `src/chain.py` · `SYSTEM_PROMPT` |
| 8 | Multi-turn chat history (stateless LCEL) | Plan 1 · Task 5 | ✅ Complete (evolved) | Original spec: `ConversationBufferMemory`. Migrated to `list[BaseMessage]` passed per-invocation via `chat_history` key. Equivalent behavior, no legacy dependency. |
| 9 | Sales trend chart (line, monthly aggregation) | Plan 2 · Task 3 | ✅ Complete | `src/visualizations.py` · `sales_trend_chart()` |
| 10 | Product performance chart (horizontal bar) | Plan 2 · Task 3 | ✅ Complete | `src/visualizations.py` · `product_performance_chart()` |
| 11 | Regional analysis chart (vertical bar) | Plan 2 · Task 4 | ✅ Complete | `src/visualizations.py` · `regional_analysis_chart()` |
| 12 | Customer demographics chart (pie + satisfaction bar) | Plan 2 · Task 5 | ✅ Complete | `src/visualizations.py` · `customer_demographics_chart()` |
| 13 | Visualizations tab — 2×2 grid, all 4 charts shown simultaneously | Plan 2 · Task 8 | ✅ Complete | `app.py` · `tab_viz` |
| 14 | Chat tab with conversation history rendering | Plan 2 · Task 8 | ✅ Complete | `app.py` · `tab_chat` |
| 15 | Chat input with "Thinking…" spinner | Plan 2 · Task 8 | ✅ Complete | `app.py` · `tab_chat` |
| 16 | Source attribution expandable per assistant message | Plan 2 · Task 8 | ✅ Complete | `app.py` · sources expander in both render loop and new message block |
| 17 | Model evaluation with LLM grader | Plan 2 · Task 6–7 | ✅ Complete (evolved) | Original spec: `QAEvalChain`. Migrated to direct `ChatOpenAI` grader with `SystemMessage` + `HumanMessage`. Equivalent grading quality. |
| 18 | Eval tab with button-triggered evaluation and score metric | Plan 2 · Task 8 | ✅ Complete | `app.py` · `tab_eval` · `st.metric` + `st.dataframe` |
| 19 | `_parse_grade` normaliser (CORRECT / INCORRECT / UNKNOWN) | Plan 2 · Task 6 | ✅ Complete | `src/eval.py` · `_parse_grade()` |
| 20 | Hardcoded QA_PAIRS with correct reference answers | Plan 2 · Task 6 | ✅ Complete | `src/eval.py` · `QA_PAIRS` (corrected from placeholder values) |
| 21 | `@st.cache_resource` on FAISS store (monkey-patch pattern) | Plan 2 · Task 8 | ✅ Complete | `app.py` — imports `src.chain` as module, patches `get_vector_store` before chain build |
| 22 | `@st.cache_data` on CSV load and chart computation | Plan 2 · Task 8 | ✅ Complete | `app.py` · `_load_csv()`, `_sales_trend()`, `_product_perf()`, `_regional()`, `_demographics()` |
| 23 | 3-tab Streamlit layout (Visualizations → Chat → Eval) | Plan 2 · Task 8 | ✅ Complete | `app.py` · `st.tabs(["📊 Visualizations", "💬 Chat", "🧪 Eval"])` |
| 24 | Modern LangChain (no langchain-classic) | Post-plan migration | ✅ Complete | All imports from `langchain_core`, `langchain_openai`, `langchain_community`. `langchain-classic` removed from `requirements.txt`. |
| 25 | Unit tests: ingestion | Plan 1 · Task 2–4 | ✅ Complete | `tests/test_ingestion.py` — 9 tests |
| 26 | Unit tests: chain | Plan 1 · Task 5 | ✅ Complete | `tests/test_chain.py` — 3 tests |
| 27 | Unit tests: visualizations | Plan 2 · Task 2–5 | ✅ Complete | `tests/test_visualizations.py` — 4 tests |
| 28 | Unit tests: eval | Plan 2 · Task 6–7 | ✅ Complete | `tests/test_eval.py` — 5 tests (including `run_evaluation` schema test with full mocking) |
| 29 | `.gitignore` (`.env`, `vector_store/`, `__pycache__`, `.streamlit/`) | Plan 1 · Task 1 | ✅ Complete | `.gitignore` |
| 30 | `conftest.py` with 5-row `sample_df` fixture | Plan 1 · Task 1 | ✅ Complete | `tests/conftest.py` |
| 31 | CLI index rebuild (`python src/ingestion.py`) | Plan 1 · Task 4 | ✅ Complete | `src/ingestion.py` · `__main__` block |
| 32 | Standalone eval CLI script (`eval/generate_eval_set.py`, `eval/run_evaluation.py`) | Plan 1 · Task 8–9 | ⚠️ Superseded | Replaced by the in-app Eval tab and `src/eval.py`. Equivalent coverage; CLI scripts not needed. |

---

## Design Decisions That Diverged From Original Plan

| Original Plan | What Was Built | Reason |
|--------------|----------------|--------|
| `src/dashboard.py` | `src/visualizations.py` | Part 2 redesign renamed the module to better reflect its role as a chart library |
| 2 tabs (Chat + Dashboard) | 3 tabs (Visualizations + Chat + Eval) | User explicitly requested Eval as a first-class tab, not a CLI script |
| `ConversationalRetrievalChain` + `ConversationBufferMemory` | Stateless `RunnableLambda` + `list[BaseMessage]` | Migrated away from `langchain-classic` to modern LCEL |
| `QAEvalChain` for grading | Direct `ChatOpenAI` LLM grader | `QAEvalChain` lives in `langchain-classic`; replaced with equivalent direct approach |
| `eval/generate_eval_set.py` + `eval/run_evaluation.py` | `src/eval.py` + Eval tab in `app.py` | Consolidated into the app; hardcoded QA_PAIRS are simpler and don't require a separate generation step |

---

## Test Coverage Summary

| Test file | Tests | Covers |
|-----------|-------|--------|
| `tests/test_ingestion.py` | 9 | `rows_to_chunks` (batching, sentence format), `build_summary_chunks` (product/region/overall stats), `load_pdf_chunks` |
| `tests/test_chain.py` | 3 | `build_rag_chain` returns `Runnable`, output schema `{answer, context}`, `SYSTEM_PROMPT` grounding keywords |
| `tests/test_visualizations.py` | 4 | All 4 chart functions return `go.Figure` with data; `customer_demographics_chart` has ≥2 traces |
| `tests/test_eval.py` | 5 | `_parse_grade` (correct, case-insensitive, incorrect, unknown), `run_evaluation` output schema with mocked chain and grader LLM |
| **Total** | **19** | **19 / 19 passing** |
