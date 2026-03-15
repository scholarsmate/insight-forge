# InsightForge Design Spec
**Date:** 2026-03-15
**Status:** Approved

## Overview

InsightForge is an AI-powered Business Intelligence Assistant that enables organizations to query and explore business data through natural language. It combines a RAG-based conversational assistant with a pre-built analytics dashboard, built on OpenAI GPT-4o, LangChain, and Streamlit.

---

## Data Sources

- `data/sales_data.csv` — 2,500 rows: Date, Product, Region, Sales, Customer_Age, Customer_Gender, Customer_Satisfaction
- `data/pdf/` — 4 PDFs covering AI/BI domain knowledge: business model innovation, BI approaches, Walmart sales analysis, time-series prediction

Both sources feed the unified RAG knowledge base.

---

## Architecture

```
data/
├── sales_data.csv          → pandas → raw chunks + pre-aggregated summaries
└── pdf/*.pdf               → PyPDFLoader → text chunks

         ↓ all chunks embedded with OpenAIEmbeddings
   FAISS Vector Store  (persisted to disk)
         ↓
   ConversationalRetrievalChain
   + ConversationBufferMemory  (session-scoped)
         ↓ OpenAI GPT-4o
   Answer + cited sources

Streamlit App
├── Tab 1: Chat          ← ConversationalRetrievalChain
└── Tab 2: Dashboard     ← pandas directly on CSV → Plotly charts
```

---

## Section 1: Data Ingestion & Knowledge Base (`src/ingestion.py`)

### CSV Processing — Two Layers

**Layer 1: Raw row chunks**
Batches of ~50 rows are converted to human-readable text sentences. Example:
> "On 2022-01-03, Widget A sold 871 units in the North region to a 40-year-old female customer with satisfaction score 4.5."

These give the retriever fine-grained row-level context for specific queries.

**Layer 2: Pre-aggregated summary chunks**
Pandas computes and serializes the following as text chunks:
- Sales by product: total, mean, median, std dev
- Sales by region: total, mean, median, std dev
- Sales by month and quarter
- Customer segmentation: age group buckets, gender split, satisfaction scores by segment
- Overall dataset statistics: global median, std dev, min, max

Pre-aggregated summaries ensure stat questions ("What is the median sales for Widget C?") retrieve a precise pre-computed answer rather than attempting to aggregate from raw rows.

### PDF Processing
- Load with `PyPDFLoader`
- Split into 1,000-token chunks with 200-token overlap using `RecursiveCharacterTextSplitter`
- Each chunk tagged with source filename as metadata

### Vector Store
- All chunks (CSV raw + CSV summaries + PDF) embedded with `OpenAIEmbeddings`
- Stored in FAISS, persisted to disk at `vector_store/`
- Loaded from disk on app startup; re-built only if missing or explicitly triggered

---

## Section 2: RAG Chain & Memory (`src/chain.py`)

**Chain:** `ConversationalRetrievalChain` from LangChain
**LLM:** OpenAI GPT-4o
**Memory:** `ConversationBufferMemory` — retains full conversation history within a session; resets on page refresh or when user clears the conversation

**Prompt Engineering:**
A system prompt wraps every query instructing the LLM to:
- Ground answers strictly in retrieved context; do not fabricate data
- Prefer pre-aggregated summary chunks over raw rows for numeric questions
- Cite the source (CSV stats vs. PDF filename) when possible
- Acknowledge uncertainty explicitly rather than hallucinate

**Retriever:** Top-k similarity search (k=5) over the FAISS store, returning chunks from both CSV and PDF sources as relevant.

---

## Section 3: Dashboard (`src/dashboard.py`)

Four Plotly charts computed directly from the CSV via pandas (bypasses RAG for numeric precision):

1. **Sales over time** — line chart aggregated by month
2. **Product performance** — bar chart of total and mean sales per product
3. **Regional breakdown** — bar chart of total sales by region
4. **Customer demographics** — age histogram + gender split with average satisfaction score overlay

---

## Section 4: Streamlit UI (`app.py`)

**Tab 1: Chat**
- Conversation thread rendered as user/assistant message bubbles
- Text input fixed at the bottom; submitting runs the `ConversationalRetrievalChain`
- Session memory stored in `st.session_state`; "Clear Conversation" button resets memory and history
- Source documents shown in a collapsible expander beneath each assistant response

**Tab 2: Dashboard**
- Displays all four Plotly charts from `src/dashboard.py`
- Computed on load directly from the CSV; no LLM involvement

---

## Section 5: Evaluation (`eval/`)

### `eval/generate_eval_set.py`
Generates ~20 Q&A pairs:
- ~15 CSV-based: questions derived from known pandas-computable facts (e.g. "What is the total sales for Widget A?") with ground truth answers computed directly from the data
- ~5 PDF-based: questions about domain concepts from the PDFs with manually written expected answers

Output: `eval/eval_set.json`

### `eval/run_evaluation.py`
- Loads `eval/eval_set.json`
- Runs each question through the RAG chain (without conversational memory — each query independent)
- Scores responses using `QAEvalChain`
- Outputs a summary report: overall % correct and per-question verdict

---

## Project Structure

```
insight-forge/
├── data/
│   ├── sales_data.csv
│   └── pdf/
├── docs/
│   └── superpowers/specs/
├── src/
│   ├── ingestion.py       # CSV + PDF → FAISS vector store
│   ├── chain.py           # ConversationalRetrievalChain + memory setup
│   └── dashboard.py       # Pandas aggregations + Plotly chart functions
├── eval/
│   ├── generate_eval_set.py
│   ├── run_evaluation.py
│   └── eval_set.json      # generated; not committed
├── vector_store/          # persisted FAISS index; not committed
├── app.py                 # Streamlit entry point
└── requirements.txt
```

---

## Coverage Against Original Plan

| Plan Item | Status | Notes |
|---|---|---|
| Data preparation | ✅ | CSV + PDF ingestion, analysis-focused |
| Knowledge base creation | ✅ | FAISS vector store with metadata |
| Sales performance by time period | ✅ | Dashboard chart + RAG summary chunks |
| Product & regional analysis | ✅ | Dashboard charts + RAG summary chunks |
| Customer segmentation by demographics | ✅ | Age/gender chart + RAG summary chunks |
| Statistical measures (median, std dev) | ✅ | Pre-aggregated pandas summaries embedded |
| Custom retriever | ✅ | Pre-aggregated stats as retrieval-optimized text |
| Prompt engineering | ✅ | System prompt for grounding + source citation |
| Chain prompts | ✅ | ConversationalRetrievalChain |
| RAG system setup | ✅ | FAISS + OpenAIEmbeddings |
| Memory integration | ✅ | ConversationBufferMemory (session-scoped) |
| QAEvalChain evaluation | ✅ | Offline eval script with generated eval set |
| Data visualizations (all 4 types) | ✅ | Plotly charts in Dashboard tab |
| Streamlit UI | ✅ | Two-tab layout: Chat + Dashboard |

---

## Dependencies

```
openai
langchain
langchain-openai
langchain-community
faiss-cpu
pypdf
streamlit
pandas
plotly
python-dotenv
```
