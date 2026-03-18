from __future__ import annotations
import os
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# --- Caching setup ---
# Import chain module BEFORE patching so we can overwrite its get_vector_store binding.
# @st.cache_resource is used (not @st.cache_data) because FAISS is not serialisable.
import src.ingestion
import src.chain as _chain_module


@st.cache_resource
def _load_store():
    return src.ingestion.get_vector_store()


# Patch chain.py so _build_retriever uses the cached store on every call.
_chain_module.get_vector_store = _load_store

# Now safe to import chain factories (they will pick up the patched name at call time).
from src.chain import build_rag_chain
from src.visualizations import (
    sales_trend_chart,
    product_performance_chart,
    regional_analysis_chart,
    customer_demographics_chart,
)
from src.eval import run_evaluation

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "sales_data.csv")


@st.cache_data
def _load_csv() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, parse_dates=["Date"])


@st.cache_data
def _sales_trend(_df):
    return sales_trend_chart(_df)


@st.cache_data
def _product_perf(_df):
    return product_performance_chart(_df)


@st.cache_data
def _regional(_df):
    return regional_analysis_chart(_df)


@st.cache_data
def _demographics(_df):
    return customer_demographics_chart(_df)


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="InsightForge", page_icon="🔍", layout="wide")
st.title("🔍 InsightForge — AI Business Intelligence Assistant")

df = _load_csv()

tab_viz, tab_chat, tab_eval = st.tabs(["📊 Visualizations", "💬 Chat", "🧪 Eval"])

# ─── Tab 1: Visualizations ───────────────────────────────────────────────────

with tab_viz:
    st.header("Data Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sales Trend Over Time")
        st.plotly_chart(_sales_trend(df), width='stretch')
    with col2:
        st.subheader("Product Performance")
        st.plotly_chart(_product_perf(df), width='stretch')
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Regional Analysis")
        st.plotly_chart(_regional(df), width='stretch')
    with col4:
        st.subheader("Customer Demographics")
        st.plotly_chart(_demographics(df), width='stretch')

# ─── Tab 2: Chat ─────────────────────────────────────────────────────────────

with tab_chat:
    st.header("Chat with InsightForge")

    # Initialize session state on first load only
    if "chain" not in st.session_state:
        st.session_state.chain = build_rag_chain()
        st.session_state.chat_history = []  # list[BaseMessage]
        st.session_state.messages = []
        st.session_state.pending_input = None

    # Example queries — shown only before the first message
    EXAMPLES = [
        "Which product had the highest total sales?",
        "Which region generates the most revenue?",
        "What age group buys the most?",
        "How do male and female customers compare in satisfaction?",
        "What are the main BI approaches discussed in the documents?",
    ]
    if not st.session_state.messages:
        st.caption("Try an example:")
        cols = st.columns(len(EXAMPLES))
        for col, example in zip(cols, EXAMPLES):
            if col.button(example, use_container_width=True):
                st.session_state.pending_input = example

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for doc in msg["sources"]:
                        label = doc.metadata.get("source", doc.metadata.get("chunk_type", "unknown"))
                        st.write(label)
                        st.caption(doc.page_content[:200])

    # Input — typed or from an example button click
    user_input = st.chat_input("Ask a business question...")
    if not user_input and st.session_state.pending_input:
        user_input = st.session_state.pending_input
        st.session_state.pending_input = None
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history,
                })
            answer = result["answer"]
            sources = result.get("context", [])
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for doc in sources:
                        label = doc.metadata.get("source", doc.metadata.get("chunk_type", "unknown"))
                        st.write(label)
                        st.caption(doc.page_content[:200])

        # Update chat history for next turn
        st.session_state.chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=answer),
        ])
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

# ─── Tab 3: Eval ─────────────────────────────────────────────────────────────

with tab_eval:
    st.header("Model Evaluation")
    st.write("Evaluate InsightForge against a set of reference questions and answers using an LLM grader.")

    if st.button("▶ Run Evaluation"):
        with st.spinner("Running evaluation — this may take a minute..."):
            results = run_evaluation(build_rag_chain())

        n_correct = sum(1 for r in results if r["grade"] == "CORRECT")
        total = len(results)
        st.metric("Score", f"{n_correct} / {total}")
        st.dataframe(results, width='stretch')
