from __future__ import annotations
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

BATCH_SIZE = 50
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.csv")
PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pdf")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store")


def _row_to_sentence(row: pd.Series) -> str:
    return (
        f"On {row['Date'].strftime('%Y-%m-%d')}, {row['Product']} sold {int(row['Sales'])} units "
        f"in the {row['Region']} region to a {int(row['Customer_Age'])}-year-old "
        f"{row['Customer_Gender'].lower()} customer with satisfaction score "
        f"{round(row['Customer_Satisfaction'], 2)}."
    )


def rows_to_chunks(df: pd.DataFrame) -> list[Document]:
    """Convert DataFrame rows into batches of 50, each batch as one Document.

    Expects df['Date'] to be pre-parsed as datetime (use parse_dates=['Date']).
    """
    chunks = []
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start : start + BATCH_SIZE]
        text = "\n".join(_row_to_sentence(row) for _, row in batch.iterrows())
        chunks.append(Document(page_content=text, metadata={"chunk_type": "rows"}))
    return chunks


def build_summary_chunks(df: pd.DataFrame) -> list[Document]:
    """Compute pre-aggregated stats and return each topic as a Document.

    Expects df['Date'] to be pre-parsed as datetime (use parse_dates=['Date']).
    Each returned Document has metadata chunk_type='summary'.
    """
    chunks = []

    # Sales by product
    prod = df.groupby("Product")["Sales"].agg(["sum", "mean", "median", "std"]).round(2)
    lines = ["Sales statistics by product:"]
    for product, row in prod.iterrows():
        lines.append(
            f"  {product}: total={int(row['sum'])}, mean={row['mean']}, "
            f"median={row['median']}, std={row['std']}"
        )
    chunks.append(Document(page_content="\n".join(lines), metadata={"chunk_type": "summary"}))

    # Sales by region
    reg = df.groupby("Region")["Sales"].agg(["sum", "mean", "median", "std"]).round(2)
    lines = ["Sales statistics by region:"]
    for region, row in reg.iterrows():
        lines.append(
            f"  {region}: total={int(row['sum'])}, mean={row['mean']}, "
            f"median={row['median']}, std={row['std']}"
        )
    chunks.append(Document(page_content="\n".join(lines), metadata={"chunk_type": "summary"}))

    # Sales by month
    df2 = df.copy()
    df2["YearMonth"] = df2["Date"].dt.to_period("M").astype(str)
    monthly = df2.groupby("YearMonth")["Sales"].agg(["sum", "mean"]).round(2)
    lines = ["Monthly sales summary:"]
    for period, row in monthly.iterrows():
        lines.append(f"  {period}: total={int(row['sum'])}, mean={row['mean']}")
    chunks.append(Document(page_content="\n".join(lines), metadata={"chunk_type": "summary"}))

    # Customer segmentation
    df2["AgeGroup"] = pd.cut(df2["Customer_Age"], bins=[0, 30, 45, 100], labels=["18-30", "31-45", "46+"])
    age_seg = df2.groupby("AgeGroup", observed=True)["Sales"].agg(["sum", "mean"]).round(2)
    gender_seg = df2.groupby("Customer_Gender")["Customer_Satisfaction"].mean().round(2)
    lines = ["Customer segmentation:"]
    lines.append("  By age group:")
    for group, row in age_seg.iterrows():
        lines.append(f"    {group}: total sales={int(row['sum'])}, mean sales={row['mean']}")
    lines.append("  Average satisfaction by gender:")
    for gender, sat in gender_seg.items():
        lines.append(f"    {gender}: {sat}")
    chunks.append(Document(page_content="\n".join(lines), metadata={"chunk_type": "summary"}))

    # Overall stats
    sales = df["Sales"]
    lines = [
        "Overall dataset statistics:",
        f"  Total rows: {len(df)}",
        f"  Global median sales: {sales.median()}",
        f"  Global mean sales: {round(sales.mean(), 2)}",
        f"  Standard deviation: {round(sales.std(), 2)}",
        f"  Min sales: {sales.min()}",
        f"  Max sales: {sales.max()}",
    ]
    chunks.append(Document(page_content="\n".join(lines), metadata={"chunk_type": "summary"}))

    return chunks


def load_pdf_chunks(pdf_dir: str) -> list[Document]:
    """Load all PDFs from a directory and split into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []
    for filename in os.listdir(pdf_dir):
        if not filename.endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, filename)
        loader = PyPDFLoader(path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata["source"] = filename
        all_chunks.extend(chunks)
    return all_chunks


def build_vector_store(csv_path: str = CSV_PATH, pdf_dir: str = PDF_DIR) -> FAISS:
    """Build FAISS vector store from CSV and PDF chunks. Persists to disk."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    all_docs = rows_to_chunks(df) + build_summary_chunks(df) + load_pdf_chunks(pdf_dir)
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(all_docs, embeddings)
    store.save_local(VECTOR_STORE_PATH)
    print(f"Vector store built: {len(all_docs)} documents indexed at {VECTOR_STORE_PATH}")
    return store


def load_vector_store() -> FAISS:
    """Load persisted FAISS vector store from disk."""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)


def get_vector_store() -> FAISS:
    """Return vector store: load from disk or build if missing/REBUILD_INDEX=1."""
    rebuild = os.getenv("REBUILD_INDEX", "0") == "1"
    if rebuild or not os.path.exists(VECTOR_STORE_PATH):
        return build_vector_store()
    return load_vector_store()


if __name__ == "__main__":
    # CLI entry point: python src/ingestion.py
    # Set REBUILD_INDEX=1 to force a rebuild even if vector_store/ exists.
    build_vector_store()
