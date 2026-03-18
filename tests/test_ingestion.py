import os
import pandas as pd
import pytest
from src.ingestion import rows_to_chunks, build_summary_chunks, load_pdf_chunks


# ── Task 2: rows_to_chunks ───────────────────────────────────────────────────

def test_rows_to_chunks_sentence_format(sample_df):
    chunks = rows_to_chunks(sample_df)
    # 5 rows / 50 per chunk = 1 chunk
    assert len(chunks) == 1
    doc = chunks[0]
    assert "Widget A" in doc.page_content
    assert "North" in doc.page_content
    assert "871" in doc.page_content
    assert doc.metadata["chunk_type"] == "rows"

def test_rows_to_chunks_batches_correctly():
    """101 rows should produce 3 chunks: 50, 50, 1."""
    big_df = pd.DataFrame({
        "Date": pd.to_datetime(["2022-01-01"] * 101),
        "Product": ["Widget A"] * 101,
        "Region": ["North"] * 101,
        "Sales": [500] * 101,
        "Customer_Age": [30] * 101,
        "Customer_Gender": ["Male"] * 101,
        "Customer_Satisfaction": [4.0] * 101,
    })
    chunks = rows_to_chunks(big_df)
    assert len(chunks) == 3


# ── Task 3: build_summary_chunks ─────────────────────────────────────────────

def test_build_summary_chunks_returns_documents(sample_df):
    chunks = build_summary_chunks(sample_df)
    assert len(chunks) >= 5  # product, region, monthly, segmentation, overall
    for doc in chunks:
        assert doc.metadata["chunk_type"] == "summary"
        assert len(doc.page_content) > 10

def test_build_summary_chunks_product_stats(sample_df):
    chunks = build_summary_chunks(sample_df)
    product_chunk = next(c for c in chunks if "product" in c.page_content.lower() and "widget" in c.page_content.lower())
    # Widget A: sales 871 + 464 = 1335
    assert "Widget A" in product_chunk.page_content
    assert "1335" in product_chunk.page_content

def test_build_summary_chunks_region_stats(sample_df):
    chunks = build_summary_chunks(sample_df)
    region_chunk = next(c for c in chunks if "region" in c.page_content.lower() and "north" in c.page_content.lower())
    assert "North" in region_chunk.page_content

def test_build_summary_chunks_overall_stats(sample_df):
    chunks = build_summary_chunks(sample_df)
    overall_chunk = next(c for c in chunks if "overall" in c.page_content.lower() or "global" in c.page_content.lower())
    assert "median" in overall_chunk.page_content.lower()
    assert "std" in overall_chunk.page_content.lower() or "standard deviation" in overall_chunk.page_content.lower()


# ── Task 4: load_pdf_chunks ───────────────────────────────────────────────────

def test_load_pdf_chunks_returns_documents():
    pdf_dir = os.path.join(os.path.dirname(__file__), "..", "data", "pdf")
    chunks = load_pdf_chunks(pdf_dir)
    assert len(chunks) > 0
    for doc in chunks:
        assert "source" in doc.metadata
        assert len(doc.page_content.strip()) > 0
