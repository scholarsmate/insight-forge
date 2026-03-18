import pytest
import plotly.graph_objects as go
from src.visualizations import (
    sales_trend_chart,
    product_performance_chart,
    regional_analysis_chart,
    customer_demographics_chart,
)


def test_sales_trend_chart_returns_figure(sample_df):
    fig = sales_trend_chart(sample_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_product_performance_chart_returns_figure(sample_df):
    fig = product_performance_chart(sample_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_regional_analysis_chart_returns_figure(sample_df):
    fig = regional_analysis_chart(sample_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_customer_demographics_chart_returns_figure_with_two_traces(sample_df):
    fig = customer_demographics_chart(sample_df)
    assert isinstance(fig, go.Figure)
    # Must have at least 2 traces: one pie (age groups) + one bar (satisfaction by gender)
    assert len(fig.data) >= 2
