from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sales_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart of monthly total sales."""
    df2 = df.copy()
    df2["YearMonth"] = df2["Date"].dt.to_period("M").astype(str)
    monthly = df2.groupby("YearMonth")["Sales"].sum().reset_index()
    fig = go.Figure(
        go.Scatter(x=monthly["YearMonth"], y=monthly["Sales"], mode="lines+markers")
    )
    fig.update_layout(title="Sales Trend Over Time", xaxis_title="Month", yaxis_title="Total Sales")
    return fig


def product_performance_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of total sales by product."""
    prod = df.groupby("Product")["Sales"].sum().reset_index().sort_values("Sales")
    fig = go.Figure(
        go.Bar(x=prod["Sales"], y=prod["Product"], orientation="h")
    )
    fig.update_layout(title="Product Performance", xaxis_title="Total Sales", yaxis_title="Product")
    return fig


def regional_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """Vertical bar chart of total sales by region."""
    reg = df.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
    fig = go.Figure(
        go.Bar(x=reg["Region"], y=reg["Sales"])
    )
    fig.update_layout(title="Regional Analysis", xaxis_title="Region", yaxis_title="Total Sales")
    return fig


def customer_demographics_chart(df: pd.DataFrame) -> go.Figure:
    """Subplots: age group distribution (pie) + avg satisfaction by gender (bar)."""
    df2 = df.copy()
    df2["AgeGroup"] = pd.cut(
        df2["Customer_Age"],
        bins=[0, 30, 45, 100],
        labels=["18-30", "31-45", "46+"],
    )
    # Drop empty categories before building pie
    age_counts = df2["AgeGroup"].value_counts().dropna()
    age_counts = age_counts[age_counts > 0]

    gender_sat = df2.groupby("Customer_Gender")["Customer_Satisfaction"].mean().reset_index()

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("Age Group Distribution", "Avg Satisfaction by Gender"),
    )
    fig.add_trace(
        go.Pie(labels=age_counts.index.tolist(), values=age_counts.values.tolist(), name="Age"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=gender_sat["Customer_Gender"],
            y=gender_sat["Customer_Satisfaction"],
            name="Satisfaction",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(title="Customer Demographics")
    return fig
