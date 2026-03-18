import pytest
import pandas as pd

@pytest.fixture
def sample_df():
    """Small 5-row DataFrame for fast tests — does not require the real CSV."""
    return pd.DataFrame({
        "Date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2022-02-01", "2022-02-02"]),
        "Product": ["Widget A", "Widget B", "Widget A", "Widget C", "Widget B"],
        "Region": ["North", "South", "East", "West", "North"],
        "Sales": [871, 850, 464, 786, 920],
        "Customer_Age": [40, 29, 31, 26, 45],
        "Customer_Gender": ["Female", "Male", "Male", "Male", "Female"],
        "Customer_Satisfaction": [4.55, 3.37, 4.56, 2.87, 4.10],
    })
