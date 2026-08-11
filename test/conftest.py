# tests/conftest.py

import pytest
import pandas as pd


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small, known DataFrame used across many tests."""
    return pd.DataFrame({
        "region": ["North", "South", "North", "North"],
        "product": ["Widget", "Gadget", "Widget", "Widget"],
        "revenue": [120.50, 89.99, None, 120.50],
        "order_date": ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-15"],
    })


@pytest.fixture
def sample_csv_bytes() -> bytes:
    """Raw CSV bytes matching sample_df, for testing the loader directly."""
    csv_content = (
        "region,product,revenue,order_date\n"
        "North,Widget,120.50,2024-01-15\n"
        "South,Gadget,89.99,2024-01-16\n"
        "North,Widget,,2024-01-17\n"
        "North,Widget,120.50,2024-01-15\n"
    )
    return csv_content.encode("utf-8")