"""Unit tests for finance_dashboard.data_processing.

These tests exercise a subset of the data processing functions to
ensure they behave correctly on small, controlled inputs.  They are
intended to serve as a starting point; contributors should add more
tests as new analytical functions are introduced.
"""

from __future__ import annotations

import pandas as pd

from finance_dashboard import data_processing as dp


def test_detect_date_column() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "Amount": [100, 200, 150],
        }
    )
    assert dp.detect_date_column(df) == "Date"


def test_detect_numeric_columns() -> None:
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1.0, 2.0, 3.5], "C": ["x", "y", "z"]})
    assert sorted(dp.detect_numeric_columns(df)) == ["A", "B"]


def test_aggregate_by_period() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2021-01-01", "2021-01-15", "2021-02-01"]),
            "Sales": [100, 200, 300],
        }
    )
    result = dp.aggregate_by_period(df, "Date", "Sales", freq="M")
    # There should be two rows: January and February
    assert len(result) == 2
    # Sum for January should be 300
    january_total = result.loc[result.index[0], "Sales"]
    assert january_total == 300


def test_compute_financial_ratios() -> None:
    # Construct a minimal income statement/balance sheet
    data = {
        "Current Assets": [500],
        "Current Liabilities": [250],
        "Cash": [100],
        "Accounts Receivable": [150],
        "Marketable Securities": [50],
        "Total Revenue": [1000],
        "Net Income": [200],
        "Total Assets": [2000],
        "Total Liabilities": [800],
        "Total Equity": [1200],
        "Cost of Goods Sold": [600],
        "Inventory": [100],
        "Gross Profit": [400],
    }
    df = pd.DataFrame(data)
    ratios = dp.compute_financial_ratios(df)
    # Check that several key ratios exist
    assert "Current Ratio" in ratios
    assert "Quick Ratio" in ratios
    assert "Net Profit Margin" in ratios
    assert "Debt to Equity" in ratios
    assert "Gross Margin" in ratios


def test_horizontal_analysis() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime([
                "2021-01-01",
                "2021-02-01",
                "2021-03-01",
                "2021-04-01",
            ]),
            "Metric": [100, 110, 120, 132],
        }
    )
    ha = dp.horizontal_analysis(df, "Date", ["Metric"], freq="M")
    # The first value should be NaN (no prior period), second should be 0.10
    assert pd.isna(ha.iloc[0, 0])
    assert abs(ha.iloc[1, 0] - 0.10) < 1e-6


def test_vertical_analysis() -> None:
    df = pd.DataFrame(
        {
            "Revenue": [100, 200],
            "COGS": [40, 80],
            "Expenses": [30, 60],
        }
    )
    va = dp.vertical_analysis(df, ["Revenue", "COGS", "Expenses"])
    # Each row should sum to 1
    assert all(abs(row.sum() - 1.0) < 1e-6 for _, row in va.iterrows())


def test_dupont_analysis() -> None:
    df = pd.DataFrame(
        {
            "Net Income": [200],
            "Total Revenue": [1000],
            "Total Assets": [2000],
            "Total Equity": [1000],
        }
    )
    result = dp.dupont_analysis(df)
    # Check that all components are present and the ROE equals the product
    assert set(result.keys()) == {"Net Profit Margin", "Asset Turnover", "Financial Leverage", "Return on Equity"}
    net_margin = result["Net Profit Margin"]
    asset_turnover = result["Asset Turnover"]
    leverage = result["Financial Leverage"]
    roe = result["Return on Equity"]
    assert abs(roe - (net_margin * asset_turnover * leverage)) < 1e-6
