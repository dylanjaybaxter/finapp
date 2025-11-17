"""Analytics utilities for financial calculations.

This module provides analytics helpers for income calculations and
other financial metrics used across pages.
"""

from .income import (
    income_series_complete,
    three_month_avg_income,
)

__all__ = [
    'income_series_complete',
    'three_month_avg_income',
]

