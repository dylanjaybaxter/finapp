"""Income calculation and analysis utilities.

This module provides functions for calculating income series and
averages from transaction data.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# Import analytics - handle both package and direct execution
try:
    from ...personal_finance_analytics import PersonalFinanceAnalytics
except ImportError:
    import sys
    from pathlib import Path as _Path
    parent_dir = _Path(__file__).resolve().parents[3]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from finance_dashboard.personal_finance_analytics import PersonalFinanceAnalytics

# Import config for income calculation settings
try:
    from ...config import get_budget_config
except ImportError:
    import sys
    from pathlib import Path as _Path
    parent_dir = _Path(__file__).resolve().parents[3]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from pages.config import get_budget_config


def income_series_complete(analytics: PersonalFinanceAnalytics) -> pd.Series:
    """Calculate complete income series from transaction data, excluding current month.
    
    This function extracts all positive transactions (income), groups them by month,
    and excludes the current month to provide a complete historical view.
    
    Args:
        analytics: Analytics object with transaction data
        
    Returns:
        Series with month strings as index and income totals as values
        
    Example:
        >>> series = income_series_complete(analytics)
        >>> series.head()
        2024-01    5000.0
        2024-02    5200.0
        ...
    """
    data = analytics.data.copy()
    if data.empty:
        return pd.Series(dtype=float)
    
    data['Month'] = data['Transaction Date'].dt.to_period('M')
    pos = data[data['Amount'] > 0].copy()
    
    if pos.empty:
        return pd.Series(dtype=float)
    
    # Exclude current month (incomplete)
    current_month = pd.Period(pd.Timestamp.utcnow().date(), freq='M')
    pos = pos[pos['Month'] < current_month]
    
    income = pos.groupby('Month')['Amount'].sum().sort_index()
    income.index = income.index.astype(str)
    
    return income


def three_month_avg_income(analytics: Optional[PersonalFinanceAnalytics]) -> float:
    """Calculate three-month average income from complete income series.
    
    Uses the most recent 3 complete months (excluding current month) to calculate
    an average income estimate.
    
    Args:
        analytics: Analytics object with transaction data, or None
        
    Returns:
        Average monthly income over the last 3 complete months, or 0.0 if insufficient data
        
    Example:
        >>> avg = three_month_avg_income(analytics)
        >>> print(f"Average monthly income: ${avg:,.2f}")
        Average monthly income: $5,000.00
    """
    if analytics is None or analytics.data.empty:
        return 0.0
    
    series = income_series_complete(analytics)
    if series.empty:
        return 0.0
    
    # Get config for number of months (defaults to 3)
    try:
        config = get_budget_config()
        months = config.get('calculations', {}).get('income_months', 3)
    except:
        months = 3
    
    recent = series.sort_index(ascending=False).head(months)
    return float(recent.mean()) if not recent.empty else 0.0

