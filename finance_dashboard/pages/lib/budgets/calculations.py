"""Budget calculation and suggestion utilities.

This module provides functions for calculating budget suggestions, aggregating
budgets by groups, calculating performance metrics, and generating budget
analytics dataframes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

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

from .categorization import infer_group


def suggested_amounts(
    categories: pd.Series,
    analytics: Optional[PersonalFinanceAnalytics],
    detailed: bool,
    groups_map: Dict[str, List[str]],
    target_map: Dict[str, float]
) -> Dict[str, float]:
    """Calculate suggested budget amounts for categories.
    
    Args:
        categories: Series of category names to calculate suggestions for
        analytics: Analytics object with spending data
        detailed: If True, use historical spending to proportionally distribute group budgets
        groups_map: Dictionary mapping group names to category lists
        target_map: Dictionary mapping group names to target percentages (as decimals)
        
    Returns:
        Dictionary mapping category names to suggested budget amounts
        
    Example:
        >>> categories = pd.Series(['Groceries', 'Dining'])
        >>> groups = {'Essentials (Needs)': ['Groceries'], 'Lifestyle (Wants)': ['Dining']}
        >>> targets = {'Essentials (Needs)': 0.5, 'Lifestyle (Wants)': 0.3}
        >>> suggestions = suggested_amounts(categories, analytics, True, groups, targets)
    """
    suggestions: Dict[str, float] = {}
    if analytics is None or categories.empty:
        return suggestions
    
    income = weighted_income_estimate(analytics)
    if income <= 0:
        return suggestions
    
    spending = analytics.calculate_category_spending()
    category_totals = spending['Total_Spent'] if not spending.empty else pd.Series(dtype=float)
    grouped_spend = (
        category_totals.groupby(category_totals.index.map(lambda c: infer_group(c, groups_map))).sum()
        if not category_totals.empty
        else pd.Series(dtype=float)
    )
    
    for category in categories:
        if pd.isna(category):
            continue
        
        group = infer_group(category, groups_map)
        group_target = income * target_map.get(group, 0.3)
        
        if detailed and not category_totals.empty and category in category_totals:
            group_spend = grouped_spend.get(group, group_target)
            share = (
                (category_totals.get(category) / group_spend)
                if group_spend
                else 1 / max(1, len(groups_map.get(group, [])))
            )
            suggestions[category] = float(group_target * share)
        else:
            suggestions[category] = float(group_target / max(1, len(groups_map.get(group, []))))
    
    return suggestions


def suggested_group_amounts(
    analytics: Optional[PersonalFinanceAnalytics],
    target_map: Dict[str, float]
) -> Dict[str, float]:
    """Calculate suggested budget amounts for groups.
    
    Args:
        analytics: Analytics object with spending data
        target_map: Dictionary mapping group names to target percentages (as decimals)
        
    Returns:
        Dictionary mapping group names to suggested budget amounts
    """
    if analytics is None:
        return {group: 0.0 for group in target_map}
    
    income = weighted_income_estimate(analytics)
    return {group: income * pct for group, pct in target_map.items()}


def weighted_income_estimate(analytics: PersonalFinanceAnalytics) -> float:
    """Calculate weighted income estimate using recent months with recency weighting.
    
    Uses weights of [0.8, 0.15, 0.05] for the 3 most recent months, giving
    more weight to recent income.
    
    Args:
        analytics: Analytics object with transaction data
        
    Returns:
        Weighted average monthly income estimate
    """
    data = analytics.data.copy()
    if data.empty:
        return 0.0
    
    data['Month'] = data['Transaction Date'].dt.to_period('M')
    pos = data[data['Amount'] > 0].copy()
    if pos.empty:
        return 0.0
    
    current_month = pd.Period(pd.Timestamp.utcnow().date(), freq='M')
    pos = pos[pos['Month'] < current_month]
    monthly = pos.groupby('Month')['Amount'].sum().sort_index(ascending=False)
    
    weights = [0.8, 0.15, 0.05]
    applicable = monthly.head(3)
    
    if applicable.empty:
        return float(monthly.mean())
    
    applied_weights = weights[:len(applicable)]
    total_weight = sum(applied_weights)
    weighted = sum(val * w for val, w in zip(applicable.values, applied_weights)) / total_weight
    return float(weighted)


def aggregate_budgets_by_group(
    budgets: Dict[str, float],
    groups_map: Dict[str, List[str]]
) -> Dict[str, float]:
    """Aggregate category budgets into group totals.
    
    Args:
        budgets: Dictionary mapping category names to budget amounts
        groups_map: Dictionary mapping group names to category lists
        
    Returns:
        Dictionary mapping group names to total budget amounts
        
    Example:
        >>> budgets = {'Groceries': 500, 'Dining': 200}
        >>> groups = {'Essentials (Needs)': ['Groceries'], 'Lifestyle (Wants)': ['Dining']}
        >>> aggregate_budgets_by_group(budgets, groups)
        {'Essentials (Needs)': 500.0, 'Lifestyle (Wants)': 200.0}
    """
    grouped: Dict[str, float] = {group: 0.0 for group in groups_map}
    for category, amount in budgets.items():
        group = infer_group(category, groups_map)
        grouped[group] = grouped.get(group, 0.0) + amount
    return grouped


def expand_group_budgets(
    group_amounts: Dict[str, float],
    analytics: Optional[PersonalFinanceAnalytics],
    groups: Dict[str, List[str]]
) -> Dict[str, float]:
    """Expand group budget amounts into category budgets based on historical spending.
    
    Args:
        group_amounts: Dictionary mapping group names to budget amounts
        analytics: Analytics object with spending data (used to proportionally distribute)
        groups: Dictionary mapping group names to category lists
        
    Returns:
        Dictionary mapping category names to budget amounts
        
    Example:
        >>> group_amounts = {'Essentials (Needs)': 1000}
        >>> groups = {'Essentials (Needs)': ['Groceries', 'Utilities']}
        >>> expanded = expand_group_budgets(group_amounts, analytics, groups)
    """
    expanded: Dict[str, float] = {}
    spending = analytics.calculate_category_spending() if analytics else pd.DataFrame()
    category_totals = spending['Total_Spent'] if not spending.empty else pd.Series(dtype=float)
    grouped_spend = (
        category_totals.groupby(category_totals.index.map(lambda c: infer_group(c, groups))).sum()
        if not category_totals.empty
        else pd.Series(dtype=float)
    )
    
    for group, amount in group_amounts.items():
        categories = groups.get(group, [])
        if not categories or amount <= 0:
            continue
        
        total_group_spend = grouped_spend.get(group, None) if not grouped_spend.empty else None
        
        for category in categories:
            if total_group_spend and category in category_totals:
                share = (
                    category_totals.get(category) / total_group_spend
                    if total_group_spend
                    else 1 / len(categories)
                )
            else:
                share = 1 / len(categories)
            expanded[category] = expanded.get(category, 0.0) + amount * share
    
    return expanded


def budget_performance_snapshot(
    analytics: Optional[PersonalFinanceAnalytics],
    budgets: Dict[str, float],
    *,
    current_only: bool = False
) -> Tuple[pd.DataFrame, Union[str, int]]:
    """Calculate budget performance snapshot for specified time period.
    
    Args:
        analytics: Analytics object with transaction data
        budgets: Dictionary mapping category names to monthly budget amounts
        current_only: If True, only analyze the most recent month; otherwise all months
        
    Returns:
        Tuple of (performance DataFrame, period label)
        DataFrame columns: Category, Budget/mo, Actual/mo, Variance/mo, Target Total,
                          Actual Total, Variance Total, Percent Used, Status
    """
    if analytics is None or not budgets or analytics.data.empty:
        return pd.DataFrame(), ''
    
    df = analytics.data.copy()
    if df.empty:
        return pd.DataFrame(), ''
    
    df['Month'] = df['Transaction Date'].dt.to_period('M')
    months = sorted(df['Month'].unique())
    if not months:
        return pd.DataFrame(), ''
    
    target_months = {months[-1]} if current_only else set(months)
    scoped = df[df['Month'].isin(target_months)]
    months_count = scoped['Month'].nunique()
    
    if months_count == 0:
        return pd.DataFrame(), ''
    
    expense = analytics._expense_rows(scoped)
    if expense.empty:
        return pd.DataFrame(), months_count if not current_only else str(months[-1])
    
    expense['AbsAmount'] = expense['Amount'].abs()
    grouped = expense.groupby('Category')['AbsAmount'].sum()
    
    rows = []
    for category, budget in budgets.items():
        target_total = budget * months_count
        actual_total = grouped.get(category, 0.0)
        actual_per_month = actual_total / months_count if months_count else 0.0
        variance_total = target_total - actual_total
        variance_monthly = budget - actual_per_month
        percent_used = (actual_total / target_total * 100.0) if target_total else None
        status = 'Over' if variance_total < 0 else 'Under'
        
        rows.append({
            'Category': category,
            'Budget/mo': budget,
            'Actual/mo': actual_per_month,
            'Variance/mo': variance_monthly,
            'Target Total': target_total,
            'Actual Total': actual_total,
            'Variance Total': variance_total,
            'Percent Used': percent_used,
            'Status': status,
        })
    
    result = pd.DataFrame(rows)
    label = str(months[-1]) if current_only else months_count
    return result, label


def budget_trend_dataframe(
    analytics: PersonalFinanceAnalytics,
    budgets: Dict[str, float]
) -> pd.DataFrame:
    """Create DataFrame showing budget vs actual spending trend over time.
    
    Args:
        analytics: Analytics object with transaction data
        budgets: Dictionary mapping category names to monthly budget amounts
        
    Returns:
        DataFrame with columns: Month, Amount, Metric (where Metric is 'Actual' or 'Budget')
    """
    history = []
    data = analytics.data.copy()
    if data.empty:
        return pd.DataFrame()
    
    data['Month'] = data['Transaction Date'].dt.to_period('M')
    expense = data[data['Amount'] < 0].copy()
    expense['Amount'] = expense['Amount'].abs()
    grouped = expense.groupby('Month')['Amount'].sum().reset_index()
    grouped['Month'] = grouped['Month'].astype(str)
    
    history.append(pd.DataFrame({
        'Month': grouped['Month'],
        'Amount': grouped['Amount'],
        'Metric': 'Actual'
    }))
    
    monthly_budget = sum(budgets.values())
    budget_line = pd.DataFrame({
        'Month': grouped['Month'],
        'Amount': monthly_budget,
        'Metric': 'Budget'
    })
    history.append(budget_line)
    
    return pd.concat(history, ignore_index=True)


def monthly_income_vs_spend(analytics: PersonalFinanceAnalytics) -> pd.DataFrame:
    """Create DataFrame showing monthly income vs spending.
    
    Args:
        analytics: Analytics object with transaction data
        
    Returns:
        DataFrame with columns: Month, Income, Spending, Savings
    """
    data = analytics.data.copy()
    if data.empty:
        return pd.DataFrame()
    
    data['Month'] = data['Transaction Date'].dt.to_period('M')
    income_by_month = data[data['Amount'] > 0].groupby('Month')['Amount'].sum()
    expense = analytics._expense_rows(data)
    expense['AbsAmount'] = expense['Amount'].abs()
    expense['Month'] = expense['Transaction Date'].dt.to_period('M')
    spending_by_month = expense.groupby('Month')['AbsAmount'].sum()
    
    months = sorted(set(income_by_month.index) | set(spending_by_month.index))
    rows = []
    for month in months:
        income = float(income_by_month.get(month, 0.0))
        spending = float(spending_by_month.get(month, 0.0))
        savings = income - spending
        rows.append({
            'Month': str(month),
            'Income': income,
            'Spending': spending,
            'Savings': savings,
        })
    
    return pd.DataFrame(rows)


def monthly_group_targets_actuals(
    analytics: PersonalFinanceAnalytics,
    targets_map: Dict[str, float],
    groups_map: Dict[str, List[str]]
) -> pd.DataFrame:
    """Create DataFrame showing monthly group targets vs actuals.
    
    Args:
        analytics: Analytics object with transaction data
        targets_map: Dictionary mapping group names to target percentages (as decimals)
        groups_map: Dictionary mapping group names to category lists
        
    Returns:
        DataFrame with columns: Month, Group, Target ($), Actual ($), Target % Income,
                                Actual % Income, Status, Variance ($)
    """
    data = analytics.data.copy()
    if data.empty:
        return pd.DataFrame()
    
    data['Month'] = data['Transaction Date'].dt.to_period('M')
    income_by_month = data[data['Amount'] > 0].groupby('Month')['Amount'].sum()
    expense = analytics._expense_rows(data)
    expense['AbsAmount'] = expense['Amount'].abs()
    expense['Month'] = expense['Transaction Date'].dt.to_period('M')
    
    actual_by_pair: Dict[Tuple, float] = {}
    for _, row in expense.iterrows():
        category = row['Category']
        group = (
            next((g for g, cats in groups_map.items() if category in cats), next(iter(groups_map.keys()), 'Other'))
            if groups_map
            else 'Other'
        )
        key = (row['Month'], group)
        actual_by_pair[key] = actual_by_pair.get(key, 0.0) + float(row['AbsAmount'])
    
    months = sorted(set(income_by_month.index) | {pair[0] for pair in actual_by_pair.keys()})
    rows = []
    
    for month in months:
        income = float(income_by_month.get(month, 0.0))
        for group, target_pct in targets_map.items():
            target_dollars = income * target_pct
            actual = actual_by_pair.get((month, group), 0.0)
            status = (
                'No Income' if income == 0
                else ('Over' if target_dollars and actual > target_dollars else 'Under')
            )
            rows.append({
                'Month': str(month),
                'Group': group,
                'Target ($)': target_dollars,
                'Actual ($)': actual,
                'Target % Income': target_pct * 100,
                'Actual % Income': (actual / income * 100) if income else 0.0,
                'Status': status,
                'Variance ($)': target_dollars - actual,
            })
    
    return pd.DataFrame(rows)


def category_percent_income(
    analytics: PersonalFinanceAnalytics,
    income_series: pd.Series
) -> pd.DataFrame:
    """Calculate category spending as percentage of income by month.
    
    Args:
        analytics: Analytics object with transaction data
        income_series: Series with month index and income values
        
    Returns:
        DataFrame with columns: Month, Category, AbsAmount, Income, Percent
    """
    data = analytics.data.copy()
    if data.empty or income_series.empty:
        return pd.DataFrame()
    
    data['Month'] = data['Transaction Date'].dt.to_period('M').astype(str)
    months = income_series.index.tolist()
    income_df = income_series.rename('Income').reset_index().rename(columns={'index': 'Month'})
    
    expense = data[(data['Amount'] < 0) & (data['Month'].isin(months))].copy()
    expense['AbsAmount'] = expense['Amount'].abs()
    cat_month = expense.groupby(['Month', 'Category'])['AbsAmount'].sum().reset_index()
    
    totals = cat_month.groupby('Month')['AbsAmount'].sum()
    savings = income_df.copy()
    savings['AbsAmount'] = savings['Income'] - savings['Month'].map(totals).fillna(0)
    savings['AbsAmount'] = savings['AbsAmount'].clip(lower=0)
    savings['Category'] = 'Savings'
    cat_month = pd.concat([cat_month, savings[['Month', 'Category', 'AbsAmount']]], ignore_index=True)
    
    merged = cat_month.merge(income_df, on='Month', how='inner')
    merged = merged[merged['Income'] > 0]
    merged['Percent'] = (merged['AbsAmount'] / merged['Income']) * 100
    merged = merged[merged['Category'].str.lower() != 'income']
    merged['Month'] = pd.Categorical(merged['Month'], categories=months, ordered=True)
    
    return merged.sort_values(['Month', 'Category'])


def percent_income_summary(
    df: pd.DataFrame,
    income_lookup: pd.Series
) -> pd.DataFrame:
    """Create summary of spending as percentage of income.
    
    Args:
        df: DataFrame from category_percent_income (with Month, Category, AbsAmount, Percent)
        income_lookup: Series with month index and income values
        
    Returns:
        DataFrame with columns: Category, Amount, Percent
    """
    if df.empty:
        return pd.DataFrame(columns=['Category', 'Amount', 'Percent'])
    
    months_used = df['Month'].unique()
    monthly_income = income_lookup[income_lookup.index.isin(months_used)]
    
    if monthly_income.empty:
        return pd.DataFrame(columns=['Category', 'Amount', 'Percent'])
    
    income_avg = monthly_income.mean()
    if income_avg <= 0:
        return pd.DataFrame(columns=['Category', 'Amount', 'Percent'])
    
    amount_by_category = df.groupby('Category')['AbsAmount'].mean().sort_values(ascending=False)
    summary = amount_by_category.reset_index().rename(columns={'AbsAmount': 'Amount'})
    spending_sum = summary['Amount'].sum()
    savings_amount = max(income_avg - spending_sum, 0.0)
    
    if 'Savings' not in summary['Category'].values or savings_amount > 0:
        summary = pd.concat([
            summary,
            pd.DataFrame({'Category': ['Savings'], 'Amount': [savings_amount]})
        ], ignore_index=True)
    
    summary['Percent'] = (summary['Amount'] / income_avg) * 100
    summary['Percent'] = summary['Percent'].round(1)
    summary['Amount'] = summary['Amount'].round(2)
    
    total_percent = summary['Percent'].sum()
    if abs(total_percent - 100.0) > 0.1:
        diff = 100 - total_percent
        idx = summary['Category'] == 'Savings'
        if idx.any():
            summary.loc[idx, 'Percent'] += diff
    
    return summary

