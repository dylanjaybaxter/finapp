import pandas as pd
from finance_dashboard.personal_finance_analytics import PersonalFinanceAnalytics


def sample_df():
    return pd.DataFrame([
        {'Transaction Date': '2024-01-01', 'Description': 'Rent Apts', 'Amount': -1500, 'Category': 'Housing'},
        {'Transaction Date': '2024-01-02', 'Description': 'Salary', 'Amount': 5000, 'Category': 'Income'},
        {'Transaction Date': '2024-01-03', 'Description': 'Transfer', 'Amount': -200, 'Category': 'Transfer'},
    ])


def test_expense_rows_include_rent():
    analytics = PersonalFinanceAnalytics(sample_df())
    expenses = analytics._expense_rows()
    assert any(expenses['Description'].str.contains('Rent')), "Rent should be treated as expense"


def test_expense_rows_handles_missing_flow_category():
    df = sample_df()
    analytics = PersonalFinanceAnalytics(sample_df())
    expenses = analytics._expense_rows(df)
    assert not expenses.empty


def test_period_summary_uses_flow_categories():
    analytics = PersonalFinanceAnalytics(sample_df())
    summary = analytics.calculate_period_summary()
    assert summary['income'] > 0
    assert summary['expenses'] > 0
