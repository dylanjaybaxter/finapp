"""Largest Expenses analysis workspace."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure package imports resolve when run as a Streamlit page
PARENT = Path(__file__).parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from shared_sidebar import render_shared_sidebar, apply_filters
from personal_finance_analytics import PersonalFinanceAnalytics


def main() -> None:
    st.set_page_config(page_title="Largest Expenses", page_icon="💸", layout="wide")
    sidebar_data = render_shared_sidebar()
    base_df = sidebar_data['base_df']
    filters = sidebar_data.get('filters', {})
    filtered_df = apply_filters(base_df, filters)

    st.header("💸 Largest Expenses Intelligence")
    st.caption("Deep-dive into the biggest outflows during the selected period with interactive tools to understand sources and trends.")

    if filtered_df.empty:
        st.info("Load transactions or widen your filters to analyze expenses.")
        return

    analytics = PersonalFinanceAnalytics(filtered_df)
    expense_df = analytics._expense_rows(filtered_df)
    if expense_df.empty:
        st.success("No expense transactions found in this time window.")
        return

    expense_df = _prepare_expense_frame(expense_df)
    _render_summary(expense_df)
    _render_analysis_tabs(expense_df)
    _render_investigator(expense_df)


def _prepare_expense_frame(expense_df: pd.DataFrame) -> pd.DataFrame:
    df = expense_df.copy()
    df['AbsAmount'] = df['Amount'].abs()
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    df['Payee'] = df.get('Description', '').astype(str)
    df['Category'] = df.get('Category', 'Uncategorized').fillna('Uncategorized')
    df = df.sort_values(['AbsAmount'], ascending=False).reset_index(drop=True)
    return df


def _render_summary(expense_df: pd.DataFrame) -> None:
    total_spend = expense_df['AbsAmount'].sum()
    median_spend = expense_df['AbsAmount'].median()
    avg_spend = expense_df['AbsAmount'].mean()
    unique_payees = expense_df['Payee'].nunique()
    top_category = (
        expense_df.groupby('Category')['AbsAmount'].sum().idxmax()
        if not expense_df.empty else 'N/A'
    )
    cols = st.columns(4)
    cols[0].metric("Total spend", f"${total_spend:,.2f}")
    cols[1].metric("Avg expense", f"${avg_spend:,.2f}")
    cols[2].metric("Median expense", f"${median_spend:,.2f}")
    cols[3].metric("Top category", top_category)


def _render_analysis_tabs(expense_df: pd.DataFrame) -> None:
    st.subheader("Analytics")
    top_n = st.slider("Top N expenses", min_value=5, max_value=100, value=20, step=5)
    trimmed = expense_df.head(top_n)

    chart_tab, category_tab, trend_tab = st.tabs([
        "🔝 High-value grid",
        "🧩 Category mix",
        "📈 Trend lines",
    ])

    with chart_tab:
        st.caption("Sort and download the highest expenses in the selected window.")
        display_cols = ['Transaction Date', 'Payee', 'Category', 'AbsAmount']
        st.data_editor(
            trimmed[display_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                'AbsAmount': st.column_config.NumberColumn("Amount ($)", format="$%0.2f"),
                'Transaction Date': st.column_config.DateColumn("Date"),
            },
            key='largest_expense_table'
        )
        st.download_button(
            "Download all expenses as CSV",
            data=expense_df[display_cols].to_csv(index=False).encode('utf-8'),
            file_name="largest_expenses.csv",
            use_container_width=True,
        )

    with category_tab:
        cat_agg = (
            expense_df.groupby('Category')['AbsAmount']
            .sum()
            .reset_index()
            .sort_values('AbsAmount', ascending=False)
        )
        fig = px.treemap(
            cat_agg,
            path=['Category'],
            values='AbsAmount',
            color='AbsAmount',
            title='Category share of spend',
            color_continuous_scale='YlOrRd'
        )
        st.plotly_chart(fig, use_container_width=True)
        payee_bar = (
            trimmed.groupby('Payee')['AbsAmount']
            .sum()
            .reset_index()
            .sort_values('AbsAmount', ascending=False)
        )
        st.plotly_chart(
            px.bar(payee_bar, x='Payee', y='AbsAmount', title='Top payees', labels={'AbsAmount': 'Amount ($)'}),
            use_container_width=True
        )

    with trend_tab:
        timeline = expense_df.copy()
        if 'Transaction Date' in timeline:
            timeline = timeline.dropna(subset=['Transaction Date'])
        timeline['Month'] = timeline['Transaction Date'].dt.to_period('M').astype(str)
        by_month = timeline.groupby('Month')['AbsAmount'].sum().reset_index()
        trend_fig = px.line(by_month, x='Month', y='AbsAmount', title='Monthly spend trend')
        trend_fig.update_xaxes(type='category')
        st.plotly_chart(trend_fig, use_container_width=True)

        category_trend = (
            timeline.groupby(['Month', 'Category'])['AbsAmount']
            .sum()
            .reset_index()
        )
        cat_fig = px.area(
            category_trend,
            x='Month',
            y='AbsAmount',
            color='Category',
            title='Category trend',
            groupnorm='fraction'
        )
        cat_fig.update_xaxes(type='category')
        st.plotly_chart(cat_fig, use_container_width=True)


def _render_investigator(expense_df: pd.DataFrame) -> None:
    st.subheader("Expense investigator")
    payees = expense_df['Payee'].dropna().unique().tolist()
    default_payee = payees[0] if payees else None
    selected_payee = st.selectbox("Select payee", options=payees, index=0 if default_payee else None)
    if not selected_payee:
        st.info("No payees available.")
        return
    subset = expense_df[expense_df['Payee'] == selected_payee]
    total = subset['AbsAmount'].sum()
    count = len(subset)
    first_seen = subset['Transaction Date'].min()
    last_seen = subset['Transaction Date'].max()

    cols = st.columns(4)
    cols[0].metric("Spend", f"${total:,.2f}")
    cols[1].metric("Payments", f"{count}")
    cols[2].metric("First seen", first_seen.strftime('%Y-%m-%d') if pd.notna(first_seen) else '-')
    cols[3].metric("Last seen", last_seen.strftime('%Y-%m-%d') if pd.notna(last_seen) else '-')

    st.caption("Payment history")
    history = subset[['Transaction Date', 'Category', 'AbsAmount', 'Memo']].copy()
    history = history.sort_values('Transaction Date')
    hist_fig = px.bar(history, x='Transaction Date', y='AbsAmount', title=f"Timeline for {selected_payee}")
    st.plotly_chart(hist_fig, use_container_width=True)
    st.dataframe(
        history.rename(columns={'AbsAmount': 'Amount ($)'}),
        hide_index=True,
        use_container_width=True
    )


if __name__ == '__main__':
    main()
