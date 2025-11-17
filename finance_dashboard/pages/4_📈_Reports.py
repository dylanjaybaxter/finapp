"""Reports Page - Detailed financial reports and analytics."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import handling for Streamlit pages
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from shared_sidebar import render_shared_sidebar, apply_filters
from personal_finance_analytics import PersonalFinanceAnalytics


def main():
    """Render the Reports page."""
    st.set_page_config(page_title="Reports", page_icon="📈", layout="wide")
    
    # Render shared sidebar
    sidebar_data = render_shared_sidebar()
    base_df = sidebar_data['base_df']
    filters = sidebar_data['filters']
    
    st.header("📊 Reports & Analytics")
    
    if base_df.empty:
        st.warning("No data available for analysis.")
        return
    
    # Apply filters
    filtered_data = apply_filters(base_df, filters)
    
    if filtered_data.empty:
        st.warning("No transactions match the selected filters.")
        return
    
    analytics = PersonalFinanceAnalytics(filtered_data)
    
    # Report type selection
    report_type = st.selectbox(
        "Select Report Type",
        ["Spending Analysis", "Budget Performance", "Financial Health", "Spending Patterns", "Export Data"]
    )
    
    if report_type == "Spending Analysis":
        _render_spending_analysis_report(analytics)
    elif report_type == "Budget Performance":
        _render_budget_performance_report(analytics)
    elif report_type == "Financial Health":
        _render_financial_health_report(analytics)
    elif report_type == "Spending Patterns":
        _render_spending_patterns_report(analytics)
    elif report_type == "Export Data":
        _render_export_data(filtered_data)


def _render_spending_analysis_report(analytics: PersonalFinanceAnalytics):
    """Render detailed spending analysis report."""
    st.subheader("Spending Analysis Report")
    
    # Category spending
    category_spending = analytics.calculate_category_spending()
    st.dataframe(category_spending, use_container_width=True)
    
    # Spending trends
    trends = analytics.calculate_spending_trends()
    if not trends.empty:
        fig = px.line(trends, x='Period', y='Amount', title='Spending Trends Over Time')
        st.plotly_chart(fig, use_container_width=True)


def _render_budget_performance_report(analytics: PersonalFinanceAnalytics):
    """Render budget performance report."""
    st.subheader("Budget Performance Report")
    
    budgets = st.session_state.get('budgets', {})
    if not budgets:
        st.info("No budgets set. Create budgets to see performance reports.")
        return
    
    budget_performance = analytics.calculate_budget_performance(budgets)
    if not budget_performance.empty:
        st.dataframe(budget_performance, use_container_width=True)
        
        # Budget performance chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Budget', x=budget_performance['Category'], y=budget_performance['Budget']))
        fig.add_trace(go.Bar(name='Actual', x=budget_performance['Category'], y=budget_performance['Actual']))
        fig.update_layout(title='Budget vs Actual Spending', barmode='group')
        st.plotly_chart(fig, use_container_width=True)


def _render_financial_health_report(analytics: PersonalFinanceAnalytics):
    """Render financial health report."""
    st.subheader("Financial Health Report")
    
    health_metrics = analytics.calculate_financial_health_metrics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("12-Month Income", f"${health_metrics['total_income_12m']:,.2f}")
        st.metric("12-Month Expenses", f"${health_metrics['total_expenses_12m']:,.2f}")
    
    with col2:
        st.metric("Monthly Income Avg", f"${health_metrics['monthly_income_avg']:,.2f}")
        st.metric("Monthly Expenses Avg", f"${health_metrics['monthly_expenses_avg']:,.2f}")
    
    with col3:
        st.metric("Savings Rate", f"{health_metrics['savings_rate']:.1f}%")
        st.metric("Monthly Savings Avg", f"${health_metrics['monthly_savings_avg']:,.2f}")


def _render_spending_patterns_report(analytics: PersonalFinanceAnalytics):
    """Render spending patterns report."""
    st.subheader("Spending Patterns Report")
    
    patterns = analytics.calculate_spending_patterns()
    
    st.write(f"**Total Transactions:** {patterns['total_transactions']}")
    st.write(f"**Average Transaction:** ${patterns.get('avg_transaction', 0):,.2f}")
    st.write(f"**Most Active Day:** {patterns.get('most_active_day', 'N/A')}")


def _render_export_data(data: pd.DataFrame):
    """Render data export options."""
    st.subheader("Export Data")
    
    st.write("Export your transaction data in various formats:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = data.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="transactions.csv",
            mime="text/csv"
        )
    
    with col2:
        json = data.to_json(orient='records', date_format='iso')
        st.download_button(
            label="📥 Download JSON",
            data=json,
            file_name="transactions.json",
            mime="application/json"
        )
    
    with col3:
        excel = data.to_excel(index=False)
        st.download_button(
            label="📥 Download Excel",
            data=excel,
            file_name="transactions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# Streamlit will execute this file directly
main()

