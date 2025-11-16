"""Overview/Dashboard Page - Main financial overview and insights."""

from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Dict

# Import handling for Streamlit pages
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from shared_sidebar import render_shared_sidebar, apply_filters
from personal_finance_ui import PersonalFinanceUI
from personal_finance_analytics import PersonalFinanceAnalytics


def main():
    """Render the Overview page."""
    st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
    
    # Render shared sidebar
    sidebar_data = render_shared_sidebar()
    base_df = sidebar_data['base_df']
    filters = sidebar_data['filters']
    
    # Initialize UI
    ui = PersonalFinanceUI()
    
    # Load budgets and goals from session state
    budgets = st.session_state.get('budgets', {})
    goals = st.session_state.get('goals', [])
    
    if base_df.empty:
        st.info("No data available. Upload a statement or switch to Database mode if you have previously imported data.")
        
        # Show loaded files info for database mode
        if sidebar_data['source_mode'] == "Database (all history)":
            import db
            loaded_files = db.get_loaded_files()
            if loaded_files:
                st.info(f"📁 Database contains {len(loaded_files)} files")
                with st.expander("View loaded files"):
                    for filename in loaded_files:
                        st.text(filename)
            else:
                st.warning("📁 No files loaded in database yet. Use the 'Scan data/raw directory' button to auto-load files.")
        return
    
    # Apply filters
    filtered_data = apply_filters(base_df, filters)
    
    # Initialize analytics
    analytics_source = filtered_data if not filtered_data.empty else base_df
    analytics = None
    if analytics_source is not None and not analytics_source.empty:
        analytics = PersonalFinanceAnalytics(analytics_source)
    
    if filtered_data.empty:
        st.warning("No transactions match the selected filters. Adjust filters to see results.")
        return
    
    # Render overview dashboard
    st.header("📊 Financial Overview")
    
    # Financial overview metrics
    ui.render_financial_overview(filtered_data, filters)
    
    # Spending breakdown
    ui.render_spending_breakdown(filtered_data, filters)
    
    # Budget tracking
    if budgets:
        ui.render_budget_tracking(filtered_data, budgets)
    
    # Spending trends
    ui.render_spending_trends(filtered_data)

    # Monthly income/expense and savings rate
    ui.render_monthly_income_expense_and_savings(filtered_data)
    
    # Financial goals
    if goals:
        ui.render_financial_goals(goals)
    
    # Financial health metrics
    if analytics:
        _render_financial_health_metrics(analytics)
    
    # Spending insights
    if analytics:
        _render_spending_insights(analytics)


def _render_financial_health_metrics(analytics: PersonalFinanceAnalytics):
    """Render financial health metrics."""
    st.subheader("💚 Financial Health Metrics")
    
    health_metrics = analytics.calculate_financial_health_metrics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Savings Rate",
            f"{health_metrics['savings_rate']:.1f}%",
            delta="Good" if health_metrics['savings_rate'] > 20 else "Needs Improvement"
        )
    
    with col2:
        st.metric(
            "Monthly Savings",
            f"${health_metrics['monthly_savings_avg']:,.2f}",
            delta="Positive" if health_metrics['monthly_savings_avg'] > 0 else "Negative"
        )
    
    with col3:
        st.metric(
            "Top Expense Category",
            health_metrics['top_expense_category'],
            delta=f"${health_metrics['top_expense_amount']:,.2f}"
        )


def _render_spending_insights(analytics: PersonalFinanceAnalytics):
    """Render personalized spending insights."""
    st.subheader("💡 Personalized Insights")
    
    insights = analytics.generate_spending_insights()
    
    for insight in insights:
        st.info(insight)


# Streamlit will execute this file directly
main()

