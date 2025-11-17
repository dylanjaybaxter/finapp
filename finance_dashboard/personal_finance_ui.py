"""Personal Finance UI Components and Layout.

This module contains enhanced UI components specifically designed for personal
finance management, including budget tracking, spending analysis, and financial
goal management. These components provide a more sophisticated and user-friendly
interface compared to the basic dashboard.
"""

from __future__ import annotations

import streamlit as st
from streamlit.errors import StreamlitAPIException
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Handle both relative and absolute imports
try:
    from .personal_finance_analytics import PersonalFinanceAnalytics
except ImportError:
    from personal_finance_analytics import PersonalFinanceAnalytics


class PersonalFinanceUI:
    """Enhanced UI components for personal finance management."""
    _PAGE_CONFIGURED = False

    def __init__(self, *, configure_page: bool = False):
        """Initialize the Personal Finance UI.

        Args:
            configure_page: When True, call ``setup_page_config`` immediately.
        """
        if configure_page:
            self.setup_page_config()
    
    def setup_page_config(self) -> None:
        """Configure Streamlit page settings for personal finance."""
        if PersonalFinanceUI._PAGE_CONFIGURED:
            return
        try:
            st.set_page_config(
                page_title="Personal Finance Dashboard",
                page_icon="💰",
                layout="wide",
                initial_sidebar_state="expanded",
                menu_items={
                    'Get Help': 'https://github.com/your-repo/personal-finance-dashboard',
                    'Report a bug': "https://github.com/your-repo/personal-finance-dashboard/issues",
                    'About': "Personal Finance Dashboard - Take control of your money!"
                }
            )
        except StreamlitAPIException:
            # Already configured upstream; avoid raising to keep reruns smooth.
            pass
        finally:
            PersonalFinanceUI._PAGE_CONFIGURED = True
    
    def render_header(self) -> None:
        """Render the main dashboard header."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("💰 Personal Finance Dashboard")
            st.markdown("Take control of your money with smart budgeting and spending insights")
        
        with col2:
            st.metric(
                label="Current Month",
                value=datetime.now().strftime("%B %Y"),
                delta=None
            )
        
        with col3:
            if st.button("⚙️ Settings", help="Configure your preferences"):
                st.session_state.show_settings = True
    
    def render_quick_actions(self) -> None:
        """Render quick action buttons for common tasks."""
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("➕ Add Transaction", help="Manually add a transaction"):
                st.session_state.show_add_transaction = True
        
        with col2:
            if st.button("📊 View Budget", help="Check your budget status"):
                st.session_state.show_budget = True
        
        with col3:
            if st.button("🎯 Set Goal", help="Create a new financial goal"):
                st.session_state.show_goal_setting = True
        
        with col4:
            if st.button("📈 View Reports", help="Generate financial reports"):
                st.session_state.show_reports = True
        
        with col5:
            if st.button("✏️ Edit Transactions", help="Edit transfers and uncategorized transactions"):
                st.session_state.show_transaction_editor = True
    
    def render_financial_overview(self, data: pd.DataFrame, filters: Dict = None) -> None:
        """Render key financial metrics overview for selected period."""
        st.subheader("📊 Financial Overview")
        
        if data.empty:
            st.warning("No data available for the selected period.")
            return
        
        # Calculate period summary instead of monthly
        analytics = PersonalFinanceAnalytics(data)

        # Get date range from filters or use all data
        start_date = filters.get('start_date') if filters else None
        end_date = filters.get('end_date') if filters else None

        period_summary = analytics.calculate_period_summary(start_date, end_date)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 Total Income",
                value=f"${period_summary['income']:,.2f}",
                help="Total income for the selected period"
            )
        
        with col2:
            st.metric(
                label="💸 Total Expenses",
                value=f"${period_summary['expenses']:,.2f}",
                help="Total expenses for the selected period"
            )
        
        with col3:
            st.metric(
                label="📈 Net Cash Flow",
                value=f"${period_summary['net_flow']:,.2f}",
                delta=f"{period_summary['net_flow']:,.2f}",
                help="Net cash flow (income - expenses)"
            )
        
        with col4:
            st.metric(
                label="💾 Savings Rate",
                value=f"{period_summary['savings_rate']:.1f}%",
                delta="Good" if period_summary['savings_rate'] > 20 else "Needs Improvement",
                help="Percentage of income saved"
            )
        
    def render_spending_breakdown(self, data: pd.DataFrame, filters: Dict = None) -> None:
        """Render spending breakdown by category with clickable pie chart."""
        st.subheader("💳 Spending Breakdown")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Category spending pie chart with clickable categories
            analytics = PersonalFinanceAnalytics(data)
            start_date = filters.get('start_date') if filters else None
            end_date = filters.get('end_date') if filters else None
            
            category_spending_df = analytics.calculate_category_spending(start_date, end_date)
            category_spending = category_spending_df['Total_Spent']
            
            fig = px.pie(
                values=category_spending.values,
                names=category_spending.index,
                title="Spending by Category (Click to view transactions)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top spending categories table with clickable categories
            st.markdown("**Top Spending Categories**")
            top_categories = category_spending.head(5)
            
            for category, amount in top_categories.items():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"• **{category}**: ${amount:,.2f}")
                with col_b:
                    if st.button("📋", key=f"cat_{category}", help=f"View {category} transactions"):
                        st.session_state[f'show_category_{category}'] = True
            
            # Show category transactions if clicked
            for category in top_categories.index:
                if st.session_state.get(f'show_category_{category}', False):
                    with st.expander(f"Transactions - {category}"):
                        category_data = analytics.get_transactions_by_category(category, start_date, end_date)
                        if not category_data.empty:
                            expense_data = category_data[category_data['Amount'] < 0].copy()
                            expense_data['Amount'] = expense_data['Amount'].abs()
                            st.dataframe(expense_data[['Transaction Date', 'Description', 'Amount', 'Type']], use_container_width=True)
                        else:
                            st.info("No transactions found for this category.")
                        if st.button("Close", key=f"close_cat_{category}"):
                            st.session_state[f'show_category_{category}'] = False
                            st.rerun()
    
    def render_budget_tracking(self, data: pd.DataFrame, budgets: Dict[str, float]) -> None:
        """Render budget tracking visualization."""
        st.subheader("📋 Budget Tracking")
        
        if not budgets:
            st.info("No budgets set. Click 'Set Budget' to create your first budget.")
            return
        
        if data.empty:
            st.info("No transactions match the current filters.")
            return

        analytics = PersonalFinanceAnalytics(data)
        performance = analytics.calculate_budget_performance(budgets)
        if performance.empty:
            st.info("No spending available for the selected period.")
            return

        total_budget = sum(budgets.values())
        total_actual = performance['Actual'].sum()
        variance = total_budget - total_actual
        cols = st.columns(3)
        cols[0].metric("Budgeted", f"${total_budget:,.2f}")
        cols[1].metric("Actual", f"${total_actual:,.2f}")
        cols[2].metric("Variance", f"${variance:,.2f}", delta_color="inverse")

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Budget', x=performance['Category'], y=performance['Budget'], marker_color='#1f77b4'))
        fig.add_trace(go.Bar(name='Actual', x=performance['Category'], y=performance['Actual'], marker_color='#ff7f0e'))
        fig.update_layout(title="Budget vs Actual", barmode='group', xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Status by category**")
        performance['Variance'] = performance['Budget'] - performance['Actual']
        st.dataframe(
            performance[['Category', 'Budget', 'Actual', 'Variance', 'Status']]
            .style.format({'Budget': '${:,.2f}', 'Actual': '${:,.2f}', 'Variance': '${:,.2f}'}),
            use_container_width=True,
        )
    
    def render_spending_trends(self, data: pd.DataFrame) -> None:
        """Render spending trends over time."""
        st.subheader("📈 Spending Trends")
        
        # Convert date column to datetime
        data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])
        
        # Group by month and category
        monthly_spending = data[data['Amount'] < 0].copy()
        monthly_spending['Month'] = monthly_spending['Transaction Date'].dt.to_period('M')
        monthly_trends = monthly_spending.groupby(['Month', 'Category'])['Amount'].sum().abs().reset_index()
        # Ensure Month is JSON serializable for Plotly
        monthly_trends['Month'] = monthly_trends['Month'].astype(str)

        # Controls: category selection with Clear All / Add All and smoothing slider
        all_categories = sorted(monthly_trends['Category'].unique().tolist())
        # Initialize widget-backed state once
        if 'trends_category_multiselect' not in st.session_state:
            st.session_state['trends_category_multiselect'] = all_categories
        if 'trends_smoothing_slider' not in st.session_state:
            st.session_state['trends_smoothing_slider'] = 1

        # Buttons first: allow mutating session_state then immediately rerun
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("Clear All"):
                st.session_state['trends_category_multiselect'] = []
                st.rerun()
        with b2:
            if st.button("Add All"):
                st.session_state['trends_category_multiselect'] = all_categories
                st.rerun()

        # Controls row: multiselect and smoothing slider
        c1, c2 = st.columns([3, 1])
        with c1:
            selected_categories = st.multiselect(
                "Categories",
                options=all_categories,
                key="trends_category_multiselect",
            )
        with c2:
            smoothing = st.slider(
                "Smoothing (months)",
                min_value=1,
                max_value=12,
                help="Applies a centered moving average to spending by category",
                key="trends_smoothing_slider",
            )
        # Read back the authoritative widget state
        selected_categories = st.session_state['trends_category_multiselect']
        smoothing = st.session_state['trends_smoothing_slider']

        # Filter by selected categories
        if selected_categories:
            monthly_trends = monthly_trends[monthly_trends['Category'].isin(selected_categories)].copy()
        else:
            # If empty selection, show empty chart gracefully
            st.info("No categories selected. Use 'Add All' or pick categories to view trends.")
            empty_fig = px.line(title="Monthly Spending Trends by Category")
            empty_fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Amount ($)",
            )
            st.plotly_chart(empty_fig, width='stretch')
            return

        # Prepare for smoothing: create a sortable MonthDate
        monthly_trends['MonthDate'] = pd.to_datetime(monthly_trends['Month'] + "-01", errors='coerce')
        monthly_trends = monthly_trends.sort_values(['Category', 'MonthDate'])

        # Apply moving average smoothing by category
        if smoothing and smoothing > 1:
            monthly_trends['AmountSmoothed'] = (
                monthly_trends.groupby('Category')['Amount']
                .transform(lambda s: s.rolling(window=smoothing, min_periods=1, center=True).mean())
            )
            y_col = 'AmountSmoothed'
        else:
            y_col = 'Amount'

        # Create line chart
        fig = px.line(
            monthly_trends,
            x='Month',
            y=y_col,
            color='Category',
            title="Monthly Spending Trends by Category",
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, width='stretch')

    def render_monthly_income_expense_and_savings(self, data: pd.DataFrame) -> None:
        """Render monthly income, expenses, and savings rate over time."""
        st.subheader("📆 Monthly Income, Expenses, and Savings Rate")

        # Ensure proper datetime
        data = data.copy()
        data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])
        data['Month'] = data['Transaction Date'].dt.to_period('M')

        # Compute monthly aggregates
        monthly = data.groupby('Month')['Amount'].agg([
            ('Income', lambda s: s[s > 0].sum()),
            ('Expenses', lambda s: s[s < 0].sum())
        ]).reset_index()
        # Convert Period to string for Plotly
        monthly['Month'] = monthly['Month'].astype(str)
        # Normalize expenses to positive values for plotting
        monthly['Expenses'] = monthly['Expenses'].abs()
        monthly['Net'] = monthly['Income'] - monthly['Expenses']
        monthly['SavingsRate'] = monthly.apply(
            lambda r: (r['Net'] / r['Income'] * 100.0) if r['Income'] > 0 else 0.0, axis=1
        )

        # Chart 1: Grouped bars for Income vs Expenses with Net as line
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(name='Income', x=monthly['Month'], y=monthly['Income'], marker_color='#2ca02c'))
        bar_fig.add_trace(go.Bar(name='Expenses', x=monthly['Month'], y=monthly['Expenses'], marker_color='#d62728'))
        bar_fig.add_trace(go.Scatter(name='Net', x=monthly['Month'], y=monthly['Net'], mode='lines+markers', yaxis='y2', line=dict(color='#1f77b4')))

        bar_fig.update_layout(
            title="Monthly Income vs Expenses (Net overlay)",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            barmode='group',
            hovermode='x unified',
            yaxis2=dict(title="Net ($)", overlaying='y', side='right', showgrid=False)
        )
        st.plotly_chart(bar_fig, width='stretch')

        # Chart 2: Savings rate over time
        rate_fig = px.line(
            monthly,
            x='Month',
            y='SavingsRate',
            title='Savings Rate Over Time',
        )
        rate_fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Savings Rate (%)",
            hovermode='x unified'
        )
        st.plotly_chart(rate_fig, width='stretch')
    
    def render_financial_goals(self, goals: List[Dict]) -> None:
        """Render financial goals progress."""
        st.subheader("🎯 Financial Goals")
        
        if not goals:
            st.info("No financial goals set. Click 'Set Goal' to create your first goal.")
            return
        
        for goal in goals:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                progress = min(goal['current_amount'] / goal['target_amount'], 1.0)
                st.progress(progress)
                st.markdown(f"**{goal['name']}**: ${goal['current_amount']:,.2f} / ${goal['target_amount']:,.2f}")
            
            with col2:
                remaining = goal['target_amount'] - goal['current_amount']
                st.metric("Remaining", f"${remaining:,.2f}")
    
    def render_sidebar_filters(self, data: pd.DataFrame, defaults: Optional[Dict] = None) -> Dict:
        """Render sidebar filters and controls."""
        defaults = defaults or {}
        st.sidebar.header("🔍 Filters & Controls")

        total_rows = len(data)
        accounts = data['profile_name'].nunique() if 'profile_name' in data.columns else 1
        amount_series = pd.to_numeric(data['Amount'], errors='coerce') if 'Amount' in data.columns else pd.Series(dtype=float)
        spend = abs(amount_series[amount_series < 0].sum()) if not amount_series.empty else 0.0
        income = amount_series[amount_series > 0].sum() if not amount_series.empty else 0.0
        meta_col1, meta_col2 = st.sidebar.columns(2)
        meta_col1.metric("Transactions", f"{total_rows:,}")
        meta_col2.metric("Profiles", f"{accounts:,}")
        st.sidebar.caption(f"Flow snapshot: +${income:,.0f} / -${spend:,.0f}")

        # Date range filter
        st.sidebar.subheader("📅 Date Range")
        date_options = ["Transaction Date", "Post Date"]
        cached_date_col = defaults.get('date_col')
        if cached_date_col not in date_options:
            cached_date_col = date_options[0]
        date_col = st.sidebar.selectbox(
            "Date Column",
            options=date_options,
            index=date_options.index(cached_date_col)
        )

        def _coerce_cached_date(text: Optional[str]) -> Optional[date]:
            if not text:
                return None
            try:
                return pd.to_datetime(text).date()
            except Exception:
                return None

        start_date_value = _coerce_cached_date(defaults.get('start_date'))
        end_date_value = _coerce_cached_date(defaults.get('end_date'))
        date_range = None
        if date_col in data.columns:
            safe_data = data.copy()
            safe_data[date_col] = pd.to_datetime(safe_data[date_col], errors='coerce')
            valid_dates = safe_data[date_col].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                if start_date_value and not (min_date <= start_date_value <= max_date):
                    start_date_value = None
                if end_date_value and not (min_date <= end_date_value <= max_date):
                    end_date_value = None
                date_range = st.sidebar.date_input(
                    "Select Date Range",
                    value=(start_date_value or min_date, end_date_value or max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date_value, end_date_value = date_range
                elif date_range:
                    start_date_value = min_date
                    end_date_value = date_range

        st.sidebar.caption("Or enter dates manually (YYYY-MM-DD)")
        manual_start = st.sidebar.text_input(
            "Start Date",
            value=start_date_value.isoformat() if start_date_value else ""
        ).strip()
        manual_end = st.sidebar.text_input(
            "End Date",
            value=end_date_value.isoformat() if end_date_value else ""
        ).strip()

        def _parse_manual(text: str):
            if not text:
                return None
            try:
                return pd.to_datetime(text).date()
            except Exception:
                st.sidebar.warning(f"Invalid date: {text}. Use YYYY-MM-DD format.")
                return None

        manual_start_date = _parse_manual(manual_start)
        manual_end_date = _parse_manual(manual_end)
        if manual_start_date:
            start_date_value = manual_start_date
        if manual_end_date:
            end_date_value = manual_end_date

        amount_series = pd.to_numeric(data['Amount'], errors='coerce') if 'Amount' in data.columns else pd.Series(dtype=float)
        categories = sorted([c for c in data['Category'].dropna().unique().tolist()]) if 'Category' in data.columns else []
        type_options = sorted(data['Type'].dropna().unique().tolist()) if 'Type' in data.columns else []

        if amount_series.dropna().empty:
            min_amount, max_amount = 0.0, 0.0
        else:
            min_amount = float(amount_series.min())
            max_amount = float(amount_series.max())
            if min_amount == max_amount:
                buffer_value = max(1.0, abs(min_amount) * 0.05)
                min_amount -= buffer_value
                max_amount += buffer_value

        cached_categories = defaults.get('selected_categories') or categories
        default_categories = [c for c in cached_categories if c in categories] if categories else []
        if categories and not default_categories:
            default_categories = categories
        cached_types = defaults.get('transaction_types') or type_options
        default_types = [t for t in cached_types if t in type_options] if type_options else []
        if type_options and not default_types:
            default_types = type_options
        cached_amount_range = defaults.get('amount_range') if defaults else None
        if isinstance(cached_amount_range, (list, tuple)) and len(cached_amount_range) == 2:
            try:
                cached_min = float(cached_amount_range[0])
                cached_max = float(cached_amount_range[1])
            except (TypeError, ValueError):
                cached_min, cached_max = min_amount, max_amount
            cached_min = max(min_amount, min(cached_min, max_amount))
            cached_max = max(min_amount, min(cached_max, max_amount))
            if cached_min > cached_max:
                cached_min, cached_max = cached_max, cached_min
            default_amount_range = (cached_min, cached_max)
        else:
            default_amount_range = (min_amount, max_amount)
        search_text_default = defaults.get('search_text') or ""
        expenses_only_default = bool(defaults.get('expenses_only'))

        with st.sidebar.popover("Advanced filters", use_container_width=True):
            st.caption("Fine-tune the view by category, transaction type, amount, or search text.")
            selected_categories = st.multiselect(
                "Categories",
                options=categories,
                default=default_categories,
            )
            selected_types = st.multiselect(
                "Transaction types",
                options=type_options,
                default=default_types,
            )
            search_text = st.text_input(
                "Keyword search",
                placeholder="e.g. rent, netflix, grocery",
                value=search_text_default,
            ).strip()
            amount_range = st.slider(
                "Amount range",
                min_value=min_amount,
                max_value=max_amount,
                value=default_amount_range
            )
            expenses_only = st.checkbox(
                "Expenses only",
                value=expenses_only_default,
                help="Hide inflows to focus purely on outgoing commitments.",
            )

        return {
            'date_range': date_range,
            'selected_categories': selected_categories,
            'amount_range': amount_range,
            'date_col': date_col,
            'start_date': start_date_value.isoformat() if start_date_value else None,
            'end_date': end_date_value.isoformat() if end_date_value else None,
            'transaction_types': selected_types,
            'search_text': search_text or None,
            'expenses_only': expenses_only,
        }
    
    def render_add_transaction_form(self) -> Optional[Dict]:
        """Render form for adding new transactions."""
        if not st.session_state.get('show_add_transaction', False):
            return None
        
        st.subheader("➕ Add New Transaction")
        
        with st.form("add_transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                date = st.date_input("Transaction Date")
                description = st.text_input("Description")
                category = st.selectbox("Category", options=[
                    "Food & Drink", "Groceries", "Shopping", "Gas", "Bills & Utilities",
                    "Travel", "Home", "Health & Wellness", "Entertainment", "Other"
                ])
            
            with col2:
                amount = st.number_input("Amount", step=0.01)
                transaction_type = st.selectbox("Type", options=["Sale", "Payment", "Transfer"])
                memo = st.text_input("Memo (optional)")
            
            submitted = st.form_submit_button("Add Transaction")
            
            if submitted:
                if not description.strip():
                    st.error("Description is required to add a transaction.")
                else:
                    return {
                        'Transaction Date': date,
                        'Post Date': date,
                        'Description': description.strip(),
                        'Category': category,
                        'Type': transaction_type,
                        'Amount': float(amount),
                        'Memo': memo.strip() if memo else None,
                        'Currency': None,
                        'Transaction Reference': None,
                        'FI Transaction Reference': None,
                        'Original Amount': None
                    }
        
        return None
    
    def render_budget_setting_form(self) -> Optional[Dict]:
        """Render form for setting budgets."""
        if not st.session_state.get('show_budget', False):
            return None
        
        st.subheader("📋 Set Budget")
        
        with st.form("budget_setting_form"):
            budgets = {}
            
            categories = ["Food & Drink", "Groceries", "Shopping", "Gas", "Bills & Utilities",
                         "Travel", "Home", "Health & Wellness", "Entertainment"]
            
            for category in categories:
                budget = st.number_input(
                    f"{category} Budget",
                    min_value=0.0,
                    step=10.0,
                    value=0.0
                )
                if budget > 0:
                    budgets[category] = budget
            
            submitted = st.form_submit_button("Save Budget")
            
            if submitted and budgets:
                return budgets
        
        return None
    
    def render_goal_setting_form(self) -> Optional[Dict]:
        """Render form for setting financial goals."""
        if not st.session_state.get('show_goal_setting', False):
            return None
        
        st.subheader("🎯 Set Financial Goal")
        
        with st.form("goal_setting_form"):
            goal_name = st.text_input("Goal Name")
            target_amount = st.number_input("Target Amount", min_value=0.0, step=100.0)
            current_amount = st.number_input("Current Amount", min_value=0.0, step=100.0)
            target_date = st.date_input("Target Date")
            
            submitted = st.form_submit_button("Save Goal")
            
            if submitted and goal_name and target_amount > 0:
                return {
                    'name': goal_name,
                    'target_amount': target_amount,
                    'current_amount': current_amount,
                    'target_date': target_date
                }
        
        return None
    
    def _filter_by_month(self, data: pd.DataFrame, month: str) -> pd.DataFrame:
        """Filter data by month."""
        if 'Transaction Date' in data.columns:
            working = data.copy()
            working['Transaction Date'] = pd.to_datetime(working['Transaction Date'], errors='coerce')
            month_period = month if isinstance(month, pd.Period) else pd.Period(str(month), freq='M')
            return working[working['Transaction Date'].dt.to_period('M') == month_period]
        return data
    
    def render_mobile_optimized_view(self) -> None:
        """Render mobile-optimized view."""
        st.markdown("""
        <style>
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
            }
            .stMetric {
                margin-bottom: 0.5rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_dark_mode_toggle(self) -> None:
        """Render dark mode toggle."""
        if st.sidebar.button("🌙 Toggle Dark Mode"):
            st.session_state.dark_mode = not st.session_state.get('dark_mode', False)
    
    def apply_dark_mode(self) -> None:
        """Apply dark mode styling."""
        if st.session_state.get('dark_mode', False):
            st.markdown("""
            <style>
            .stApp {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .stMetric {
                background-color: #2d2d2d;
                padding: 1rem;
                border-radius: 0.5rem;
            }
            </style>
            """, unsafe_allow_html=True)
