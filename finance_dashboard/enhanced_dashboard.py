"""Enhanced Personal Finance Dashboard.

This module provides a comprehensive personal finance dashboard with advanced
features for budgeting, spending analysis, goal tracking, and financial insights.
It integrates the enhanced UI components and personal finance analytics.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os

# Import our custom modules
# Handle both relative and absolute imports
try:
    from . import data_processing as dp
    from . import visualization as viz
    from . import db
    from .personal_finance_ui import PersonalFinanceUI
    from .personal_finance_analytics import PersonalFinanceAnalytics
except ImportError:
    # Fallback for direct execution
    import data_processing as dp
    import visualization as viz
    import db
    from personal_finance_ui import PersonalFinanceUI
    from personal_finance_analytics import PersonalFinanceAnalytics


class EnhancedPersonalFinanceDashboard:
    """Enhanced personal finance dashboard with advanced features."""
    
    def __init__(self):
        """Initialize the enhanced dashboard."""
        self.ui = PersonalFinanceUI()
        self.data = None
        # Initialize database
        db.init_db()
        self.analytics = None
        self.budgets = {}
        self.goals = []
        self._load_saved_data()
    
    def _load_saved_data(self) -> None:
        """Load saved budgets and goals from session state or file."""
        # Load budgets
        if 'budgets' in st.session_state:
            self.budgets = st.session_state.budgets
        else:
            self.budgets = {}
        
        # Load goals
        if 'goals' in st.session_state:
            self.goals = st.session_state.goals
        else:
            self.goals = []
    
    def _save_data(self) -> None:
        """Save budgets and goals to session state."""
        st.session_state.budgets = self.budgets
        st.session_state.goals = self.goals
    
    def run(self) -> None:
        """Run the enhanced personal finance dashboard."""
        # Apply dark mode if enabled
        self.ui.apply_dark_mode()
        # Ensure database is initialized
        db.init_db()
        
        # Render header
        self.ui.render_header()
        
        # Render dark mode toggle
        self.ui.render_dark_mode_toggle()
        
        # Auto-load section
        st.sidebar.subheader("📂 Auto-Load Files")
        if st.sidebar.button("🔄 Scan data/raw directory"):
            results = db.auto_load_raw_files()
            
            if results['loaded_files']:
                st.sidebar.success(f"✅ Loaded {len(results['loaded_files'])} new files")
                for filename in results['loaded_files']:
                    st.sidebar.text(f"  • {filename}")
            
            if results['skipped_files']:
                st.sidebar.info(f"⏭️ Skipped {len(results['skipped_files'])} already loaded files")
            
            if results['error_files']:
                st.sidebar.error(f"❌ {len(results['error_files'])} files had errors:")
                for error in results['error_files']:
                    st.sidebar.text(f"  • {error}")
            
            if results['total_inserted'] > 0:
                st.sidebar.info(f"📊 Imported {results['total_inserted']} new transactions, skipped {results['total_skipped']} duplicates")
            
            if not any([results['loaded_files'], results['skipped_files'], results['error_files']]):
                st.sidebar.info("No files found in data/raw directory")

        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "📁 Upload Bank Statement",
            type=["csv", "xls", "xlsx"],
            help="Upload your bank statement in CSV or Excel format"
        )

        # Data source selection
        source_mode = st.sidebar.radio(
            "Data Source",
            options=["Database (all history)", "Current upload only"],
            index=0,
            help="Choose whether to analyze all persisted data or only the current upload",
        )
        
        if uploaded_file is not None:
            # Load and process data
            self.data = self._load_data(uploaded_file)
            if not self.data.empty:
                # Persist to database with deduplication
                inserted, skipped = db.upsert_transactions(self.data, account='default', source_file=getattr(uploaded_file, 'name', None))
                st.success(f"Imported {inserted} new transactions. Skipped {skipped} duplicates.")
                self.analytics = PersonalFinanceAnalytics(self.data)
                self._render_main_dashboard(source_mode)
            else:
                st.error("Failed to load data. Please check your file format.")
        else:
            self._render_welcome_screen()
            # If user has previously imported data, allow viewing DB data
            if source_mode == "Database (all history)":
                # Proceed to main dashboard using DB source
                self._render_main_dashboard(source_mode)
                return
    
    def _load_data(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded file with format detection."""
        try:
            # Read the file
            df = dp.read_file(uploaded_file)
            
            if df.empty:
                st.error("The uploaded file appears to be empty.")
                return pd.DataFrame()
            
            # Detect format and normalize columns
            normalized_df, format_name = dp.detect_format_and_normalize(df, uploaded_file.name)
            
            # Show format detection result
            st.info(f"📋 Detected format: **{format_name}**")
            
            # Show column mapping if any
            if hasattr(normalized_df, 'attrs') and 'column_mapping' in normalized_df.attrs:
                mapping = normalized_df.attrs['column_mapping']
                if mapping:
                    with st.expander("🔍 Column Mapping Details"):
                        st.write("**Original → Normalized:**")
                        for orig, norm in mapping.items():
                            st.write(f"• `{orig}` → `{norm}`")
            
            # Store in database
            inserted, skipped = db.upsert_transactions(normalized_df, account='manual_upload', source_file=uploaded_file.name)
            
            if inserted > 0:
                st.success(f"✅ Successfully imported {inserted} transactions")
            if skipped > 0:
                st.info(f"⏭️ Skipped {skipped} duplicate transactions")
            
            return normalized_df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return pd.DataFrame()
    
    def _render_welcome_screen(self) -> None:
        """Render welcome screen when no data is loaded."""
        st.markdown("""
        # Welcome to Your Personal Finance Dashboard! 💰
        
        This enhanced dashboard helps you:
        - 📊 **Track your spending** by category and time
        - 📋 **Create and monitor budgets** for each spending category
        - 🎯 **Set and achieve financial goals**
        - 📈 **Analyze spending patterns** and trends
        - 💡 **Get personalized insights** for better financial decisions
        
        ## Getting Started
        
        1. **Upload your bank statement** using the sidebar
        2. **Set up your budgets** for different spending categories
        3. **Create financial goals** to work towards
        4. **Explore your spending patterns** with interactive charts
        
        ## Supported File Formats
        
        - CSV files (comma-separated values)
        - Excel files (.xlsx, .xls)
        - Bank statement exports from major banks
        
        ## Sample Data Structure
        
        Your file should contain columns like:
        - Transaction Date
        - Description
        - Category
        - Amount
        - Type
        
        Upload a file to get started! 🚀
        """)
        
        # Show sample data if available
        if os.path.exists("sample_data/Chase1337_Activity20230913_20250913_20250914.CSV"):
            st.markdown("### Sample Data Available")
            if st.button("Load Sample Data"):
                sample_data = pd.read_csv("sample_data/Chase1337_Activity20230913_20250913_20250914.CSV")
                self.data = sample_data
                self.analytics = PersonalFinanceAnalytics(self.data)
                st.rerun()
    
    def _render_main_dashboard(self, source_mode: str) -> None:
        """Render the main dashboard with all features."""
        # Render quick actions
        self.ui.render_quick_actions()
        
        # Decide base DataFrame according to source_mode
        if source_mode == "Database (all history)":
            # Load entire dataset from DB first (filters will be applied below)
            base_df = db.fetch_transactions()
        else:
            base_df = self.data

        if base_df is None or base_df.empty:
            st.info("No data available. Upload a statement or switch to Database mode if you have previously imported data.")
            
            # Show loaded files info for database mode
            if source_mode == "Database (all history)":
                loaded_files = db.get_loaded_files()
                if loaded_files:
                    st.info(f"📁 Database contains {len(loaded_files)} files")
                    with st.expander("View loaded files"):
                        for filename in loaded_files:
                            st.text(filename)
                else:
                    st.warning("📁 No files loaded in database yet. Use the 'Scan data/raw directory' button to auto-load files.")
            return

        # Render sidebar filters using base_df columns
        filters = self.ui.render_sidebar_filters(base_df)
        
        # Apply filters to data
        # Apply filters to base_df
        filtered_data = self._apply_filters(base_df, filters)
        
        # Choose the dataset to render with: filtered if any rows, else original
        data_for_view = filtered_data if not filtered_data.empty else self.data
        
        # Update analytics with the data being viewed
        if not data_for_view.empty:
            self.analytics = PersonalFinanceAnalytics(data_for_view)
        
        # Render main content based on current view
        if st.session_state.get('show_budget', False):
            self._render_budget_view()
        elif st.session_state.get('show_goal_setting', False):
            self._render_goals_view()
        elif st.session_state.get('show_reports', False):
            self._render_reports_view()
        elif st.session_state.get('show_add_transaction', False):
            self._render_add_transaction_view()
        else:
            self._render_overview_dashboard(data_for_view, filters)
        
        # Handle form submissions
        self._handle_form_submissions()
    
    def _apply_filters(self, data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to the data."""
        filtered_data = data.copy()
        
        # Date range filter
        if filters['date_range'] and len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            date_col = filters['date_col']
            if date_col in filtered_data.columns:
                filtered_data[date_col] = pd.to_datetime(filtered_data[date_col])
                filtered_data = filtered_data[
                    (filtered_data[date_col] >= pd.to_datetime(start_date)) &
                    (filtered_data[date_col] <= pd.to_datetime(end_date))
                ]
        
        # Category filter
        if filters['selected_categories']:
            filtered_data = filtered_data[filtered_data['Category'].isin(filters['selected_categories'])]
        
        # Amount range filter
        if filters['amount_range']:
            min_amount, max_amount = filters['amount_range']
            filtered_data = filtered_data[
                (filtered_data['Amount'] >= min_amount) &
                (filtered_data['Amount'] <= max_amount)
            ]
        
        return filtered_data
    
    def _render_overview_dashboard(self, data: pd.DataFrame, filters: Dict = None) -> None:
        """Render the main overview dashboard for the provided data."""
        # Financial overview metrics
        self.ui.render_financial_overview(data, filters)
        
        # Spending breakdown
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.ui.render_spending_breakdown(data, filters)
        
        with col2:
            # Monthly panel instead of recent transactions
            self.ui.render_monthly_panel(data, filters)
        
        # Budget tracking
        if self.budgets:
            self.ui.render_budget_tracking(data, self.budgets)
        
        # Spending trends
        self.ui.render_spending_trends(data)

        # Monthly income/expense and savings rate
        self.ui.render_monthly_income_expense_and_savings(data)
        
        # Financial goals
        if self.goals:
            self.ui.render_financial_goals(self.goals)
        
        # Financial health metrics
        if self.analytics:
            self._render_financial_health_metrics()
        
        # Spending insights
        if self.analytics:
            self._render_spending_insights()
    
    def _render_budget_view(self) -> None:
        """Render budget management view."""
        st.header("📋 Budget Management")
        
        # Budget setting form
        new_budgets = self.ui.render_budget_setting_form()
        if new_budgets:
            self.budgets.update(new_budgets)
            self._save_data()
            st.success("Budget updated successfully!")
            st.rerun()
        
        # Current budgets
        if self.budgets:
            st.subheader("Current Budgets")
            budget_df = pd.DataFrame(list(self.budgets.items()), columns=['Category', 'Budget'])
            st.dataframe(budget_df, use_container_width=True)
            
            # Budget performance
            if self.analytics:
                budget_performance = self.analytics.calculate_budget_performance(self.budgets)
                st.subheader("Budget Performance")
                st.dataframe(budget_performance, use_container_width=True)
        else:
            st.info("No budgets set. Use the form above to create your first budget.")
    
    def _render_goals_view(self) -> None:
        """Render financial goals view."""
        st.header("🎯 Financial Goals")
        
        # Goal setting form
        new_goal = self.ui.render_goal_setting_form()
        if new_goal:
            self.goals.append(new_goal)
            self._save_data()
            st.success("Goal added successfully!")
            st.rerun()
        
        # Current goals
        if self.goals:
            st.subheader("Current Goals")
            for i, goal in enumerate(self.goals):
                with st.expander(f"Goal: {goal['name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Target Amount", f"${goal['target_amount']:,.2f}")
                        st.metric("Current Amount", f"${goal['current_amount']:,.2f}")
                    with col2:
                        remaining = goal['target_amount'] - goal['current_amount']
                        progress = goal['current_amount'] / goal['target_amount']
                        st.metric("Remaining", f"${remaining:,.2f}")
                        st.progress(progress)
            
            # Goal progress analysis
            if self.analytics:
                goal_progress = self.analytics.calculate_goal_progress(self.goals)
                st.subheader("Goal Progress Analysis")
                progress_df = pd.DataFrame(goal_progress)
                st.dataframe(progress_df, use_container_width=True)
        else:
            st.info("No goals set. Use the form above to create your first financial goal.")
    
    def _render_reports_view(self) -> None:
        """Render reports and analytics view."""
        st.header("📊 Reports & Analytics")
        
        if not self.analytics:
            st.warning("No data available for analysis.")
            return
        
        # Report type selection
        report_type = st.selectbox(
            "Select Report Type",
            ["Spending Analysis", "Budget Performance", "Financial Health", "Spending Patterns", "Export Data"]
        )
        
        if report_type == "Spending Analysis":
            self._render_spending_analysis_report()
        elif report_type == "Budget Performance":
            self._render_budget_performance_report()
        elif report_type == "Financial Health":
            self._render_financial_health_report()
        elif report_type == "Spending Patterns":
            self._render_spending_patterns_report()
        elif report_type == "Export Data":
            self._render_export_data()
    
    def _render_spending_analysis_report(self) -> None:
        """Render detailed spending analysis report."""
        st.subheader("Spending Analysis Report")
        
        # Category spending
        category_spending = self.analytics.calculate_category_spending()
        st.dataframe(category_spending, use_container_width=True)
        
        # Spending trends
        trends = self.analytics.calculate_spending_trends()
        fig = px.line(trends, x='Period', y='Amount', title='Spending Trends Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_budget_performance_report(self) -> None:
        """Render budget performance report."""
        st.subheader("Budget Performance Report")
        
        if not self.budgets:
            st.info("No budgets set. Create budgets to see performance reports.")
            return
        
        budget_performance = self.analytics.calculate_budget_performance(self.budgets)
        st.dataframe(budget_performance, use_container_width=True)
        
        # Budget performance chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Budget', x=budget_performance['Category'], y=budget_performance['Budget']))
        fig.add_trace(go.Bar(name='Actual', x=budget_performance['Category'], y=budget_performance['Actual']))
        fig.update_layout(title='Budget vs Actual Spending', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_financial_health_report(self) -> None:
        """Render financial health report."""
        st.subheader("Financial Health Report")
        
        health_metrics = self.analytics.calculate_financial_health_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("12-Month Income", f"${health_metrics['total_income_12m']:,.2f}")
            st.metric("12-Month Expenses", f"${health_metrics['total_expenses_12m']:,.2f}")
            st.metric("Net Worth Change", f"${health_metrics['net_worth_change_12m']:,.2f}")
        
        with col2:
            st.metric("Monthly Income Avg", f"${health_metrics['monthly_income_avg']:,.2f}")
            st.metric("Monthly Expenses Avg", f"${health_metrics['monthly_expenses_avg']:,.2f}")
            st.metric("Savings Rate", f"{health_metrics['savings_rate']:.1f}%")
        
        # Debt analysis
        debt_analysis = self.analytics.calculate_debt_analysis()
        st.subheader("Debt Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Monthly Debt Payments", f"${debt_analysis['monthly_debt_payments']:,.2f}")
        with col2:
            st.metric("Debt-to-Income Ratio", f"{debt_analysis['debt_to_income_ratio']:.1f}%")
    
    def _render_spending_patterns_report(self) -> None:
        """Render spending patterns report."""
        st.subheader("Spending Patterns Report")
        
        patterns = self.analytics.calculate_spending_patterns()
        
        # Weekday spending
        st.subheader("Spending by Day of Week")
        fig = px.bar(x=patterns['weekday_spending'].index, y=patterns['weekday_spending'].values)
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly spending
        st.subheader("Spending by Month")
        fig = px.bar(x=patterns['monthly_spending'].index, y=patterns['monthly_spending'].values)
        st.plotly_chart(fig, use_container_width=True)
        
        # Spending anomalies
        st.subheader("Spending Anomalies")
        anomalies = self.analytics.detect_spending_anomalies()
        if not anomalies.empty:
            st.dataframe(anomalies, use_container_width=True)
        else:
            st.info("No spending anomalies detected.")
    
    def _render_export_data(self) -> None:
        """Render data export options."""
        st.subheader("Export Data")
        
        if st.button("Export Analysis Report"):
            filename = self.analytics.export_analysis_report()
            st.success(f"Report exported to {filename}")
        
        if st.button("Export Raw Data"):
            csv = self.data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"personal_finance_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    def _render_add_transaction_view(self) -> None:
        """Render add transaction view."""
        st.header("➕ Add Transaction")
        
        new_transaction = self.ui.render_add_transaction_form()
        if new_transaction:
            # Add transaction to data
            new_row = pd.DataFrame([new_transaction])
            self.data = pd.concat([self.data, new_row], ignore_index=True)
            st.success("Transaction added successfully!")
            st.rerun()
    
    def _render_financial_health_metrics(self) -> None:
        """Render financial health metrics."""
        st.subheader("💚 Financial Health")
        
        health_metrics = self.analytics.calculate_financial_health_metrics()
        
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
    
    def _render_spending_insights(self) -> None:
        """Render personalized spending insights."""
        st.subheader("💡 Personalized Insights")
        
        insights = self.analytics.generate_spending_insights()
        
        for insight in insights:
            st.info(insight)
    
    def _handle_form_submissions(self) -> None:
        """Handle form submissions and update data."""
        # This method is called after rendering to handle any form submissions
        pass


def main() -> None:
    """Main entry point for the enhanced personal finance dashboard."""
    dashboard = EnhancedPersonalFinanceDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
