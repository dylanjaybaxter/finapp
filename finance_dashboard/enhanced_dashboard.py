"""Enhanced Personal Finance Dashboard - Main Entry Point.

This is the main entry point for the multi-page Streamlit dashboard.
The actual page content is in the pages/ directory.
"""

from __future__ import annotations

import streamlit as st

# Import our custom modules
try:
    from . import db
    from . import config
    from .personal_finance_ui import PersonalFinanceUI
    from .shared_sidebar import render_shared_sidebar, apply_filters as shared_apply_filters
except ImportError:
    # Fallback for direct execution
    import db
    import config
    from personal_finance_ui import PersonalFinanceUI
    from shared_sidebar import render_shared_sidebar, apply_filters as shared_apply_filters


def main():
    """Main entry point for the enhanced personal finance dashboard."""
    # Set page config
    st.set_page_config(
        page_title="Personal Finance Dashboard",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize database and directories
    config.ensure_data_directories()
    db.init_db()
    
    # Initialize session state defaults
    if 'budgets' not in st.session_state:
        st.session_state.budgets = {}
    if 'goals' not in st.session_state:
        st.session_state.goals = []
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # Render welcome screen if no data
    sidebar_data = render_shared_sidebar()
    base_df = sidebar_data['base_df']
    
    if base_df.empty:
        _render_welcome_screen()


def _render_welcome_screen() -> None:
    """Render welcome screen when no data is loaded."""
    st.markdown("""
    # Welcome to Your Personal Finance Dashboard! 💰
    
    This enhanced dashboard helps you:
    - 📊 **Track your spending** by category and time
    - 📋 **Create and monitor budgets** for each spending category
    - 🎯 **Set and achieve financial goals**
    - 📈 **Analyze spending patterns** and trends
    - 💡 **Get personalized insights** for better financial decisions
    - ✏️ **Edit and categorize transactions**
    
    ## Getting Started
    
    1. **Upload your bank statement** using the sidebar
    2. **Use the pages** in the sidebar to navigate different sections:
       - 📊 **Overview**: Main financial dashboard
       - 📋 **Budgets**: Create and track budgets
       - 🎯 **Goals**: Set financial goals
       - 📈 **Reports**: Detailed analytics and reports
       - ✏️ **Transactions**: Edit and categorize transactions
    3. **Set up your budgets** for different spending categories
    4. **Create financial goals** to work towards
    5. **Explore your spending patterns** with interactive charts
    
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
    import os
    import pandas as pd
    if os.path.exists("sample_data/Chase1337_Activity20230913_20250913_20250914.CSV"):
        st.markdown("### Sample Data Available")
        if st.button("Load Sample Data"):
            sample_data = pd.read_csv("sample_data/Chase1337_Activity20230913_20250913_20250914.CSV")
            # Process and import sample data
            try:
                from . import data_processing as dp
            except ImportError:
                import data_processing as dp
            normalized_df, profile_label = dp.detect_format_and_normalize(sample_data, "sample_data.csv")
            profile_identity = normalized_df.attrs.get('profile_name') or profile_label
            db.upsert_transactions(
                normalized_df,
                account='sample',
                source_file='sample_data.csv',
                profile_name=profile_identity,
            )
            st.success("Sample data loaded! Navigate to Overview page to see it.")
            st.rerun()


# Legacy class-based approach - kept for backward compatibility
class EnhancedPersonalFinanceDashboard:
    """Enhanced personal finance dashboard with advanced features.
    
    NOTE: This class is deprecated. The dashboard now uses a multi-page structure.
    Use the pages in the pages/ directory instead.
    """
    
    def __init__(self):
        """Initialize the enhanced dashboard."""
        # Ensure data directories exist
        config.ensure_data_directories()
        # Initialize database
        db.init_db()
        self.ui = PersonalFinanceUI(configure_page=True)
    
    def run(self) -> None:
        """Run the enhanced personal finance dashboard."""
        self._ensure_session_defaults()
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

            if results.get('issues'):
                st.sidebar.warning("⚠️ Normalization notes:")
                for fname, notes in results['issues'].items():
                    st.sidebar.write(f"**{fname}**")
                    for note in notes:
                        st.sidebar.write(f"  - {note}")

            if results['total_inserted'] > 0:
                st.sidebar.info(f"📊 Imported {results['total_inserted']} new transactions, skipped {results['total_skipped']} duplicates")
            
            if not any([results['loaded_files'], results['skipped_files'], results['error_files']]):
                st.sidebar.info("No files found in data/raw directory")

        # Database management
        st.sidebar.subheader("🗄️ Database Management")
        if st.sidebar.button("🗑️ Clear Database", help="Delete all transactions from the database"):
            if 'confirm_clear' not in st.session_state:
                st.session_state.confirm_clear = False
            st.session_state.confirm_clear = True
        
        if st.session_state.get('confirm_clear', False):
            st.sidebar.warning("⚠️ This will delete ALL transactions!")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("✅ Confirm", key="confirm_clear_btn"):
                    if db.clear_database():
                        st.session_state.confirm_clear = False
                        st.sidebar.success("Database cleared successfully!")
                        st.rerun()
            with col2:
                if st.button("❌ Cancel", key="cancel_clear_btn"):
                    st.session_state.confirm_clear = False
                    st.rerun()

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
            if self.data.empty:
                st.error("Failed to load data. Please check your file format.")
        
        # Determine base dataset according to selected source
        if source_mode == "Database (all history)":
            base_df = db.fetch_transactions()
        else:
            base_df = self._current_upload.copy() if isinstance(self._current_upload, pd.DataFrame) else pd.DataFrame()

        # Keep a local copy so manual additions can reuse it
        self.data = base_df.copy() if not base_df.empty else base_df
        self._active_source_mode = source_mode

        if base_df.empty:
            self._render_welcome_screen()
            # Still render dashboard to allow quick actions (e.g. manual add)
            self._render_main_dashboard(base_df, source_mode)
            return

        self._render_main_dashboard(base_df, source_mode)
    
    def _load_data(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded file with format detection."""
        try:
            # Read the file
            df = dp.read_file(uploaded_file)
            
            if df.empty:
                st.error("The uploaded file appears to be empty.")
                self._current_upload = None
                return pd.DataFrame()
            
            # Detect format and normalize columns
            normalized_df, format_name = dp.detect_format_and_normalize(df, uploaded_file.name)
            profile_identity = normalized_df.attrs.get('profile_name') or format_name
            
            # Show format detection result
            st.info(f"📋 Detected format: **{format_name}**")
            
            # Show column mapping if any
            if hasattr(normalized_df, 'attrs') and 'column_mapping' in normalized_df.attrs:
                mapping = normalized_df.attrs['column_mapping']
                if mapping:
                    with st.expander("🔍 Column Mapping Details"):
                        st.write("**Original → Normalized:**")
                        for normalized_name, original_name in mapping.items():
                            st.write(f"• `{original_name}` → `{normalized_name}`")
            issues = normalized_df.attrs.get('normalization_issues')
            if issues:
                with st.expander("⚠️ Normalization Notes"):
                    for issue in issues:
                        st.write(f"- {issue}")
            unused_columns = normalized_df.attrs.get('unmapped_columns') or []
            if unused_columns:
                with st.expander("📦 Unmapped Source Columns"):
                    st.write(", ".join(unused_columns))
            
            # Store in database with profile name
            inserted, skipped = db.upsert_transactions(
                normalized_df,
                account='manual_upload',
                source_file=uploaded_file.name,
                profile_name=profile_identity,
            )
            
            if inserted > 0:
                st.success(f"✅ Successfully imported {inserted} transactions")
            if skipped > 0:
                st.info(f"⏭️ Skipped {skipped} duplicate transactions")

            self._current_upload = normalized_df
            return normalized_df

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            self._current_upload = None
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
    
    def _render_main_dashboard(self, base_df: pd.DataFrame, source_mode: str) -> None:
        """Render the main dashboard with all features."""
        # Render quick actions
        self.ui.render_quick_actions()

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
        filtered_data = self._apply_filters(base_df, filters)
        data_for_view = filtered_data

        analytics_source = data_for_view if not data_for_view.empty else base_df
        if analytics_source is not None and not analytics_source.empty:
            self.analytics = PersonalFinanceAnalytics(analytics_source)
        else:
            self.analytics = None

        if data_for_view.empty:
            st.warning("No transactions match the selected filters. Adjust filters to see results.")

        # Render main content based on current view
        if st.session_state.get('show_transaction_editor', False):
            self._render_transaction_editor()
        elif st.session_state.get('show_budget', False):
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
        return shared_apply_filters(data, filters)
    
    def _render_overview_dashboard(self, data: pd.DataFrame, filters: Dict = None) -> None:
        """Render the main overview dashboard for the provided data."""
        # Financial overview metrics
        self.ui.render_financial_overview(data, filters)
        
        # Spending breakdown
        self.ui.render_spending_breakdown(data, filters)
        
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
            transaction_row = new_transaction.copy()
            transaction_row['Transaction Date'] = pd.to_datetime(transaction_row['Transaction Date'])
            transaction_row['Post Date'] = pd.to_datetime(transaction_row['Post Date'])
            display_row = pd.DataFrame([{
                'account': 'manual_entry',
                'Transaction Date': transaction_row['Transaction Date'],
                'Post Date': transaction_row['Post Date'],
                'Description': transaction_row['Description'],
                'Category': transaction_row['Category'],
                'Type': transaction_row['Type'],
                'Amount': transaction_row['Amount'],
                'Memo': transaction_row['Memo'],
                'Currency': transaction_row.get('Currency'),
                'Transaction Reference': transaction_row.get('Transaction Reference'),
                'FI Transaction Reference': transaction_row.get('FI Transaction Reference'),
                'Original Amount': transaction_row.get('Original Amount'),
                'source_file': 'manual_entry'
            }])

            inserted, skipped = db.upsert_transactions(pd.DataFrame([transaction_row]), account='manual_entry', source_file='manual_entry', profile_name='manual_entry')

            if inserted:
                st.success("Transaction added successfully!")
            else:
                st.info("Transaction already exists and was not added again.")

            if self._active_source_mode == "Database (all history)":
                self.data = db.fetch_transactions()
            else:
                if self.data is None or self.data.empty:
                    self.data = display_row
                else:
                    missing_cols = [col for col in display_row.columns if col not in self.data.columns]
                    for col in missing_cols:
                        self.data[col] = None
                    display_row = display_row.reindex(columns=self.data.columns, fill_value=None)
                    self.data = pd.concat([self.data, display_row], ignore_index=True)
                self._current_upload = self.data.copy()

            if self.data is not None and not self.data.empty:
                self.analytics = PersonalFinanceAnalytics(self.data)

            st.session_state.show_add_transaction = False
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
    
    def _render_transaction_editor(self) -> None:
        """Render transaction editor view with tabs for Transfers and Uncategorized."""
        st.header("✏️ Transaction Editor")
        st.markdown("Edit transfers, uncategorized transactions, and other transactions.")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🔄 Transfers", "❓ Uncategorized", "📋 All Transactions"])
        
        with tab1:
            self._render_transfers_tab()
        
        with tab2:
            self._render_uncategorized_tab()
        
        with tab3:
            self._render_all_transactions_tab()
    
    def _render_transfers_tab(self) -> None:
        """Render transfers tab."""
        st.subheader("🔄 Transfer Transactions")
        st.markdown("Review and categorize transfer transactions.")
        
        # Fetch transfers
        transfers_df = db.fetch_transactions_by_category('Transfers')
        if transfers_df.empty:
            transfers_df = db.fetch_transactions_by_category('Transfer')
        
        if transfers_df.empty:
            st.info("No transfer transactions found.")
            return
        
        st.info(f"Found {len(transfers_df)} transfer transactions.")
        
        # Display transactions in editable format
        for idx, row in transfers_df.iterrows():
            with st.expander(f"💰 {row.get('Description', 'Unknown')} - ${row.get('Amount', 0):,.2f} | {row.get('Transaction Date', 'N/A')}"):
                self._render_transaction_edit_form(row)
    
    def _render_uncategorized_tab(self) -> None:
        """Render uncategorized transactions tab."""
        st.subheader("❓ Uncategorized Transactions")
        st.markdown("Categorize transactions that haven't been assigned a category.")
        
        # Fetch uncategorized transactions
        uncategorized_df = db.fetch_uncategorized_transactions()
        
        if uncategorized_df.empty:
            st.success("✅ All transactions are categorized!")
            return
        
        st.warning(f"Found {len(uncategorized_df)} uncategorized transactions.")
        
        # Display transactions in editable format
        for idx, row in uncategorized_df.iterrows():
            with st.expander(f"❓ {row.get('Description', 'Unknown')} - ${row.get('Amount', 0):,.2f} | {row.get('Transaction Date', 'N/A')}"):
                self._render_transaction_edit_form(row)
    
    def _render_all_transactions_tab(self) -> None:
        """Render all transactions tab for editing."""
        st.subheader("📋 All Transactions")
        st.markdown("Edit any transaction in the database.")
        
        # Fetch all transactions (limited to recent 100 for performance)
        all_transactions = db.fetch_transactions()
        if not all_transactions.empty:
            # Show most recent first, limit to 100
            all_transactions = all_transactions.tail(100).iloc[::-1]
            st.info(f"Showing most recent 100 transactions (total: {len(db.fetch_transactions())})")
            
            # Search/filter
            search_term = st.text_input("🔍 Search transactions", placeholder="Search by description...")
            if search_term:
                all_transactions = all_transactions[
                    all_transactions['Description'].str.contains(search_term, case=False, na=False)
                ]
            
            # Display transactions
            for idx, row in all_transactions.iterrows():
                with st.expander(f"{row.get('Category', '❓')} | {row.get('Description', 'Unknown')} - ${row.get('Amount', 0):,.2f} | {row.get('Transaction Date', 'N/A')}"):
                    self._render_transaction_edit_form(row)
        else:
            st.info("No transactions found in database.")
    
    def _render_transaction_edit_form(self, transaction: pd.Series) -> None:
        """Render form to edit a single transaction."""
        transaction_id = transaction.get('id')
        if transaction_id is None:
            st.error("Transaction ID not found. Cannot edit this transaction.")
            return
        
        # Get current values
        current_category = transaction.get('Category', '')
        current_description = transaction.get('Description', '')
        current_amount = transaction.get('Amount', 0.0)
        current_memo = transaction.get('Memo', '')
        current_date = transaction.get('Transaction Date', '')
        
        # Get available categories
        available_categories = [''] + db.fetch_distinct_categories()
        # Add common categories if not present
        common_categories = ['Income', 'Transfers', 'Transfer', 'Dining', 'Groceries', 'Transportation', 
                           'Utilities', 'Shopping', 'Entertainment', 'Health', 'Housing', 'Travel', 'Other']
        for cat in common_categories:
            if cat not in available_categories:
                available_categories.append(cat)
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_category = st.selectbox(
                "Category",
                options=available_categories,
                index=available_categories.index(current_category) if current_category in available_categories else 0,
                key=f"cat_{transaction_id}"
            )
            
            new_description = st.text_input(
                "Description",
                value=str(current_description) if pd.notna(current_description) else '',
                key=f"desc_{transaction_id}"
            )
            
            new_amount = st.number_input(
                "Amount",
                value=float(current_amount) if pd.notna(current_amount) else 0.0,
                step=0.01,
                format="%.2f",
                key=f"amt_{transaction_id}"
            )
        
        with col2:
            new_memo = st.text_input(
                "Memo",
                value=str(current_memo) if pd.notna(current_memo) else '',
                key=f"memo_{transaction_id}"
            )
            
            if pd.notna(current_date):
                try:
                    if isinstance(current_date, str):
                        date_val = pd.to_datetime(current_date).date()
                    else:
                        date_val = current_date.date() if hasattr(current_date, 'date') else current_date
                    new_date = st.date_input(
                        "Transaction Date",
                        value=date_val,
                        key=f"date_{transaction_id}"
                    )
                except:
                    new_date = st.date_input(
                        "Transaction Date",
                        key=f"date_{transaction_id}"
                    )
            else:
                new_date = st.date_input(
                    "Transaction Date",
                    key=f"date_{transaction_id}"
                )
        
        # Save button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("💾 Save Changes", key=f"save_{transaction_id}"):
                updates = {}
                if new_category != current_category:
                    updates['category'] = new_category if new_category else None
                if new_description != current_description:
                    updates['description'] = new_description
                if abs(new_amount - float(current_amount)) > 0.01:
                    updates['amount'] = new_amount
                if new_memo != str(current_memo):
                    updates['memo'] = new_memo if new_memo else None
                if new_date and pd.notna(current_date):
                    try:
                        new_date_str = new_date.isoformat() if hasattr(new_date, 'isoformat') else str(new_date)
                        old_date_str = current_date.isoformat() if hasattr(current_date, 'isoformat') else str(current_date)
                        if new_date_str != old_date_str:
                            updates['transaction_date'] = new_date_str
                    except:
                        pass
                
                if updates:
                    success = db.update_transaction(transaction_id, **updates)
                    if success:
                        st.success("✅ Transaction updated successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to update transaction.")
                else:
                    st.info("No changes to save.")
        
        with col2:
            if st.button("↩️ Back to List", key=f"back_{transaction_id}"):
                st.rerun()
    
    def _handle_form_submissions(self) -> None:
        """Handle form submissions and update data."""
        # This method is called after rendering to handle any form submissions
        pass


if __name__ == "__main__":
    main()
