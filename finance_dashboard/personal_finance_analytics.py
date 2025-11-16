"""Personal Finance Analytics and Data Processing.

This module contains specialized functions for personal finance analysis,
including budget tracking, spending patterns, financial goal management,
and personal finance-specific calculations.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

TRANSFER_CATEGORY_LABELS = {'transfer', 'transfers', 'internal transfer'}
TRANSFER_TYPE_LABELS = {'transfer'}
HOUSING_CATEGORY_LABELS = {
    'rent', 'rent/mortgage', 'housing', 'mortgage', 'housing & rent', 'rent expense'
}
HOUSING_KEYWORDS = {
    'apartment', 'apartments', 'apt', 'enclave', 'rent', 'lease', 'mortgage'
}
HOUSING_KEYWORD_PATTERN = '|'.join(sorted(HOUSING_KEYWORDS))


class PersonalFinanceAnalytics:
    """Personal finance analytics and calculations."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with transaction data."""
        self.data = data.copy()
        self._prepare_data()
    
    def _prepare_data(self) -> None:
        """Prepare data for analysis."""
        # Convert date columns to datetime
        if 'Transaction Date' in self.data.columns:
            self.data['Transaction Date'] = pd.to_datetime(self.data['Transaction Date'])
        if 'Post Date' in self.data.columns:
            self.data['Post Date'] = pd.to_datetime(self.data['Post Date'])
        
        # Ensure Amount is numeric
        self.data['Amount'] = pd.to_numeric(self.data['Amount'], errors='coerce').fillna(0.0)

        # Normalize category/type text so downstream filters behave consistently
        self.data['Category'] = (
            self.data.get('Category', 'Uncategorized')
            .fillna('Uncategorized')
            .astype(str)
        )
        type_series = self.data.get('Type')
        if isinstance(type_series, pd.Series):
            self.data['Type'] = type_series.fillna('').astype(str)
        else:
            self.data['Type'] = ''

        lowered_category = self.data['Category'].str.strip().str.lower()
        lowered_type = self.data['Type'].str.strip().str.lower()
        desc_series = self.data.get('Description')
        if isinstance(desc_series, pd.Series):
            description_lower = desc_series.fillna('').astype(str).str.lower()
        else:
            description_lower = pd.Series('', index=self.data.index)

        is_transfer = lowered_category.isin(TRANSFER_CATEGORY_LABELS)
        uncategorized_mask = lowered_category.isin({'', 'uncategorized', 'none'})
        is_transfer |= uncategorized_mask & lowered_type.isin(TRANSFER_TYPE_LABELS)

        housing_like = lowered_category.isin(HOUSING_CATEGORY_LABELS)
        if HOUSING_KEYWORD_PATTERN:
            housing_like |= description_lower.str.contains(HOUSING_KEYWORD_PATTERN, na=False)
        is_transfer = np.where(housing_like, False, is_transfer)
        is_income = (self.data['Amount'] > 0) & ~is_transfer

        self.data['Flow Category'] = 'Expense'
        self.data.loc[is_income, 'Flow Category'] = 'Income'
        self.data.loc[is_transfer, 'Flow Category'] = 'Transfer'
        
        # Create additional time-based columns
        if 'Transaction Date' in self.data.columns:
            self.data['Year'] = self.data['Transaction Date'].dt.year
            self.data['Month'] = self.data['Transaction Date'].dt.month
            self.data['Day'] = self.data['Transaction Date'].dt.day
            self.data['Weekday'] = self.data['Transaction Date'].dt.day_name()
            self.data['Month_Name'] = self.data['Transaction Date'].dt.month_name()

    def _expense_rows(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        source = df if df is not None else self.data
        if 'Flow Category' not in source.columns:
            tmp = source.copy()
            tmp['Flow Category'] = np.where(tmp.get('Amount', 0) >= 0, 'Income', 'Expense')
            source = tmp
        expenses = source[source['Flow Category'] == 'Expense'].copy()
        if expenses.empty:
            return expenses
        mask = ~expenses['Category'].str.lower().isin({'income', 'transfer', 'transfers'})
        return expenses[mask]

    def _income_rows(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        source = df if df is not None else self.data
        if 'Flow Category' not in source.columns:
            tmp = source.copy()
            tmp['Flow Category'] = np.where(tmp.get('Amount', 0) >= 0, 'Income', 'Expense')
            source = tmp
        return source[source['Flow Category'] == 'Income'].copy()
    
    def calculate_monthly_summary(self, year: int = None, month: int = None) -> Dict[str, float]:
        """Calculate monthly financial summary."""
        if year is None:
            year = datetime.now().year
        if month is None:
            month = datetime.now().month
        
        monthly_data = self.data[
            (self.data['Year'] == year) & 
            (self.data['Month'] == month)
        ]
        
        income = self._income_rows(monthly_data)['Amount'].sum()
        expenses = abs(self._expense_rows(monthly_data)['Amount'].sum())
        net_flow = income - expenses
        savings_rate = (net_flow / income * 100) if income > 0 else 0
        
        return {
            'income': income,
            'expenses': expenses,
            'net_flow': net_flow,
            'savings_rate': savings_rate,
            'transaction_count': len(monthly_data)
        }
    
    def calculate_period_summary(self, start_date: str = None, end_date: str = None) -> Dict:
        """Calculate financial summary for a specific time period."""
        filtered_data = self._filter_by_date_range(start_date, end_date)

        income = self._income_rows(filtered_data)['Amount'].sum()
        expense_rows = self._expense_rows(filtered_data)
        expense_rows = expense_rows[~expense_rows['Category'].str.lower().isin({'income', 'transfer', 'transfers'})]
        expenses = abs(expense_rows['Amount'].sum())
        net_flow = income - expenses
        savings_rate = (net_flow / income * 100) if income > 0 else 0

        # Calculate largest transactions (true expenses only)
        largest_expenses = expense_rows[
            ~expense_rows['Category'].str.lower().isin(HOUSING_CATEGORY_LABELS)
        ].copy()
        if not largest_expenses.empty:
            largest_expenses['Amount'] = largest_expenses['Amount'].abs()
            largest_expenses = largest_expenses.sort_values('Amount', ascending=False).head(10)

        return {
            'income': income,
            'expenses': expenses,
            'net_flow': net_flow,
            'savings_rate': savings_rate,
            'transaction_count': len(filtered_data),
            'largest_expenses': largest_expenses
        }
    
    def calculate_monthly_breakdown(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Calculate monthly income/expenses breakdown for interactive panel."""
        filtered_data = self._filter_by_date_range(start_date, end_date)
        
        # Create year and month columns to avoid conflicts
        filtered_data = filtered_data.copy()
        filtered_data['Year_Col'] = filtered_data['Transaction Date'].dt.year
        filtered_data['Month_Col'] = filtered_data['Transaction Date'].dt.month
        
        # Group by month using the new column names
        monthly_data = filtered_data.groupby(['Year_Col', 'Month_Col']).agg({
            'Amount': lambda x: {
                'income': x[x > 0].sum(),
                'expenses': abs(x[x < 0].sum()),
                'net': x.sum(),
                'transaction_count': len(x)
            }
        })
        
        # Reset index and rename columns
        monthly_data = monthly_data.reset_index()
        monthly_data.columns = ['Year', 'Month', 'Amount']
        
        # Expand the Amount dictionary into separate columns
        monthly_data['Income'] = monthly_data['Amount'].apply(lambda x: x['income'])
        monthly_data['Expenses'] = monthly_data['Amount'].apply(lambda x: x['expenses'])
        monthly_data['Net'] = monthly_data['Amount'].apply(lambda x: x['net'])
        monthly_data['Transaction_Count'] = monthly_data['Amount'].apply(lambda x: x['transaction_count'])
        
        # Create month-year labels
        monthly_data['Month_Label'] = monthly_data.apply(
            lambda row: f"{row['Year']}-{row['Month']:02d}", axis=1
        )
        
        return monthly_data[['Month_Label', 'Year', 'Month', 'Income', 'Expenses', 'Net', 'Transaction_Count']]
    
    def get_transactions_by_month(self, year: int, month: int) -> pd.DataFrame:
        """Get all transactions for a specific month."""
        monthly_data = self.data[
            (self.data['Year'] == year) & 
            (self.data['Month'] == month)
        ].copy()
        
        # Sort by date and amount
        monthly_data = monthly_data.sort_values(['Transaction Date', 'Amount'])
        
        return monthly_data
    
    def get_transactions_by_category(self, category: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get all transactions for a specific category."""
        filtered_data = self._filter_by_date_range(start_date, end_date)
        category_data = filtered_data[filtered_data['Category'] == category].copy()
        
        # Sort by date and amount
        category_data = category_data.sort_values(['Transaction Date', 'Amount'])
        
        return category_data
    
    def calculate_category_spending(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Calculate spending by category for a date range.
        
        Excludes Income and Transfers from spending calculations.
        """
        filtered_data = self._filter_by_date_range(start_date, end_date)
        expenses = self._expense_rows(filtered_data)
        expenses['Amount'] = expenses['Amount'].abs()
        
        if expenses.empty:
            return pd.DataFrame(columns=['Total_Spent', 'Transaction_Count', 'Avg_Transaction', 'First_Transaction', 'Last_Transaction'])
        
        category_spending = expenses.groupby('Category').agg({
            'Amount': ['sum', 'count', 'mean'],
            'Transaction Date': ['min', 'max']
        }).round(2)
        
        category_spending.columns = ['Total_Spent', 'Transaction_Count', 'Avg_Transaction', 'First_Transaction', 'Last_Transaction']
        category_spending = category_spending.sort_values('Total_Spent', ascending=False)
        
        return category_spending
    
    def calculate_spending_trends(self, category: str = None, period: str = 'monthly') -> pd.DataFrame:
        """Calculate spending trends over time.
        
        Excludes Income and Transfers from spending calculations.
        """
        data = self.data.copy()
        if category:
            data = data[data['Category'] == category]
        
        expenses = self._expense_rows(data)
        expenses['Amount'] = expenses['Amount'].abs()
        
        if period == 'monthly':
            expenses['Period'] = expenses['Transaction Date'].dt.to_period('M')
        elif period == 'weekly':
            expenses['Period'] = expenses['Transaction Date'].dt.to_period('W')
        elif period == 'daily':
            expenses['Period'] = expenses['Transaction Date'].dt.date
        
        trends = expenses.groupby('Period')['Amount'].sum().reset_index()
        trends['Period'] = trends['Period'].astype(str)
        
        return trends
    
    def calculate_budget_performance(self, budgets: Dict[str, float], year: int = None, month: int = None) -> pd.DataFrame:
        """Calculate budget performance vs actual spending.
        
        Excludes Income and Transfers from spending calculations.
        """
        if year is None:
            year = datetime.now().year
        if month is None:
            month = datetime.now().month
        
        monthly_data = self.data[
            (self.data['Year'] == year) & 
            (self.data['Month'] == month)
        ]
        
        monthly_data = self._expense_rows(monthly_data)
        
        budget_performance = []
        
        # Categories to exclude from budget performance calculations
        exclude_categories = {'income', 'transfer', 'transfers'}
        
        for category, budget_amount in budgets.items():
            # Skip if category is Income or Transfer (case-insensitive)
            if category.lower() in exclude_categories:
                continue
                
            category_mask = monthly_data['Category'] == category
            actual_spending = abs(
                monthly_data.loc[category_mask & (monthly_data['Amount'] < 0), 'Amount'].sum()
            )
            remaining = budget_amount - actual_spending
            percentage_used = (actual_spending / budget_amount * 100) if budget_amount > 0 else 0
            status = 'Over Budget' if actual_spending > budget_amount else 'Under Budget'

            budget_performance.append({
                'Category': category,
                'Budget': budget_amount,
                'Actual': actual_spending,
                'Remaining': remaining,
                'Percentage_Used': percentage_used,
                'Status': status
            })
        
        return pd.DataFrame(budget_performance)
    
    def calculate_financial_health_metrics(self) -> Dict[str, float]:
        """Calculate personal finance health metrics."""
        # Get last 12 months of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        recent_data = self.data[
            (self.data['Transaction Date'] >= start_date) & 
            (self.data['Transaction Date'] <= end_date)
        ]

        if recent_data.empty:
            recent_data = self.data.copy()

        if recent_data.empty:
            return {
                'total_income_12m': 0.0,
                'total_expenses_12m': 0.0,
                'net_worth_change_12m': 0.0,
                'monthly_income_avg': 0.0,
                'monthly_expenses_avg': 0.0,
                'monthly_savings_avg': 0.0,
                'savings_rate': 0.0,
                'top_expense_category': 'N/A',
                'top_expense_amount': 0.0,
                'spending_consistency': 0.0
            }

        # Calculate metrics
        total_income = self._income_rows(recent_data)['Amount'].sum()
        total_expenses = abs(self._expense_rows(recent_data)['Amount'].sum())
        net_worth_change = total_income - total_expenses

        # Monthly averages
        month_count = recent_data['Transaction Date'].dt.to_period('M').nunique()
        month_count = max(month_count, 1)
        monthly_income = total_income / month_count
        monthly_expenses = total_expenses / month_count
        monthly_savings = monthly_income - monthly_expenses

        # Savings rate
        savings_rate = (monthly_savings / monthly_income * 100) if monthly_income > 0 else 0

        # Expense categories analysis (exclude Income and Transfers)
        expense_data = self._expense_rows(recent_data)
        category_expenses = expense_data.groupby('Category')['Amount'].sum().abs()
        top_expense_category = category_expenses.idxmax() if len(category_expenses) > 0 else 'N/A'
        top_expense_amount = category_expenses.max() if len(category_expenses) > 0 else 0
        
        # Spending consistency (coefficient of variation)
        monthly_spending = self._expense_rows(recent_data).groupby(
            recent_data['Transaction Date'].dt.to_period('M')
        )['Amount'].sum().abs()

        spending_consistency = 0.0
        if len(monthly_spending) > 1 and monthly_spending.mean() != 0:
            spending_consistency = (monthly_spending.std() / monthly_spending.mean()) * 100

        return {
            'total_income_12m': total_income,
            'total_expenses_12m': total_expenses,
            'net_worth_change_12m': net_worth_change,
            'monthly_income_avg': monthly_income,
            'monthly_expenses_avg': monthly_expenses,
            'monthly_savings_avg': monthly_savings,
            'savings_rate': savings_rate,
            'top_expense_category': top_expense_category,
            'top_expense_amount': top_expense_amount,
            'spending_consistency': spending_consistency
        }
    
    def calculate_spending_patterns(self) -> Dict[str, Any]:
        """Analyze spending patterns and behaviors."""
        expenses = self._expense_rows()
        expenses['Amount'] = expenses['Amount'].abs()
        
        # Day of week patterns
        weekday_spending = expenses.groupby('Weekday')['Amount'].sum().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]).fillna(0.0)

        # Monthly patterns
        monthly_spending = expenses.groupby('Month_Name')['Amount'].sum().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]).fillna(0.0)

        # Time of day patterns (if we had time data)
        # For now, we'll use transaction count as proxy
        # Spending frequency
        avg_daily_spending_series = expenses.groupby(expenses['Transaction Date'].dt.date)['Amount'].sum()
        avg_daily_spending = avg_daily_spending_series.mean()
        if pd.isna(avg_daily_spending):
            avg_daily_spending = 0.0
        days_span = 0
        if not expenses.empty:
            days_span = (expenses['Transaction Date'].max() - expenses['Transaction Date'].min()).days + 1
        days_span = max(days_span, 1)
        spending_frequency = len(expenses) / days_span

        # Large transaction analysis
        large_transactions = expenses[expenses['Amount'] > expenses['Amount'].quantile(0.95)]
        
        return {
            'weekday_spending': weekday_spending,
            'monthly_spending': monthly_spending,
            'avg_daily_spending': avg_daily_spending,
            'spending_frequency': spending_frequency,
            'large_transactions': large_transactions,
            'total_transactions': len(expenses)
        }
    
    def calculate_goal_progress(self, goals: List[Dict]) -> List[Dict]:
        """Calculate progress towards financial goals."""
        goal_progress = []

        latest_summary = None
        if 'Transaction Date' in self.data.columns and not self.data.empty:
            latest_txn = self.data['Transaction Date'].max()
            latest_summary = self.calculate_monthly_summary(
                year=int(latest_txn.year),
                month=int(latest_txn.month)
            )

        for goal in goals:
            # Calculate current progress based on savings
            current_amount = goal.get('current_amount', 0)
            target_amount = goal.get('target_amount', 0)
            
            if target_amount > 0:
                progress_percentage = (current_amount / target_amount) * 100
                remaining_amount = target_amount - current_amount

                # Calculate time to goal based on latest available monthly savings
                monthly_savings = latest_summary['net_flow'] if latest_summary else 0
                months_to_goal = remaining_amount / monthly_savings if monthly_savings > 0 else float('inf')

                goal_progress.append({
                    'name': goal.get('name', 'Unnamed Goal'),
                    'current_amount': current_amount,
                    'target_amount': target_amount,
                    'progress_percentage': min(progress_percentage, 100),
                    'remaining_amount': remaining_amount,
                    'months_to_goal': months_to_goal,
                    'status': 'Completed' if progress_percentage >= 100 else 'In Progress'
                })
        
        return goal_progress
    
    def detect_spending_anomalies(self, threshold: float = 2.0) -> pd.DataFrame:
        """Detect unusual spending patterns."""
        expenses = self.data[self.data['Amount'] < 0].copy()
        expenses['Amount'] = expenses['Amount'].abs()
        
        # Calculate z-scores for each category
        anomalies = []
        
        for category in expenses['Category'].unique():
            category_data = expenses[expenses['Category'] == category]['Amount']
            
            if len(category_data) > 1:
                mean_amount = category_data.mean()
                std_amount = category_data.std()
                
                if std_amount > 0:
                    z_scores = (category_data - mean_amount) / std_amount
                    anomaly_mask = abs(z_scores) > threshold
                    
                    if anomaly_mask.any():
                        anomaly_transactions = expenses[
                            (expenses['Category'] == category) & 
                            anomaly_mask
                        ]
                        
                        for _, transaction in anomaly_transactions.iterrows():
                            anomalies.append({
                                'Date': transaction['Transaction Date'],
                                'Description': transaction['Description'],
                                'Category': category,
                                'Amount': transaction['Amount'],
                                'Z_Score': z_scores[transaction.name],
                                'Expected_Range': f"${mean_amount - std_amount:.2f} - ${mean_amount + std_amount:.2f}"
                            })
        
        return pd.DataFrame(anomalies)
    
    def calculate_debt_analysis(self) -> Dict[str, float]:
        """Calculate debt-related metrics."""
        # This is a simplified version - in reality, you'd need more specific debt data
        expenses = self._expense_rows()
        
        # Identify potential debt payments (recurring negative amounts)
        recurring_expenses = expenses.groupby('Description')['Amount'].sum()
        debt_payments = recurring_expenses[recurring_expenses < -100]  # Regular payments > $100
        
        # Calculate debt-to-income ratio (simplified)
        monthly_income = self.calculate_monthly_summary()['income']
        monthly_debt_payments = abs(debt_payments.sum()) if len(debt_payments) > 0 else 0
        debt_to_income_ratio = (monthly_debt_payments / monthly_income * 100) if monthly_income > 0 else 0
        
        return {
            'monthly_debt_payments': monthly_debt_payments,
            'debt_to_income_ratio': debt_to_income_ratio,
            'debt_payment_count': len(debt_payments),
            'avg_debt_payment': abs(debt_payments.mean()) if len(debt_payments) > 0 else 0
        }
    
    def generate_spending_insights(self) -> List[str]:
        """Generate actionable spending insights."""
        insights = []
        
        # Get recent data
        recent_data = self.data[self.data['Transaction Date'] >= datetime.now() - timedelta(days=30)]
        
        # Category spending analysis
        recent_expenses = self._expense_rows(recent_data)
        category_spending = recent_expenses.groupby('Category')['Amount'].sum().abs()
        top_category = category_spending.idxmax() if len(category_spending) > 0 else None
        
        if top_category:
            insights.append(f"Your biggest expense category this month is {top_category} (${category_spending[top_category]:,.2f})")
        
        # Spending trend analysis
        if len(category_spending) > 1:
            second_highest = category_spending.nlargest(2).iloc[1]
            insights.append(f"Consider reducing spending in {category_spending.nlargest(2).index[1]} (${second_highest:,.2f})")
        
        # Savings rate analysis
        monthly_summary = self.calculate_monthly_summary()
        savings_rate = monthly_summary['savings_rate']
        
        if savings_rate < 10:
            insights.append("Your savings rate is below 10%. Consider increasing your savings goal.")
        elif savings_rate > 20:
            insights.append("Great job! Your savings rate is above 20%.")
        
        # Spending frequency analysis
        daily_spending = recent_expenses.groupby(recent_expenses['Transaction Date'].dt.date).size()
        avg_daily_transactions = daily_spending.mean()
        
        if avg_daily_transactions > 3:
            insights.append("You're making many small transactions. Consider consolidating purchases to save on fees.")
        
        return insights
    
    def _filter_by_date_range(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Filter data by date range."""
        data = self.data.copy()
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data['Transaction Date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data['Transaction Date'] <= end_date]
        
        return data
    
    def export_analysis_report(self, filename: str = None) -> str:
        """Export comprehensive analysis report."""
        if filename is None:
            filename = f"personal_finance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Generate comprehensive report
        report_data = []

        if 'Transaction Date' in self.data.columns and not self.data.empty:
            monthly_periods = (
                self.data['Transaction Date']
                .dt.to_period('M')
                .sort_values()
                .unique()
            )
        else:
            monthly_periods = []

        # Monthly summaries for up to the last 12 distinct periods
        for period in monthly_periods[-12:]:
            summary = self.calculate_monthly_summary(year=int(period.year), month=int(period.month))
            report_data.append({
                'Period': f"{period.year}-{period.month:02d}",
                'Type': 'Monthly Summary',
                'Income': summary['income'],
                'Expenses': summary['expenses'],
                'Net Flow': summary['net_flow'],
                'Savings Rate': summary['savings_rate']
            })

        # Category spending
        category_spending = self.calculate_category_spending()
        for category, row in category_spending.iterrows():
            report_data.append({
                'Period': 'All Time',
                'Type': 'Category Spending',
                'Category': category,
                'Total Spent': row['Total_Spent'],
                'Transaction Count': row['Transaction_Count'],
                'Avg Transaction': row['Avg_Transaction']
            })
        
        # Convert to DataFrame and save
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(filename, index=False)
        
        return filename
