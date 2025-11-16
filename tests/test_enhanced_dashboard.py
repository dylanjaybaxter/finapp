#!/usr/bin/env python3
"""Test script for the Enhanced Personal Finance Dashboard.

This script tests the basic functionality of the enhanced dashboard
components without requiring Streamlit to be running.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from finance_dashboard.personal_finance_analytics import PersonalFinanceAnalytics


def test_enhanced_dashboard():
    """Test the enhanced dashboard functionality."""
    print("🧪 Testing Enhanced Personal Finance Dashboard...")
    
    # Load sample data
    sample_data_path = project_root / "sample_data" / "Chase1337_Activity20230913_20250913_20250914.CSV"
    
    if not sample_data_path.exists():
        print("❌ Sample data not found. Please ensure the sample CSV file exists.")
        return False
    
    try:
        # Load data
        data = pd.read_csv(sample_data_path)
        print(f"✅ Loaded {len(data)} transactions from sample data")
        
        # Initialize analytics
        analytics = PersonalFinanceAnalytics(data)
        print("✅ Personal Finance Analytics initialized")
        
        # Test monthly summary
        monthly_summary = analytics.calculate_monthly_summary()
        print(f"✅ Monthly summary calculated: Income=${monthly_summary['income']:,.2f}, Expenses=${monthly_summary['expenses']:,.2f}")
        
        # Test category spending
        category_spending = analytics.calculate_category_spending()
        print(f"✅ Category spending calculated for {len(category_spending)} categories")
        
        # Test financial health metrics
        health_metrics = analytics.calculate_financial_health_metrics()
        print(f"✅ Financial health metrics calculated: Savings Rate={health_metrics['savings_rate']:.1f}%")
        
        # Test spending patterns
        patterns = analytics.calculate_spending_patterns()
        print(f"✅ Spending patterns analyzed: {patterns['total_transactions']} transactions")
        
        # Test budget performance (with sample budgets)
        sample_budgets = {
            'Food & Drink': 500.0,
            'Shopping': 300.0,
            'Gas': 200.0
        }
        budget_performance = analytics.calculate_budget_performance(sample_budgets)
        print(f"✅ Budget performance calculated for {len(budget_performance)} budget categories")
        
        # Test goal progress (with sample goals)
        sample_goals = [
            {
                'name': 'Emergency Fund',
                'target_amount': 5000.0,
                'current_amount': 2500.0
            },
            {
                'name': 'Vacation Fund',
                'target_amount': 2000.0,
                'current_amount': 800.0
            }
        ]
        goal_progress = analytics.calculate_goal_progress(sample_goals)
        print(f"✅ Goal progress calculated for {len(goal_progress)} goals")
        
        # Test spending insights
        insights = analytics.generate_spending_insights()
        print(f"✅ Generated {len(insights)} spending insights")
        
        # Test anomaly detection
        anomalies = analytics.detect_spending_anomalies()
        print(f"✅ Detected {len(anomalies)} spending anomalies")
        
        print("\n🎉 All tests passed! The Enhanced Personal Finance Dashboard is working correctly.")
        print("\nTo run the dashboard:")
        print("1. Run: make run (or bash scripts/run_enhanced_dashboard_module.sh)")
        print("2. Open: http://localhost:8501")
        print("3. Upload your bank statement or use the sample data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_enhanced_dashboard()
    sys.exit(0 if success else 1)

