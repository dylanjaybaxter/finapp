"""Goals Page - Financial goal tracking and management."""

from __future__ import annotations

import streamlit as st
import pandas as pd

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
    """Render the Goals page."""
    st.set_page_config(page_title="Goals", page_icon="🎯", layout="wide")
    
    # Render shared sidebar
    sidebar_data = render_shared_sidebar()
    base_df = sidebar_data['base_df']
    filters = sidebar_data['filters']
    
    # Initialize UI
    ui = PersonalFinanceUI()
    
    # Load goals from session state
    if 'goals' not in st.session_state:
        st.session_state.goals = []
    goals = st.session_state.goals
    
    st.header("🎯 Financial Goals")
    
    # Goal setting form
    st.subheader("Create New Goal")
    new_goal = ui.render_goal_setting_form()
    if new_goal:
        goals.append(new_goal)
        st.session_state.goals = goals
        st.success("Goal added successfully!")
        st.rerun()
    
    # Current goals
    if goals:
        st.subheader("Current Goals")
        for i, goal in enumerate(goals):
            with st.expander(f"Goal: {goal['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Target Amount", f"${goal['target_amount']:,.2f}")
                    st.metric("Current Amount", f"${goal['current_amount']:,.2f}")
                with col2:
                    remaining = goal['target_amount'] - goal['current_amount']
                    progress = goal['current_amount'] / goal['target_amount'] if goal['target_amount'] > 0 else 0
                    st.metric("Remaining", f"${remaining:,.2f}")
                    st.progress(min(progress, 1.0))
                
                # Delete button
                if st.button("🗑️ Delete Goal", key=f"delete_goal_{i}"):
                    goals.pop(i)
                    st.session_state.goals = goals
                    st.rerun()
        
        # Goal progress analysis
        if not base_df.empty:
            filtered_data = apply_filters(base_df, filters)
            if not filtered_data.empty:
                analytics = PersonalFinanceAnalytics(filtered_data)
                goal_progress = analytics.calculate_goal_progress(goals)
                st.subheader("Goal Progress Analysis")
                progress_df = pd.DataFrame(goal_progress)
                st.dataframe(progress_df, use_container_width=True)
    else:
        st.info("No goals set. Use the form above to create your first financial goal.")


# Streamlit will execute this file directly
main()

