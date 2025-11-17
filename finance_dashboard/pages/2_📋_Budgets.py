"""Next-generation budget planning workspace.

This file has been refactored to use the modular library structure.
The original implementation is preserved in 2_📋_Budgets.py.backup for reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PARENT = Path(__file__).parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from shared_sidebar import render_shared_sidebar, apply_filters
from personal_finance_analytics import PersonalFinanceAnalytics

# Import library modules - handle both Streamlit page execution and direct execution
# Streamlit runs pages from the pages/ directory, so we need to add it to the path
PAGES_DIR = Path(__file__).parent
if str(PAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PAGES_DIR))

# Use importlib to load modules dynamically (works in Streamlit's execution context)
import importlib.util

# Import budgets module
budgets_init_path = PAGES_DIR / 'lib' / 'budgets' / '__init__.py'
if not budgets_init_path.exists():
    raise ImportError(f"Could not find budgets module at {budgets_init_path}")

spec = importlib.util.spec_from_file_location('lib.budgets', budgets_init_path)
budgets_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budgets_module)

load_saved_budgets = budgets_module.load_saved_budgets
get_default_envelopes = budgets_module.get_default_envelopes
ensure_savings_entry = budgets_module.ensure_savings_entry

# Import UI components
ui_components_path = PAGES_DIR / 'lib' / 'budgets' / 'ui_components.py'
if not ui_components_path.exists():
    raise ImportError(f"Could not find ui_components.py at {ui_components_path}")

spec = importlib.util.spec_from_file_location('lib.budgets.ui_components', ui_components_path)
ui_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui_module)

render_metrics = ui_module.render_metrics
render_percent_income_analysis = ui_module.render_percent_income_analysis
render_budget_name_input = ui_module.render_budget_name_input
render_delete_budget_section = ui_module.render_delete_budget_section
render_budget_selector = ui_module.render_budget_selector
render_unified_budget_system = ui_module.render_unified_budget_system
render_budget_insights = ui_module.render_budget_insights


def main() -> None:
    """Main entry point for the Budgets page."""
    st.set_page_config(page_title="Budgets", page_icon="📋", layout="wide")
    sidebar = render_shared_sidebar()
    base_df = sidebar['base_df']
    filters = sidebar['filters']
    filtered_df = apply_filters(base_df, filters) if not base_df.empty else pd.DataFrame()
    analytics = PersonalFinanceAnalytics(filtered_df) if not filtered_df.empty else None

    # Initialize session state with defaults
    budgets = st.session_state.setdefault('budgets', {})
    default_envelopes = get_default_envelopes()
    envelope_config = st.session_state.setdefault('envelope_groups_config', [dict(row) for row in default_envelopes])
    saved_budgets = load_saved_budgets()
    saved_names = sorted(saved_budgets.keys())
    
    # Determine selected budget name
    selected_budget_name = st.session_state.get('selected_budget_name', '')
    if selected_budget_name and selected_budget_name not in saved_budgets:
        # Selected budget no longer exists (was deleted), clear selection
        selected_budget_name = ''
        st.session_state.selected_budget_name = ''
    
    # Auto-select first saved budget if none selected
    if not selected_budget_name and saved_names:
        selected_budget_name = saved_names[0]
        st.session_state.selected_budget_name = selected_budget_name
    
    # Auto-load selected budget into session state if not already loaded
    # Load even if empty - we'll show guidance in the UI
    if selected_budget_name and selected_budget_name in saved_budgets:
        payload = saved_budgets[selected_budget_name]
        if isinstance(payload, dict):
            saved_budget = payload.get('budgets', {})
            if isinstance(saved_budget, dict):
                # Load budget values (even if empty) - normalize to floats
                normalized_budget = {k: float(v) for k, v in saved_budget.items() if k and v is not None}
                # Only update if session state budgets are empty or different
                if not budgets or budgets != normalized_budget:
                    budgets = normalized_budget
                    st.session_state.budgets = budgets
            
            # Load envelope configuration if available
            saved_envelopes = payload.get('envelopes', [])
            if saved_envelopes:
                envelope_config = ensure_savings_entry(saved_envelopes)
                st.session_state.envelope_groups_config = envelope_config
    
    # Render main page header and metrics
    st.header("📋 Budget Command Center")
    render_metrics(budgets, analytics)

    # Create tabs for different views
    analytics_tab, manage_tab, insights_tab = st.tabs([
        "📊 Analysis",
        "🛠 Manage Budgets",
        "📈 Budget Insights",
    ])

    with analytics_tab:
        render_percent_income_analysis(analytics)

    with manage_tab:
        # Show budget name input at top (no save button here)
        budget_name_input = render_budget_name_input()
        st.divider()
        render_unified_budget_system(budgets, envelope_config, analytics, budget_name_input, saved_budgets)
        st.divider()
        # Delete section at bottom
        saved_budgets = render_delete_budget_section(saved_budgets)

    with insights_tab:
        selected_budget_name, selected_budget = render_budget_selector(saved_budgets, selected_budget_name, current_budget=budgets)
        # Fallback: if selector returned empty budget but we have a selected name, try to load it
        if selected_budget_name and selected_budget_name != 'Current (unsaved)' and not selected_budget:
            payload = saved_budgets.get(selected_budget_name, {})
            if isinstance(payload, dict):
                selected_budget = payload.get('budgets', {})
                if not isinstance(selected_budget, dict):
                    selected_budget = {}
                else:
                    # Ensure budget values are floats
                    selected_budget = {k: float(v) for k, v in selected_budget.items() if k and v is not None}
                
                # Update session state
                st.session_state.budgets = selected_budget
                
                # Load envelope configuration
                saved_envelopes = payload.get('envelopes', [])
                if saved_envelopes:
                    st.session_state.envelope_groups_config = ensure_savings_entry(saved_envelopes)
        render_budget_insights(selected_budget, analytics, selected_budget_name)


if __name__ == '__main__':
    main()
