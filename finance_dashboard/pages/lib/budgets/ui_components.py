"""UI components for budget page rendering.

This module provides all Streamlit UI rendering functions for the budgets page,
organized into logical sections: metrics, selectors, editors, and insights.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import analytics
try:
    from ...personal_finance_analytics import PersonalFinanceAnalytics
except ImportError:
    import sys
    from pathlib import Path as _Path
    parent_dir = _Path(__file__).resolve().parents[3]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from finance_dashboard.personal_finance_analytics import PersonalFinanceAnalytics

# Import budget modules
from .storage import load_saved_budgets, delete_budget, save_budget
from .envelopes import (
    ensure_savings_entry,
    finalize_envelope_config,
    groups_from_envelopes,
    targets_from_envelopes,
    get_default_envelopes,
)
from .categorization import infer_group
from .calculations import (
    suggested_amounts,
    expand_group_budgets,
    aggregate_budgets_by_group,
    budget_performance_snapshot,
    budget_trend_dataframe,
    category_percent_income,
    percent_income_summary,
    monthly_income_vs_spend,
    monthly_group_targets_actuals,
)
from ..analytics.income import income_series_complete, three_month_avg_income
from ..common.formatting import escape_dollar_for_markdown

# Import db for category fetching
try:
    from ... import db
except ImportError:
    import sys
    from pathlib import Path as _Path
    parent_dir = _Path(__file__).resolve().parents[3]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from finance_dashboard import db

# Import config
try:
    from ...config import get_budget_config
except ImportError:
    import sys
    from pathlib import Path as _Path
    parent_dir = _Path(__file__).resolve().parents[3]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from pages.config import get_budget_config


def _get_config() -> Dict[str, Any]:
    """Get budget configuration."""
    return get_budget_config()


def _get_constants() -> Dict[str, Any]:
    """Get constants from configuration."""
    return _get_config()['constants']


def _get_default_envelopes() -> List[Dict[str, Any]]:
    """Get default envelopes from configuration."""
    from .envelopes import get_default_envelopes
    return get_default_envelopes()


def _fetch_categories() -> List[str]:
    """Fetch all distinct categories from the database."""
    records = db.fetch_distinct_categories()
    return [c for c in records if c]


# ============================================================================
# Metrics and Analysis Rendering
# ============================================================================

def render_metrics(budgets: Dict[str, float], analytics: Optional[PersonalFinanceAnalytics]) -> None:
    """Render budget performance metrics.
    
    Args:
        budgets: Dictionary mapping category names to budget amounts
        analytics: Analytics object with transaction data
    """
    if not budgets or analytics is None:
        st.info("Create a budget and load transactions to unlock analytics.")
        return
    
    performance = analytics.calculate_budget_performance(budgets)
    total_budget = sum(budgets.values())
    total_actual = performance['Actual'].sum() if not performance.empty else 0.0
    variance = total_budget - total_actual
    overspent_count = int((performance['Status'] == 'Over Budget').sum())
    
    cols = st.columns(4)
    cols[0].metric("Budgeted", f"${total_budget:,.2f}")
    cols[1].metric("Actual", f"${total_actual:,.2f}")
    cols[2].metric("Variance", f"${variance:,.2f}", delta_color="inverse")
    cols[3].metric("Categories over budget", overspent_count)


def render_percent_income_analysis(analytics: Optional[PersonalFinanceAnalytics]) -> None:
    """Render percent of income analysis by category.
    
    Args:
        analytics: Analytics object with transaction data
    """
    st.subheader("Percent of income by category")
    if analytics is None or analytics.data.empty:
        st.info("Load transactions to estimate category share of income.")
        return
    
    incomes = income_series_complete(analytics)
    if incomes.empty:
        st.info("Need historical income to compute percentages.")
        return
    
    latest_month = incomes.index.max()
    last_three = incomes.tail(3)
    latest_value = float(last_three.iloc[-1]) if not last_three.empty else 0.0
    avg_last_three = float(last_three.mean()) if not last_three.empty else latest_value
    avg_all = float(incomes.mean())
    
    cols = st.columns(3)
    cols[0].metric("Most recent month", f"${latest_value:,.2f}")
    cols[1].metric("3-month avg", f"${avg_last_three:,.2f}")
    cols[2].metric("All-time avg", f"${avg_all:,.2f}")
    
    df = category_percent_income(analytics, incomes)
    if df.empty:
        st.info("Need at least one complete month of income and expense data.")
        return
    
    months = sorted(df['Month'].unique())
    last_months = months[-3:]
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Last 3 complete months")
        summary_recent = percent_income_summary(df[df['Month'].isin(last_months)], incomes)
        if summary_recent.empty:
            st.info("Not enough complete months yet.")
        else:
            st.dataframe(
                summary_recent.style.format({'Amount': '${:,.0f}', 'Percent': '{:,.1f}%'}),
                use_container_width=True,
                hide_index=True,
            )
    with col2:
        st.caption("All-time average")
        summary_all = percent_income_summary(df, incomes)
        st.dataframe(
            summary_all.style.format({'Amount': '${:,.0f}', 'Percent': '{:,.1f}%'}),
            use_container_width=True,
            hide_index=True,
        )
    
    st.caption("Top categories over the last 3 months")
    chart_df = summary_recent.head(10) if not summary_recent.empty else summary_all.head(10)
    if not chart_df.empty:
        fig = px.bar(
            chart_df,
            x='Category',
            y='Percent',
            text=chart_df['Amount'].apply(lambda x: f"${x:,.0f}"),
            title='Share of income',
            labels={'Percent': '% of income'}
        )
        fig.update_layout(yaxis_tickformat='.1f%%', uniformtext_minsize=10, uniformtext_mode='hide')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Budget Selection and Management UI
# ============================================================================

def render_budget_name_input() -> str:
    """Render budget name input field.
    
    Returns:
        The budget name entered by the user
    """
    config = _get_config()
    default_name = config.get('constants', {}).get('default_budget_name', 'My Budget')
    
    st.subheader("Budget Name")
    current_name = st.session_state.get('selected_budget_name', default_name)
    saved_budgets = load_saved_budgets()
    
    if current_name and current_name in saved_budgets:
        default_value = current_name
    else:
        default_value = default_name
    
    ui_config = config.get('ui', {})
    labels = ui_config.get('labels', {})
    help_text = ui_config.get('help_text', {})
    
    budget_name = st.text_input(
        labels.get('budget_name_input', "Name for this budget"),
        value=default_value,
        key='save_budget_name',
        help=help_text.get('budget_name', "Enter a name to save your budget configuration")
    )
    return budget_name.strip() if budget_name else default_name


def render_delete_budget_section(saved: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Render delete budget section at the bottom.
    
    Args:
        saved: Dictionary of saved budgets
        
    Returns:
        Updated dictionary of saved budgets (reloaded after delete operations)
    """
    config = _get_config()
    ui_config = config.get('ui', {})
    labels = ui_config.get('labels', {})
    messages = ui_config.get('messages', {})
    
    st.subheader("🗑️ Delete Saved Budgets")
    if not saved:
        st.info("No saved budgets to delete.")
        return saved
    
    col1, col2 = st.columns([3, 1])
    with col1:
        to_delete = st.selectbox(
            "Select a budget to delete",
            options=["(none)"] + sorted(saved.keys()),
            index=0,
            key='delete_budget_select',
            help="Choose a saved budget to permanently delete"
        )
    with col2:
        delete_label = labels.get('delete_button', 'Delete')
        if st.button(delete_label, key='delete_budget_button', type="secondary", use_container_width=True):
            if to_delete != "(none)":
                try:
                    delete_budget(to_delete)
                    if st.session_state.get('selected_budget_name') == to_delete:
                        st.session_state.selected_budget_name = ''
                        st.session_state.budgets = {}
                    saved = load_saved_budgets()
                    deleted_msg = messages.get('budget_deleted', "🗑️ Deleted budget '{name}'").format(name=to_delete)
                    st.warning(deleted_msg)
                    st.rerun()
                except OSError as e:
                    st.error(f"Failed to delete budget: {e}")
    
    return saved


def render_budget_selector(
    saved: Dict[str, Dict[str, Any]],
    selected_name: str,
    current_budget: Dict[str, float]
) -> Tuple[str, Dict[str, float]]:
    """Render budget selector and load selected budget into session state.
    
    Args:
        saved: Dictionary of saved budgets from disk
        selected_name: Currently selected budget name
        current_budget: Current unsaved budget from session state
    
    Returns:
        Tuple of (selected_name, budget_dict)
    """
    st.subheader("Select budget for insights")
    
    # Build options with indicators for empty budgets
    options = []
    option_labels = []
    
    if current_budget:
        options.append('Current (unsaved)')
        budget_count = len(current_budget)
        label = f"Current (unsaved)" + (f" - {budget_count} categories" if budget_count > 0 else " - empty")
        option_labels.append(label)
    
    for name in sorted(saved.keys()):
        options.append(name)
        payload = saved.get(name, {})
        budget_dict = payload.get('budgets', {}) if isinstance(payload, dict) else {}
        budget_count = len(budget_dict) if isinstance(budget_dict, dict) else 0
        label = name + (f" - {budget_count} categories" if budget_count > 0 else " - empty")
        option_labels.append(label)
    
    if not options:
        st.info("No budgets available. Create and save one in the **🛠 Manage Budgets** tab.")
        return '', {}
    
    # Determine default index
    default_index = 0
    if selected_name:
        if selected_name in options:
            default_index = options.index(selected_name)
        elif selected_name == 'Current (unsaved)' and 'Current (unsaved)' in options:
            default_index = 0
    
    # Use the actual option names for the selectbox, but show labels
    chosen = st.selectbox(
        "Budgets",
        options=options,
        index=default_index,
        key='budget_selector',
        format_func=lambda x: option_labels[options.index(x)] if x in options else x
    )
    
    if chosen == 'Current (unsaved)':
        budget = dict(current_budget)
        st.session_state.selected_budget_name = ''
    else:
        # Load from saved budgets
        payload = saved.get(chosen, {})
        if not isinstance(payload, dict):
            st.error(f"Invalid budget data for '{chosen}'")
            return '', {}
        
        budget = payload.get('budgets', {})
        if not isinstance(budget, dict):
            budget = {}
        
        # Ensure budget values are floats
        budget = {k: float(v) for k, v in budget.items() if k and v is not None}
        
        # Show status
        if not budget:
            st.info(f"📋 **{chosen}** is loaded but has no budget amounts. Go to **🛠 Manage Budgets** to add budget amounts.")
        else:
            st.success(f"✅ Loaded **{chosen}** with {len(budget)} budget category(ies)")
        
        # Load envelope configuration
        envelopes = payload.get('envelopes', [])
        if envelopes:
            st.session_state.envelope_groups_config = ensure_savings_entry(envelopes)
        
        # Update session state
        st.session_state.budgets = budget
        st.session_state.selected_budget_name = chosen
    
    return chosen, budget


# ============================================================================
# Envelope Rendering
# ============================================================================

def render_envelope_breakdown(config: List[Dict[str, str]], income_avg: float = 0.0) -> None:
    """Render envelope breakdown visualization.
    
    Args:
        config: List of envelope configuration dictionaries
        income_avg: Average monthly income for calculations
    """
    if not config:
        return
    
    st.subheader("Envelope Breakdown")
    rows = []
    for row in config:
        target_pct = float(row.get('Target %', 0.0))
        target_dollars = (target_pct / 100.0) * income_avg if income_avg > 0 else 0.0
        rows.append({
            'Envelope': row.get('Group', 'Unknown'),
            'Type': row.get('Type', 'expenses'),
            'Target %': target_pct,
            'Target $': target_dollars,
            'Categories': row.get('Categories', ''),
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================================
# Performance Rendering
# ============================================================================

def render_perf_table(perf_df: pd.DataFrame) -> None:
    """Render budget performance table.
    
    Args:
        perf_df: DataFrame from budget_performance_snapshot
    """
    if perf_df.empty:
        st.info("No budget performance data for this period.")
        return
    
    st.dataframe(
        perf_df.style.format({
            'Budget/mo': '${:,.2f}',
            'Actual/mo': '${:,.2f}',
            'Variance/mo': '${:,.2f}',
            'Target Total': '${:,.2f}',
            'Actual Total': '${:,.2f}',
            'Variance Total': '${:,.2f}',
            'Percent Used': '{:.1f}%',
        }),
        use_container_width=True,
        hide_index=True,
    )


def render_perf_chart(perf_df: pd.DataFrame, *, title: str, use_total: bool = False) -> None:
    """Render budget performance chart.
    
    Args:
        perf_df: DataFrame from budget_performance_snapshot
        title: Chart title
        use_total: If True, use total columns; otherwise use monthly columns
    """
    if perf_df.empty:
        return
    
    budget_col = 'Target Total' if use_total else 'Budget/mo'
    actual_col = 'Actual Total' if use_total else 'Actual/mo'
    
    fig = px.bar(
        perf_df,
        x='Category',
        y=[budget_col, actual_col],
        barmode='group',
        title=title,
        labels={'value': 'Amount ($)', 'variable': 'Type'}
    )
    st.plotly_chart(fig, use_container_width=True)


def render_variance_focus(perf_df: pd.DataFrame) -> None:
    """Render variance focus chart showing over/under budget.
    
    Args:
        perf_df: DataFrame from budget_performance_snapshot
    """
    if perf_df.empty:
        return
    
    fig = px.bar(
        perf_df,
        x='Category',
        y='Variance/mo',
        color='Status',
        title='Budget Variance by Category',
        labels={'Variance/mo': 'Variance ($/month)'},
        color_discrete_map={'Over': 'red', 'Under': 'green'}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Helper Functions for Session State Access
# ============================================================================

def _current_envelope_groups() -> Dict[str, List[str]]:
    """Get current envelope groups from session state."""
    default_env = _get_default_envelopes()
    config = st.session_state.get('envelope_groups_config', default_env)
    return groups_from_envelopes(config)


def _current_target_map() -> Dict[str, float]:
    """Get current target map from session state."""
    default_env = _get_default_envelopes()
    config = st.session_state.get('envelope_groups_config', default_env)
    return targets_from_envelopes(config)


# ============================================================================
# Large UI Components - Unified Budget System
# ============================================================================

def render_unified_budget_system(
    budgets: Dict[str, float],
    envelope_config: List[Dict[str, Any]],
    analytics: Optional[PersonalFinanceAnalytics],
    budget_name: str,
    saved_budgets: Dict[str, Dict[str, Any]]
) -> None:
    """Unified interface combining envelope settings and budget editor.
    
    Args:
        budgets: Current budget dictionary
        envelope_config: Current envelope configuration
        analytics: Analytics object with transaction data
        budget_name: Name for saving the budget
        saved_budgets: Dictionary of all saved budgets
    """
    constants = _get_constants()
    envelope_types = constants['envelope_types']
    expenses_type = envelope_types['expenses']
    ui_config = _get_config().get('ui', {})
    messages = ui_config.get('messages', {})
    
    st.header("📝 Budget Planning")
    
    # Get income estimate
    income_estimate = three_month_avg_income(analytics)
    
    # Income summary at top
    if income_estimate > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Monthly Income", f"${income_estimate:,.2f}", help="3-month average")
        
        # Calculate budgeted amount from expenses envelopes only (excludes General Savings)
        # This is the allocation to expense envelopes, not individual category budgets
        expenses_pct = sum(
            float(row.get('Target %', 0.0))
            for row in envelope_config
            if row.get('Type', expenses_type) == expenses_type
        )
        # Cap at 100% to ensure it never exceeds income
        expenses_pct = min(expenses_pct, 100.0)
        total_budget = (expenses_pct / 100.0) * income_estimate
        
        with col2:
            st.metric("Budgeted", f"${total_budget:,.2f}", delta=f"{expenses_pct:.1f}%", help="Total allocated to expense envelopes (excludes General Savings)")
        
        remaining = income_estimate - total_budget
        remaining_pct = (remaining / income_estimate * 100) if income_estimate > 0 else 0.0
        with col3:
            st.metric("Remaining", f"${remaining:,.2f}", delta=f"{remaining_pct:.1f}%", help="Available for savings (General Savings)")
        
        savings_pct = 100.0 - expenses_pct
        with col4:
            st.metric("Savings Target", f"{savings_pct:.1f}%", f"${(savings_pct/100)*income_estimate:,.2f}", help="Auto-calculated from envelope allocation")
    else:
        no_income_msg = messages.get('no_income_data', "⚠️ No income data available. Upload transaction data with income to enable percentage-based budgeting.")
        st.warning(no_income_msg)
    
    # Mode selector
    labels = ui_config.get('labels', {})
    view_mode = st.radio(
        "View Mode",
        options=["Envelope View", "Category View"],
        horizontal=True,
        help="Envelope View: Quick setup by groups. Category View: Detailed control per category."
    )
    
    if view_mode == "Envelope View":
        render_unified_envelope_view(budgets, envelope_config, analytics, income_estimate, budget_name, saved_budgets)
    else:
        render_detailed_budget_editor(budgets, _current_envelope_groups(), _current_target_map(), analytics, income_estimate, budget_name, saved_budgets)
        if st.button("🧹 Clear all budgets"):
            st.session_state.budgets = {}
            st.warning("Budgets cleared.")
            st.rerun()


def render_unified_envelope_view(
    budgets: Dict[str, float],
    envelope_config: List[Dict[str, Any]],
    analytics: Optional[PersonalFinanceAnalytics],
    income_estimate: float,
    budget_name: str,
    saved_budgets: Dict[str, Dict[str, Any]]
) -> None:
    """Unified view showing envelope configuration and budget setting together.
    
    Args:
        budgets: Current budget dictionary
        envelope_config: Current envelope configuration
        analytics: Analytics object with transaction data
        income_estimate: Estimated monthly income
        budget_name: Name for saving the budget
        saved_budgets: Dictionary of all saved budgets
    """
    constants = _get_constants()
    savings_group_name = constants['savings_group_name']
    envelope_types = constants['envelope_types']
    expenses_type = envelope_types['expenses']
    ui_config = _get_config().get('ui', {})
    messages = ui_config.get('messages', {})
    labels = ui_config.get('labels', {})
    
    config = ensure_savings_entry(envelope_config)
    all_categories = sorted(set(_fetch_categories() + list(st.session_state.budgets.keys())))
    groups = groups_from_envelopes(config)
    targets = targets_from_envelopes(config)
    group_budgets = aggregate_budgets_by_group(budgets, groups)
    
    # Separate expenses and savings
    expenses_envelopes = [
        row for row in config
        if row.get('Type', expenses_type) == expenses_type and row.get('Group') != savings_group_name
    ]
    savings_row = next((row for row in config if row.get('Group') == savings_group_name), None)
    
    # Check for unassigned categories
    assigned_categories = set()
    for row in config:
        cats = [c.strip() for c in row.get('Categories', '').split(',') if c.strip()]
        assigned_categories.update(cats)
    
    unassigned_categories = [c for c in all_categories if c not in assigned_categories]
    if unassigned_categories:
        unassigned_msg = messages.get('unassigned_categories', "⚠️ {count} unassigned category(ies): {cats}. Please assign all categories to envelopes below.")
        cats_display = ', '.join(unassigned_categories[:10]) + ('...' if len(unassigned_categories) > 10 else '')
        st.warning(unassigned_msg.format(count=len(unassigned_categories), cats=cats_display))
    
    st.markdown("### Configure Envelopes & Set Budgets")
    st.caption("Configure your envelope groups and set budget amounts all in one place. Changes to envelope percentages automatically update budget suggestions.")
    
    # Add envelope button
    col_add, col_info = st.columns([1, 4])
    with col_add:
        add_label = labels.get('add_envelope', '➕ Add Envelope')
        if st.button(add_label, use_container_width=True):
            non_savings = [row for row in config if row.get('Group') != savings_group_name]
            savings = next(row for row in config if row.get('Group') == savings_group_name)
            non_savings.append({
                'Group': f"Envelope {len(non_savings)+1}",
                'Type': expenses_type,
                'Target %': 0.0,
                'Categories': ''
            })
            st.session_state.envelope_groups_config = finalize_envelope_config(non_savings, savings)
            st.rerun()
    with col_info:
        if income_estimate > 0:
            # Escape dollar sign to prevent markdown/LaTeX interpretation
            income_str = f"&#36;{income_estimate:,.2f}"
            st.markdown(f"<small>💡 Set budget as percentage of income ({income_str} monthly). Dollar amounts are calculated automatically.</small>", unsafe_allow_html=True)
        else:
            st.caption("⚠️ No income data available. Upload transaction data to enable percentage-based budgeting.")
    
    updated_entries = []
    
    # Render each expenses envelope with integrated budget setting
    for idx, row in enumerate(expenses_envelopes):
        envelope_name = row.get('Group', f"Envelope {idx+1}")
        current_target_pct = float(row.get('Target %', 0.0))
        
        with st.container():
            # Envelope header with budget summary
            col_header1, col_header2, col_header3 = st.columns([3, 2, 1])
            with col_header1:
                name = st.text_input("Envelope Name", value=envelope_name, key=f"env_name_{idx}", label_visibility="collapsed")
            with col_header2:
                if income_estimate > 0:
                    help_text = ui_config.get('help_text', {}).get('target_percent', "Percentage of monthly income to budget for this envelope")
                    target_pct = st.number_input(
                        "Budget % of Income",
                        min_value=0.0,
                        max_value=100.0,
                        value=current_target_pct,
                        step=0.5,
                        key=f"env_target_{idx}",
                        help=help_text,
                        label_visibility="collapsed"
                    )
                    target_dollars = (target_pct / 100.0) * income_estimate
                else:
                    target_pct = current_target_pct
                    target_dollars = 0.0
                    st.caption("No income data")
            with col_header3:
                if st.button("🗑️", key=f"env_remove_{idx}", help="Remove envelope"):
                    continue
            
            # Show calculated dollar amount
            if income_estimate > 0 and target_pct > 0:
                # Escape dollar signs to prevent markdown/LaTeX interpretation
                dollar_str = f"&#36;{target_dollars:,.2f}"
                income_str = f"&#36;{income_estimate:,.2f}"
                st.markdown(f"<small>💵 {dollar_str} per month (calculated from {target_pct:.1f}% of {income_str} income)</small>", unsafe_allow_html=True)
            
            # Category selection
            default_categories = [c.strip() for c in row.get('Categories', '').split(',') if c.strip()]
            missing_defaults = [c for c in default_categories if c not in all_categories]
            option_list = sorted(set(all_categories + missing_defaults))
            default_categories = [c for c in default_categories if c in option_list]
            
            # Highlight unassigned categories
            unassigned_in_options = [c for c in option_list if c in unassigned_categories]
            help_text_cats = ui_config.get('help_text', {}).get('category_selection', "Select expense categories that belong to this envelope. Unassigned categories should be assigned.")
            selected_categories = st.multiselect(
                f"Categories for {name}",
                options=option_list,
                default=default_categories,
                key=f"env_cats_{idx}",
                help=help_text_cats
            )
            
            # Show indicator if this envelope has unassigned categories
            if unassigned_in_options and any(c in selected_categories for c in unassigned_in_options):
                assigned_count = len([c for c in selected_categories if c in unassigned_in_options])
                st.caption(f"✅ Assigning {assigned_count} previously unassigned category(ies)")
            
            st.divider()
            
            updated_entries.append({
                'Group': name.strip() or f"Envelope {idx+1}",
                'Type': expenses_type,
                'Target %': target_pct,
                'Categories': ', '.join(selected_categories)
            })
    
    # Show General Savings (read-only)
    if savings_row:
        savings_target = finalize_envelope_config(updated_entries, savings_row)[-1]['Target %']
        st.markdown("### 💾 General Savings")
        savings_categories = [c.strip() for c in savings_row.get('Categories', '').split(',') if c.strip()]
        col_sav1, col_sav2 = st.columns([1, 2])
        with col_sav1:
            if income_estimate > 0:
                savings_dollars = (savings_target / 100.0) * income_estimate
                st.metric("Auto-Allocated", f"{savings_target:.1f}%", f"${savings_dollars:,.2f}")
            else:
                st.metric("Auto-Allocated", "N/A", "No income data")
        with col_sav2:
            cats_display = ', '.join(savings_categories[:8]) + ('...' if len(savings_categories) > 8 else '')
            st.caption(f"Categories: {cats_display}")
            st.info(f"General Savings automatically receives the remainder ({savings_target:.1f}%) after all expense envelopes are allocated. This is not manually editable.")
    
    # Summary and save
    expenses_pct = sum(float(e.get('Target %', 0.0)) for e in updated_entries)
    final_config = finalize_envelope_config(updated_entries, savings_row)
    
    st.markdown("---")
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        if income_estimate > 0:
            expenses_dollars = (expenses_pct / 100.0) * income_estimate
            st.metric("Expenses Allocation", f"{expenses_pct:.1f}%", f"${expenses_dollars:,.2f}")
    with col_sum2:
        if income_estimate > 0 and savings_row:
            savings_pct = final_config[-1]['Target %']
            savings_dollars = (savings_pct / 100.0) * income_estimate
            st.metric("Savings Allocation", f"{savings_pct:.1f}%", f"${savings_dollars:,.2f}")
    with col_sum3:
        total_pct = expenses_pct + (final_config[-1]['Target %'] if final_config else 0.0)
        if abs(total_pct - 100.0) < 0.01:
            st.success("✅ 100% Allocated")
        elif expenses_pct > 100.01:
            st.error(f"⚠️ Over 100%")
        else:
            st.info(f"Total: {total_pct:.1f}%")
    
    # Check for unassigned categories after edits
    final_assigned = set()
    for entry in updated_entries:
        cats = [c.strip() for c in entry.get('Categories', '').split(',') if c.strip()]
        final_assigned.update(cats)
    if savings_row:
        savings_cats = [c.strip() for c in savings_row.get('Categories', '').split(',') if c.strip()]
        final_assigned.update(savings_cats)
    
    final_unassigned = [c for c in all_categories if c not in final_assigned]
    
    # Single save button - saves both envelope config and budget amounts, then saves to disk
    save_label = labels.get('save_button', '💾 Save Budget')
    if st.button(save_label, type="primary", use_container_width=True):
        if final_unassigned:
            unassigned_msg = messages.get('unassigned_categories', "⚠️ Cannot save: {count} category(ies) are not assigned to any envelope: {cats}. Please assign all categories to envelopes.")
            cats_display = ', '.join(final_unassigned[:10]) + ('...' if len(final_unassigned) > 10 else '')
            st.error(unassigned_msg.format(count=len(final_unassigned), cats=cats_display))
        elif income_estimate > 0 and expenses_pct > 100.01:
            exceeds_msg = messages.get('exceeds_100_percent', "⚠️ Expenses allocation exceeds 100% ({percent}%). Please adjust.")
            st.error(exceeds_msg.format(percent=expenses_pct))
        else:
            # Save envelope configuration
            st.session_state.envelope_groups_config = final_config
            
            # Calculate budget amounts from percentages
            envelope_budget_amounts = {}
            for entry in updated_entries:
                envelope_name = entry['Group']
                target_pct = entry['Target %']
                if target_pct > 0 and income_estimate > 0:
                    envelope_budget_amounts[envelope_name] = (target_pct / 100.0) * income_estimate
            
            # Expand envelope budgets to category budgets
            if envelope_budget_amounts:
                expanded = expand_group_budgets(envelope_budget_amounts, analytics, groups)
                st.session_state.budgets = expanded
                # Calculate total budget excluding Income and Transfer categories
                exclude_categories = {'income', 'transfer', 'transfers'}
                total_budget = sum(
                    amount for category, amount in expanded.items()
                    if category.lower() not in exclude_categories
                )
            else:
                st.session_state.budgets = {}
                total_budget = 0.0
            
            # Save to disk with the provided budget name
            config_constants = _get_constants()
            default_name = config_constants.get('default_budget_name', 'My Budget')
            if not budget_name or not budget_name.strip():
                budget_name = default_name
            
            try:
                save_budget(budget_name.strip(), dict(st.session_state.budgets), final_config)
                st.session_state.selected_budget_name = budget_name.strip()
                if total_budget > 0:
                    total_str = escape_dollar_for_markdown(total_budget)
                    saved_msg = messages.get('budget_saved', "✅ Saved budget '{name}'! {count} categories budgeted ({total} total, {percent}% of income)")
                    st.success(saved_msg.format(
                        name=budget_name.strip(),
                        count=len(st.session_state.budgets),
                        total=total_str,
                        percent=expenses_pct
                    ))
                else:
                    st.success(f"✅ Saved budget '{budget_name.strip()}' (envelope configuration saved)")
                st.rerun()
            except (ValueError, OSError) as e:
                st.error(f"Failed to save budget: {e}")


def render_envelope_budget_editor(
    groups: Dict[str, List[str]],
    targets: Dict[str, float],
    budgets: Dict[str, float],
    analytics: Optional[PersonalFinanceAnalytics],
    income_estimate: float,
    budget_name: str,
    saved_budgets: Dict[str, Dict[str, Any]]
) -> None:
    """Render envelope-based budget editor using percentage of income only.
    
    Args:
        groups: Dictionary mapping group names to category lists
        targets: Dictionary mapping group names to target percentages (as decimals)
        budgets: Current budget dictionary
        analytics: Analytics object with transaction data
        income_estimate: Estimated monthly income
        budget_name: Name for saving the budget
        saved_budgets: Dictionary of all saved budgets
    """
    constants = _get_constants()
    envelope_types = constants['envelope_types']
    expenses_type = envelope_types['expenses']
    savings_type = envelope_types['savings']
    ui_config = _get_config().get('ui', {})
    messages = ui_config.get('messages', {})
    labels = ui_config.get('labels', {})
    
    st.markdown("#### Quick Setup: Envelope Groups")
    st.caption("Set budgets by envelope group as percentage of income. Changes will be distributed to categories within each group.")
    
    if income_estimate <= 0:
        no_income_msg = messages.get('no_income_data', "⚠️ No income data available. Upload transaction data with income to enable percentage-based budgeting.")
        st.warning(no_income_msg)
        return
    
    income_str = escape_dollar_for_markdown(income_estimate)
    st.info(f"💡 Set budgets as **percentage of income** ({income_str} monthly). Dollar amounts are calculated automatically.")
    
    # Get envelope config to determine which are savings
    default_env = _get_default_envelopes()
    envelope_config = st.session_state.get('envelope_groups_config', default_env)
    envelope_types_map = {row.get('Group'): row.get('Type', expenses_type) for row in envelope_config}
    
    group_budgets = aggregate_budgets_by_group(budgets, groups)
    
    # Separate expenses and savings groups
    expenses_groups = [g for g in groups.keys() if envelope_types_map.get(g, expenses_type) == expenses_type]
    savings_groups = [g for g in groups.keys() if envelope_types_map.get(g) == savings_type]
    
    # Create editor dataframe - only show expenses groups for editing
    group_df = pd.DataFrame({'Group': expenses_groups})
    group_df['Recommended %'] = group_df['Group'].map(lambda g: targets.get(g, 0.0) * 100)
    # Calculate current percentage from existing budgets
    group_df['Budget (%)'] = group_df.apply(
        lambda row: (group_budgets.get(row['Group'], 0.0) / income_estimate * 100) if income_estimate > 0 and group_budgets.get(row['Group'], 0.0) > 0 else 0.0,
        axis=1
    )
    group_df['Budget ($)'] = group_df.apply(
        lambda row: (row['Budget (%)'] / 100.0) * income_estimate if row['Budget (%)'] > 0 else 0.0,
        axis=1
    )
    group_df['Suggested (%)'] = group_df['Group'].map(lambda g: targets.get(g, 0.0) * 100)
    
    # Show savings groups separately (read-only)
    if savings_groups:
        st.info(f"💾 **Savings Envelopes** ({', '.join(savings_groups)}) are calculated automatically as the remainder after expenses. They are not included in the allocation below.")
    
    edited = st.data_editor(
        group_df,
        use_container_width=True,
        hide_index=True,
        key='budget_editor_groups_percent',
        column_config={
            'Budget (%)': st.column_config.NumberColumn(
                "Budget (%)",
                format="%0.2f%%",
                help="Enter percentage of income for this envelope"
            ),
            'Budget ($)': st.column_config.NumberColumn("Budget ($)", format="$%0.2f", disabled=True, help="Calculated from percentage"),
            'Suggested (%)': st.column_config.NumberColumn("Suggested (%)", format="%0.2f%%", disabled=True),
            'Recommended %': st.column_config.NumberColumn("Recommended %", format="%0.2f%%", disabled=True),
        }
    )
    
    # Process edits - use percentage only
    save_label = labels.get('save_button', '💾 Save Budget')
    if st.button(save_label, type="primary", key='save_group_budgets', use_container_width=True):
        group_amounts = {}
        total_pct = 0.0
        
        for _, row in edited.iterrows():
            group = row['Group']
            budget_pct = float(row['Budget (%)']) if pd.notna(row['Budget (%)']) else 0.0
            
            if budget_pct > 0 and income_estimate > 0:
                # Convert percentage to dollars
                budget_dollars = (budget_pct / 100.0) * income_estimate
                group_amounts[group] = budget_dollars
                total_pct += budget_pct
        
        # Validate total percentage (only expenses count)
        if total_pct > 100.01:
            exceeds_msg = messages.get('exceeds_100_percent', "⚠️ Expenses budget exceeds 100% of income ({percent}%). Please adjust. Savings will be calculated as the remainder.")
            st.error(exceeds_msg.format(percent=total_pct))
        else:
            # Expand group budgets to category budgets
            expanded = expand_group_budgets(group_amounts, analytics, groups)
            st.session_state.budgets = expanded
            
            # Save to disk with the provided budget name
            config_constants = _get_constants()
            default_name = config_constants.get('default_budget_name', 'My Budget')
            if not budget_name or not budget_name.strip():
                budget_name = default_name
            
            default_env = _get_default_envelopes()
            envelope_config = st.session_state.get('envelope_groups_config', default_env)
            try:
                save_budget(budget_name.strip(), dict(expanded), envelope_config)
                st.session_state.selected_budget_name = budget_name.strip()
                savings_pct = max(0.0, 100.0 - total_pct)
                total_str = escape_dollar_for_markdown(sum(group_amounts.values()))
                saved_msg = messages.get('budget_saved', "✅ Saved budget '{name}'! {count} categories ({percent}% of income, {total} total). Savings: {savings}%")
                st.success(saved_msg.format(
                    name=budget_name.strip(),
                    count=len(expanded),
                    percent=total_pct,
                    total=total_str,
                    savings=savings_pct
                ))
                st.rerun()
            except (ValueError, OSError) as e:
                st.error(f"Failed to save budget: {e}")


def render_detailed_budget_editor(
    budgets: Dict[str, float],
    groups: Dict[str, List[str]],
    targets: Dict[str, float],
    analytics: Optional[PersonalFinanceAnalytics],
    income_estimate: float,
    budget_name: str,
    saved_budgets: Dict[str, Dict[str, Any]]
) -> None:
    """Render detailed category budget editor using percentage of income only.
    
    Args:
        budgets: Current budget dictionary
        groups: Dictionary mapping group names to category lists
        targets: Dictionary mapping group names to target percentages (as decimals)
        analytics: Analytics object with transaction data
        income_estimate: Estimated monthly income
        budget_name: Name for saving the budget
        saved_budgets: Dictionary of all saved budgets
    """
    ui_config = _get_config().get('ui', {})
    messages = ui_config.get('messages', {})
    labels = ui_config.get('labels', {})
    
    st.markdown("#### Detailed Setup: Individual Categories")
    st.caption("Set budgets for each category individually as percentage of income.")
    
    if income_estimate <= 0:
        no_income_msg = messages.get('no_income_data', "⚠️ No income data available. Upload transaction data with income to enable percentage-based budgeting.")
        st.warning(no_income_msg)
        return
    
    income_str = escape_dollar_for_markdown(income_estimate)
    st.info(f"💡 Set budgets as percentage of income ({income_str} monthly). Dollar amounts are calculated automatically.")
    
    categories = sorted(set(budgets.keys()) | set(_fetch_categories()))
    if analytics is not None and not analytics.calculate_category_spending().empty:
        categories = sorted(set(categories) | set(analytics.calculate_category_spending().index.tolist()))
    
    if not categories:
        no_cats_msg = messages.get('no_categories', "No categories available. Import transaction data to see categories.")
        st.info(no_cats_msg)
        return
    
    editor_df = pd.DataFrame({'Category': categories})
    editor_df['Group'] = editor_df['Category'].map(lambda c: infer_group(c, groups))
    # Calculate current percentage from existing budgets
    editor_df['Monthly Budget (%)'] = editor_df.apply(
        lambda row: (budgets.get(row['Category'], 0.0) / income_estimate * 100) if income_estimate > 0 and budgets.get(row['Category'], 0.0) > 0 else 0.0,
        axis=1
    )
    editor_df['Monthly Budget ($)'] = editor_df.apply(
        lambda row: (row['Monthly Budget (%)'] / 100.0) * income_estimate if row['Monthly Budget (%)'] > 0 else 0.0,
        axis=1
    )
    
    suggestions_dict = suggested_amounts(editor_df['Category'], analytics, True, groups, targets)
    editor_df['Suggested (%)'] = editor_df.apply(
        lambda row: (suggestions_dict.get(row['Category'], 0.0) / income_estimate * 100) if income_estimate > 0 and suggestions_dict.get(row['Category'], 0.0) > 0 else 0.0,
        axis=1
    )
    editor_df['Suggested ($)'] = editor_df['Category'].map(suggestions_dict).fillna(0.0)
    editor_df['Use suggestion'] = False
    
    edited = st.data_editor(
        editor_df,
        num_rows="dynamic",
        use_container_width=True,
        key='budget_editor_detailed_percent',
        column_config={
            'Monthly Budget (%)': st.column_config.NumberColumn(
                "Monthly Budget (%)",
                format="%0.2f%%",
                help="Enter percentage of income for this category"
            ),
            'Monthly Budget ($)': st.column_config.NumberColumn("Monthly Budget ($)", format="$%0.2f", disabled=True, help="Calculated from percentage"),
            'Suggested (%)': st.column_config.NumberColumn("Suggested (%)", format="%0.2f%%", disabled=True),
            'Suggested ($)': st.column_config.NumberColumn("Suggested ($)", format="$%0.2f", disabled=True),
            'Use suggestion': st.column_config.CheckboxColumn("Auto", help="Use suggested percentage"),
        },
        hide_index=True,
    )
    
    save_label = labels.get('save_button', '💾 Save Budget')
    if st.button(save_label, type="primary", key='save_detailed_budgets', use_container_width=True):
        updated = {}
        total_pct = 0.0
        
        for row in edited.to_dict('records'):
            category = row.get('Category')
            if not category:
                continue
            
            # Handle "Use suggestion" checkbox
            if row.get('Use suggestion', False):
                budget_pct = float(row.get('Suggested (%)', 0.0)) if pd.notna(row.get('Suggested (%)')) else 0.0
            else:
                budget_pct = float(row.get('Monthly Budget (%)', 0.0)) if pd.notna(row.get('Monthly Budget (%)')) else 0.0
            
            if budget_pct > 0 and income_estimate > 0:
                # Convert percentage to dollars
                amount = (budget_pct / 100.0) * income_estimate
                updated[category] = float(amount)
                total_pct += budget_pct
        
        # Validate total percentage
        if total_pct > 100.01:
            exceeds_msg = messages.get('total_exceeds_100_percent', "⚠️ Total budget exceeds 100% of income ({percent}%). Please adjust.")
            st.error(exceeds_msg.format(percent=total_pct))
        else:
            st.session_state.budgets = updated
            
            # Save to disk with the provided budget name
            config_constants = _get_constants()
            default_name = config_constants.get('default_budget_name', 'My Budget')
            if not budget_name or not budget_name.strip():
                budget_name = default_name
            
            default_env = _get_default_envelopes()
            envelope_config = st.session_state.get('envelope_groups_config', default_env)
            try:
                save_budget(budget_name.strip(), dict(updated), envelope_config)
                st.session_state.selected_budget_name = budget_name.strip()
                total_dollars = sum(updated.values())
                total_str = escape_dollar_for_markdown(total_dollars)
                saved_msg = messages.get('budget_saved', "✅ Saved budget '{name}'! {count} categories ({percent}% of income, {total} total)")
                st.success(saved_msg.format(
                    name=budget_name.strip(),
                    count=len(updated),
                    percent=total_pct,
                    total=total_str
                ))
                st.rerun()
            except (ValueError, OSError) as e:
                st.error(f"Failed to save budget: {e}")


def render_budget_insights(
    budgets: Dict[str, float],
    analytics: Optional[PersonalFinanceAnalytics],
    budget_name: str,
    allow_empty: bool = False
) -> None:
    """Render comprehensive budget insights and performance analysis.
    
    Modern, interactive budget insights with professional UI/UX inspired by
    leading financial tools. Organized by user priority: quick overview first,
    then detailed analysis.
    
    Args:
        budgets: Dictionary mapping category names to budget amounts
        analytics: Analytics object with transaction data
        budget_name: Name of the current budget
        allow_empty: If True, allow rendering even when budgets are empty
    """
    if analytics is None or analytics.data.empty:
        st.info("📊 Load transactions to analyze budgets.")
        return
    
    if not budgets:
        st.warning("⚠️ **No budget amounts set**")
        if budget_name:
            st.info(f"""
            **Current budget: "{budget_name}"** (saved but empty)
            
            To see insights:
            1. Go to the **🛠 Manage Budgets** tab
            2. Use the budget editor to set monthly budget amounts for categories
            3. Click **💾 Save budgets** to save your changes
            4. Return here to see insights
            """)
        else:
            st.info("""
            **To get started:**
            1. Go to the **🛠 Manage Budgets** tab
            2. Use the budget editor to set monthly budget amounts for categories
            3. Click **💾 Save budgets** to save your budget
            4. Return here to see insights
            """)
        return
    
    # Exclude Income and Transfer categories from budget calculations
    exclude_categories = {'income', 'transfer', 'transfers'}
    expense_budgets = {
        k: v for k, v in budgets.items()
        if k.lower() not in exclude_categories
    }
    
    if not expense_budgets:
        st.warning("⚠️ No expense budgets set (only Income/Transfer categories found).")
        return
    
    # Prepare data
    data = analytics.data.copy()
    data['Month'] = data['Transaction Date'].dt.to_period('M')
    if data.empty or 'Month' not in data:
        st.info("No data available in the selected window.")
        return
    
    # Get configuration
    current_month = data['Month'].max()
    income_estimate = three_month_avg_income(analytics)
    default_env = _get_default_envelopes()
    envelopes = st.session_state.get('envelope_groups_config', default_env)
    groups_map = groups_from_envelopes(envelopes)
    targets_map = targets_from_envelopes(envelopes)
    
    # Calculate current month performance
    current_expense = analytics._expense_rows(data[data['Month'] == current_month])
    current_expense['AbsAmount'] = current_expense['Amount'].abs()
    spend_by_group: Dict[str, float] = {}
    for _, row in current_expense.iterrows():
        category = row['Category']
        amount = float(row['AbsAmount'])
        group = (
            next((g for g, cats in groups_map.items() if category in cats), next(iter(groups_map.keys()), 'Other'))
            if groups_map
            else 'Other'
        )
        spend_by_group[group] = spend_by_group.get(group, 0.0) + amount
    
    # Get performance data
    perf_current, _ = budget_performance_snapshot(analytics, expense_budgets, current_only=True)
    perf_all, months_count = budget_performance_snapshot(analytics, expense_budgets, current_only=False)
    # perf_all is used in expandable section below
    
    # Calculate envelope metrics (used throughout)
    constants = _get_constants()
    envelope_types = constants['envelope_types']
    expenses_type = envelope_types['expenses']
    
    envelope_data = []
    if targets_map:
        for group, target_pct in targets_map.items():
            # Only count expenses envelopes (exclude General Savings)
            envelope_type = next(
                (row.get('Type', expenses_type) for row in envelopes if row.get('Group') == group),
                expenses_type
            )
            if envelope_type == expenses_type:
                target_dollars = income_estimate * target_pct if income_estimate > 0 else 0.0
                actual = spend_by_group.get(group, 0.0)
                pct_used = (actual / target_dollars * 100) if target_dollars > 0 else 0.0
                variance = actual - target_dollars
                
                envelope_data.append({
                    'Envelope': group,
                    'Target %': target_pct * 100,
                    'Target $': target_dollars,
                    'Actual $': actual,
                    'Variance': variance,
                    '% Used': pct_used,
                    'Status': 'Over' if variance > 0 else 'Under'
                })
    
    env_df = pd.DataFrame(envelope_data) if envelope_data else pd.DataFrame()
    
    # ============================================================================
    # SECTION 1: QUICK OVERVIEW - Key Metrics (Envelope-Based)
    # ============================================================================
    st.markdown("### 📊 Quick Overview")
    
    # Calculate monthly budget from envelope targets × income (NOT sum of categories)
    expenses_pct = sum(
        float(row.get('Target %', 0.0))
        for row in envelopes
        if row.get('Type', expenses_type) == expenses_type
    ) if envelopes else 0.0
    expenses_pct = min(expenses_pct, 100.0)  # Cap at 100%
    total_budget = (expenses_pct / 100.0) * income_estimate if income_estimate > 0 else 0.0
    
    current_spend_total = sum(spend_by_group.values())
    savings_amt = income_estimate - current_spend_total if income_estimate else 0.0
    savings_rate = (savings_amt / income_estimate * 100) if income_estimate > 0 else 0.0
    budget_utilization = (current_spend_total / total_budget * 100) if total_budget > 0 else 0.0
    
    # Count over-budget envelopes (for potential future use)
    # over_budget_envelopes = int((env_df['Status'] == 'Over').sum()) if not env_df.empty else 0
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Monthly Income",
            f"${income_estimate:,.0f}",
            help="3-month average income"
        )
    with col2:
        st.metric(
            "Monthly Budget",
            f"${total_budget:,.0f}",
            delta=f"{expenses_pct:.1f}% of income",
            help="Total allocated to expense envelopes"
        )
    with col3:
        st.metric(
            "Current Month Spend",
            f"${current_spend_total:,.0f}",
            delta=f"{budget_utilization:.1f}% of budget",
            delta_color="inverse" if budget_utilization > 100 else "normal",
            help="Total spending this month"
        )
    with col4:
        st.metric(
            "Savings Rate",
            f"{savings_rate:.1f}%",
            delta=f"${savings_amt:,.0f}",
            help="Percentage of income saved this month"
        )
    
    st.divider()
    
    # ============================================================================
    # SECTION 2: ENVELOPE PERFORMANCE - PRIMARY FOCUS
    # ============================================================================
    if not env_df.empty:
        st.markdown("### 💼 Envelope Performance")
        st.caption(f"Current month ({current_month}) - Target vs Actual spending by envelope")
        
        # Sort by % used (most critical first)
        env_df_display = env_df.sort_values('% Used', ascending=False).copy()
        
        # Create detailed envelope cards with metrics
        for _, row in env_df_display.iterrows():
            with st.container():
                # Create a card-like container
                envelope_name = row['Envelope']
                target_pct = row['Target %']
                target_dollars = row['Target $']
                actual_dollars = row['Actual $']
                variance = row['Variance']
                pct_used = row['% Used']
                status = row['Status']
                
                # Determine status icon
                if status == 'Over' or pct_used > 100:
                    status_icon = "⚠️"
                elif pct_used > 80:
                    status_icon = "⚡"
                else:
                    status_icon = "✅"
                
                # Create envelope card
                card_col1, card_col2 = st.columns([2, 1])
                
                with card_col1:
                    st.markdown(f"#### {status_icon} {envelope_name}")
                    
                    # Progress bar with custom styling
                    progress_value = min(pct_used / 100.0, 1.0)
                    st.progress(
                        progress_value,
                        text=f"${actual_dollars:,.0f} of ${target_dollars:,.0f} ({pct_used:.1f}%)"
                    )
                    
                    # Metrics row
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Target", f"${target_dollars:,.0f}", f"{target_pct:.1f}% of income")
                    with metric_col2:
                        st.metric("Actual", f"${actual_dollars:,.0f}")
                    with metric_col3:
                        delta_color = "inverse" if variance > 0 else "normal"
                        st.metric("Variance", f"${abs(variance):,.0f}", 
                                delta=f"{'Over' if variance > 0 else 'Under'}",
                                delta_color=delta_color)
                
                with card_col2:
                    # Visual indicator
                    remaining = target_dollars - actual_dollars if variance <= 0 else 0
                    remaining_pct = (remaining / target_dollars * 100) if target_dollars > 0 else 0
                    
                    if variance > 0:
                        st.error(f"**Over by ${abs(variance):,.0f}**")
                        st.caption(f"{pct_used:.1f}% of target used")
                    else:
                        st.success(f"**${remaining:,.0f} remaining**")
                        st.caption(f"{remaining_pct:.1f}% of target left")
                
                st.divider()
        
        # Envelope comparison chart
        if len(env_df_display) > 0:
            fig_envelopes = go.Figure()
            fig_envelopes.add_trace(go.Bar(
                name='Target',
                x=env_df_display['Envelope'],
                y=env_df_display['Target $'],
                marker_color='#3b82f6',
                text=env_df_display['Target $'].apply(lambda x: f'${x:,.0f}'),
                textposition='outside'
            ))
            fig_envelopes.add_trace(go.Bar(
                name='Actual',
                x=env_df_display['Envelope'],
                y=env_df_display['Actual $'],
                marker_color=['#ef4444' if s == 'Over' else '#10b981' for s in env_df_display['Status']],
                text=env_df_display['Actual $'].apply(lambda x: f'${x:,.0f}'),
                textposition='outside'
            ))
            fig_envelopes.update_layout(
                title='Envelope Performance: Target vs Actual (Current Month)',
                barmode='group',
                xaxis_tickangle=-45,
                height=450,
                showlegend=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig_envelopes, use_container_width=True)
        
        st.divider()
        
        # ============================================================================
        # SECTION 3: ENVELOPE ALERTS - Actionable Items
        # ============================================================================
        over_envelopes = env_df_display[env_df_display['Status'] == 'Over']
        if not over_envelopes.empty:
            st.markdown("### ⚠️ Envelope Alerts")
            st.warning(f"{len(over_envelopes)} envelope(s) exceeding their targets this month")
            
            for _, row in over_envelopes.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.markdown(f"**{row['Envelope']}**")
                    with col2:
                        st.markdown(f"Target: ${row['Target $']:,.0f} → Actual: ${row['Actual $']:,.0f}")
                    with col3:
                        st.error(f"${abs(row['Variance']):,.0f} over ({row['% Used']:.1f}%)")
            
            st.divider()
    
    # ============================================================================
    # SECTION 4: ENVELOPE TRENDS - Historical Performance
    # ============================================================================
    if targets_map:
        st.markdown("### 📈 Envelope Trends")
        
        monthly_groups = monthly_group_targets_actuals(analytics, targets_map, groups_map)
        if not monthly_groups.empty:
            # Filter to expenses envelopes only
            expenses_envelopes = [
                row.get('Group') for row in envelopes
                if row.get('Type', expenses_type) == expenses_type
            ]
            monthly_expenses = monthly_groups[monthly_groups['Group'].isin(expenses_envelopes)]
            
            if not monthly_expenses.empty:
                # Get available months
                available_months = sorted(monthly_expenses['Month'].unique())
                
                # Time period selector
                col1, col2 = st.columns([1, 3])
                with col1:
                    period_options = {
                        "Last 3 months": 3,
                        "Last 6 months": 6,
                        "Last 12 months": 12,
                        "All available": len(available_months)
                    }
                    selected_period = st.selectbox(
                        "Time Period",
                        options=list(period_options.keys()),
                        index=1,  # Default to 6 months
                        key="envelope_trends_period"
                    )
                    months_to_show = period_options[selected_period]
                
                # Select months based on period
                if months_to_show >= len(available_months):
                    recent_months = available_months
                    period_label = f"All {len(available_months)} months"
                else:
                    recent_months = available_months[-months_to_show:]
                    period_label = f"Last {months_to_show} months"
                
                recent = monthly_expenses[monthly_expenses['Month'].isin(recent_months)]
                
                st.caption(f"Historical performance: {period_label}")
                
                # Create a comprehensive chart showing all envelopes
                fig_trends = go.Figure()
                for group in recent['Group'].unique():
                    sub = recent[recent['Group'] == group].sort_values('Month')
                    fig_trends.add_trace(go.Scatter(
                        name=f'{group} Actual',
                        x=sub['Month'],
                        y=sub['Actual ($)'],
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(size=8)
                    ))
                    fig_trends.add_trace(go.Scatter(
                        name=f'{group} Target',
                        x=sub['Month'],
                        y=sub['Target ($)'],
                        mode='lines',
                        line=dict(dash='dash', width=1),
                        opacity=0.6,
                        showlegend=False
                    ))
                
                fig_trends.update_layout(
                    title=f'Envelope Performance Trends ({period_label})',
                    xaxis_title='Month',
                    yaxis_title='Amount ($)',
                    height=450,
                    hovermode='x unified',
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_trends, use_container_width=True)
                
                # Summary table
                summary_data = []
                for group in recent['Group'].unique():
                    sub = recent[recent['Group'] == group]
                    avg_actual = sub['Actual ($)'].mean()
                    avg_target = sub['Target ($)'].mean()
                    avg_variance = avg_actual - avg_target
                    
                    summary_data.append({
                        'Envelope': group,
                        'Avg Target': avg_target,
                        'Avg Actual': avg_actual,
                        'Avg Variance': avg_variance,
                        'Avg % Used': (avg_actual / avg_target * 100) if avg_target > 0 else 0
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df = summary_df.sort_values('Avg % Used', ascending=False)
                
                def _color_summary(row):
                    pct = row.get('Avg % Used', 0)
                    if pct > 100:
                        return ['background-color: #fca5a5; color: #7f1d1d; font-weight: 600'] * len(row)
                    elif pct > 80:
                        return ['background-color: #fbbf24; color: #78350f; font-weight: 600'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    summary_df.style.apply(_color_summary, axis=1).format({
                        'Avg Target': '${:,.0f}',
                        'Avg Actual': '${:,.0f}',
                        'Avg Variance': '${:,.0f}',
                        'Avg % Used': '{:.1f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Month-to-Month Variance Heatmap
                st.markdown("#### 📊 Monthly Variance: Over/Under Budget")
                st.caption("Visualization showing how much each envelope was over (red) or under (green) budget each month")
                
                # Prepare data for heatmap
                variance_data = recent.copy()
                variance_data['Variance ($)'] = variance_data['Target ($)'] - variance_data['Actual ($)']
                # Positive variance = under budget (good), Negative variance = over budget (bad)
                
                # Create pivot table for heatmap
                variance_pivot = variance_data.pivot_table(
                    index='Group',
                    columns='Month',
                    values='Variance ($)',
                    aggfunc='first'
                ).fillna(0)
                
                # Prepare data
                groups = variance_pivot.index.tolist()
                months = [str(m) for m in variance_pivot.columns]
                variance_values = variance_pivot.values
                
                # Create custom colorscale: green for positive (under budget), red for negative (over budget)
                # Normalize values for better color distribution
                max_abs = np.abs(variance_values).max() if variance_values.size > 0 else 1
                if max_abs == 0:
                    max_abs = 1
                
                # Create text annotations with dollar amounts
                text_annotations = []
                for i, group in enumerate(groups):
                    row = []
                    for j, month in enumerate(months):
                        value = variance_values[i, j]
                        if abs(value) < 0.01:
                            row.append("$0")
                        else:
                            sign = "+" if value >= 0 else ""
                            row.append(f"{sign}${value:,.0f}")
                    text_annotations.append(row)
                
                # Create heatmap
                fig_variance = go.Figure(data=go.Heatmap(
                    z=variance_values,
                    x=months,
                    y=groups,
                    colorscale=[
                        [0, '#dc2626'],      # Red for over budget (negative values)
                        [0.5, '#fef3c7'],   # Yellow for at budget (zero)
                        [1, '#10b981']      # Green for under budget (positive values)
                    ],
                    zmid=0,  # Center the colorscale at zero
                    text=text_annotations,
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(
                        title=dict(text="Variance ($)", side="right"),
                        tickformat="$,.0f"
                    ),
                    hovertemplate='<b>%{y}</b><br>%{x}<br>Variance: $%{z:,.2f}<extra></extra>'
                ))
                
                fig_variance.update_layout(
                    title='Monthly Envelope Variance: Target vs Actual Spending',
                    xaxis_title='Month',
                    yaxis_title='Envelope',
                    height=max(400, len(groups) * 50),
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig_variance, use_container_width=True)
                
                # Add explanation
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("🟢 **Green**: Under budget (spent less than target)")
                with col2:
                    st.markdown("🟡 **Yellow**: At budget (spent close to target)")
                with col3:
                    st.markdown("🔴 **Red**: Over budget (spent more than target)")
                
            else:
                st.info("No envelope trend data available.")
        else:
            st.info("No historical envelope data available.")
        
        st.divider()
    
    # ============================================================================
    # SECTION 5: SAVINGS & INCOME TRENDS
    # ============================================================================
    st.markdown("### 💰 Savings & Income Overview")
    
    monthly = monthly_income_vs_spend(analytics)
    if not monthly.empty:
        if 'Saved % of Income' not in monthly.columns:
            monthly['Saved % of Income'] = (monthly['Savings'] / monthly['Income'] * 100).fillna(0.0)
        
        # Compact savings rate chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig_savings = go.Figure()
            fig_savings.add_trace(go.Scatter(
                x=monthly['Month'],
                y=monthly['Saved % of Income'],
                mode='lines+markers',
                name='Savings Rate',
                line=dict(color='#10b981', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)'
            ))
            fig_savings.add_hline(
                y=20,
                line_dash="dash",
                line_color="gray",
                annotation_text="20% target",
                annotation_position="right"
            )
            fig_savings.update_layout(
                title='Savings Rate Over Time',
                xaxis_title='Month',
                yaxis_title='Savings % of Income',
                height=300,
                hovermode='x unified',
                showlegend=False
            )
            st.plotly_chart(fig_savings, use_container_width=True)
        
        with col2:
            # Current month summary
            latest = monthly.iloc[-1] if not monthly.empty else None
            if latest is not None:
                st.metric("Latest Month Income", f"${latest['Income']:,.0f}")
                st.metric("Latest Month Spending", f"${latest['Spending']:,.0f}")
                st.metric("Latest Month Savings", f"${latest['Savings']:,.0f}", 
                         delta=f"{latest['Saved % of Income']:.1f}%")
        
        st.divider()
    
    # ============================================================================
    # SECTION 6: DETAILED ANALYSIS - Expandable Sections (Categories & Details)
    # ============================================================================
    with st.expander("📋 Individual Category Performance", expanded=False):
        st.caption("Detailed breakdown by individual category (less commonly used)")
        
        if not perf_current.empty:
            display_df = perf_current.copy()
            display_df['% Used'] = (display_df['Actual/mo'] / display_df['Budget/mo'] * 100).fillna(0)
            display_df = display_df.sort_values('% Used', ascending=False)
            
            display_df = display_df[[
                'Category', 'Budget/mo', 'Actual/mo', 'Variance/mo', '% Used', 'Status'
            ]].copy()
            display_df.columns = ['Category', 'Budget', 'Actual', 'Variance', '% Used', 'Status']
            
            def _color_code_performance(row):
                pct_used = row.get('% Used', 0)
                status = row.get('Status', '')
                if status == 'Over' or pct_used > 100:
                    return ['background-color: #fca5a5; color: #7f1d1d; font-weight: 600'] * len(row)
                elif pct_used > 80:
                    return ['background-color: #fbbf24; color: #78350f; font-weight: 600'] * len(row)
                else:
                    return [''] * len(row)
            
            st.dataframe(
                display_df.style.apply(_color_code_performance, axis=1).format({
                    'Budget': '${:,.0f}',
                    'Actual': '${:,.0f}',
                    'Variance': '${:,.0f}',
                    '% Used': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Category chart (compact)
            if len(display_df) > 0:
                top_categories = display_df.head(8)  # Show top 8
                fig_cats = go.Figure()
                fig_cats.add_trace(go.Bar(
                    name='Budget',
                    x=top_categories['Category'],
                    y=top_categories['Budget'],
                    marker_color='#3b82f6'
                ))
                fig_cats.add_trace(go.Bar(
                    name='Actual',
                    x=top_categories['Category'],
                    y=top_categories['Actual'],
                    marker_color='#ef4444'
                ))
                fig_cats.update_layout(
                    title='Top Categories: Budget vs Actual',
                    barmode='group',
                    xaxis_tickangle=-45,
                    height=350,
                    showlegend=True
                )
                st.plotly_chart(fig_cats, use_container_width=True)
        else:
            st.info("No category performance data available.")
    
    with st.expander("📊 All-Time Category Summary", expanded=False):
        if not perf_all.empty:
            st.dataframe(
                perf_all.style.format({
                    'Budget/mo': '${:,.0f}',
                    'Actual/mo': '${:,.0f}',
                    'Variance/mo': '${:,.0f}',
                    'Target Total': '${:,.0f}',
                    'Actual Total': '${:,.0f}',
                    'Variance Total': '${:,.0f}',
                    'Percent Used': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No performance data available.")
    
    with st.expander("📉 Category Share of Income", expanded=False):
        income_series = income_series_complete(analytics)
        percent_df = category_percent_income(analytics, income_series)
        if not percent_df.empty:
            avg_pct = percent_df.groupby('Category')['Percent'].mean().sort_values(ascending=False).head(10)
            fig_pct = go.Figure()
            fig_pct.add_trace(go.Bar(
                x=avg_pct.values,
                y=avg_pct.index,
                orientation='h',
                marker_color='#8b5cf6',
                text=[f'{v:.1f}%' for v in avg_pct.values],
                textposition='outside'
            ))
            fig_pct.update_layout(
                title='Top Categories: Average % of Income',
                xaxis_title='% of Income',
                height=400
            )
            st.plotly_chart(fig_pct, use_container_width=True)
        else:
            st.info("Need income data to compute category percentages.")

