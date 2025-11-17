"""Category Rules Command Center."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

PARENT = Path(__file__).parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from shared_sidebar import render_shared_sidebar
from category_rules import CategoryRulesManager, apply_category_rules_to_transactions
from profile_manager import get_registry
import db


def main() -> None:
    st.set_page_config(page_title="Category Rules", page_icon="🔧", layout="wide")
    sidebar = render_shared_sidebar()
    manager = CategoryRulesManager()
    uncategorized = db.fetch_uncategorized_transactions()

    _render_summary_cards(manager, uncategorized)

    manage_tab, insights_tab, apply_tab = st.tabs([
        "🛠 Manage Rules",
        "📈 Insights",
        "🔄 Sync & Apply",
    ])

    with manage_tab:
        _render_manage_rules(manager, uncategorized)

    with insights_tab:
        _render_insights(uncategorized)

    with apply_tab:
        _render_apply_rules(manager, uncategorized)


def _render_summary_cards(manager: CategoryRulesManager, uncategorized: pd.DataFrame) -> None:
    total_rules = len(manager.get_rules())
    enabled_rules = sum(1 for r in manager.get_rules() if r.enabled)
    rule_profiles = sum(len(r.profiles) or 1 for r in manager.get_rules())
    uncategorized_count = len(uncategorized)

    cols = st.columns(4)
    cols[0].metric("Total rules", f"{total_rules}")
    cols[1].metric("Enabled", f"{enabled_rules}")
    cols[2].metric("Profile targets", f"{rule_profiles}")
    cols[3].metric("Uncategorized", f"{uncategorized_count}")


def _render_manage_rules(manager: CategoryRulesManager, uncategorized: pd.DataFrame) -> None:
    st.subheader("Create & maintain rules")
    col_form, col_table = st.columns([1.1, 1.9], gap="large")

    with col_form:
        _render_add_rule_form(manager, uncategorized)

    with col_table:
        _render_rules_grid(manager)


def _render_add_rule_form(manager: CategoryRulesManager, uncategorized: pd.DataFrame) -> None:
    registry = get_registry()
    profile_options = list(registry._profiles.keys())
    categories = sorted(db.fetch_distinct_categories())
    category_options = ["Select"] + categories + ["➕ New category"]

    with st.form("add_rule_form", clear_on_submit=True):
        keyword = st.text_input("Keyword", placeholder="e.g. ENCLAVE, NETFLIX", help="Text to match inside the transaction description.")
        category_choice = st.selectbox("Category", options=category_options)
        new_category = None
        if category_choice == "➕ New category":
            new_category = st.text_input("New category name")
        case_sensitive = st.checkbox("Case sensitive", value=False)
        whole_word = st.checkbox("Whole word", value=False)
        target_profiles = st.multiselect(
            "Profiles (optional)",
            options=profile_options,
            help="Leave empty to apply to all profiles."
        )
        
        # Date range inputs
        st.markdown("**Date Range (optional):**")
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input(
                "Start Date",
                value=None,
                help="Rule applies to transactions on or after this date. Leave empty for all-time."
            )
        with date_col2:
            end_date = st.date_input(
                "End Date",
                value=None,
                help="Rule applies to transactions on or before this date. Leave empty for all-time."
            )
        
        preview_placeholder = st.empty()
        if keyword:
            preview_placeholder.caption(_preview_matches(keyword, target_profiles, case_sensitive, uncategorized))
        submitted = st.form_submit_button("Add rule", use_container_width=True)

        if submitted:
            selected_category = new_category or (category_choice if category_choice not in {"Select", "➕ New category"} else "")
            if not keyword.strip():
                st.error("Provide a keyword.")
            elif not selected_category.strip():
                st.error("Choose or create a category.")
            else:
                # Convert dates to strings if provided
                start_date_str = start_date.isoformat() if start_date else None
                end_date_str = end_date.isoformat() if end_date else None
                
                manager.add_rule(
                    keyword=keyword.strip(),
                    category=selected_category.strip(),
                    profiles=target_profiles or None,
                    case_sensitive=case_sensitive,
                    whole_word=whole_word,
                    start_date=start_date_str,
                    end_date=end_date_str
                )
                _apply_rules_for_profiles(target_profiles)
                st.success(f"Rule '{keyword.strip()}' → '{selected_category.strip()}' added.")
                st.rerun()


def _preview_matches(keyword: str, profiles: List[str], case_sensitive: bool, uncategorized: pd.DataFrame) -> str:
    if uncategorized.empty:
        return "All transactions categorized."
    mask = uncategorized['Description'].fillna('').str.contains(keyword, case=case_sensitive, regex=False, na=False)
    if profiles:
        mask &= uncategorized['profile_name'].isin(profiles)
    matches = mask.sum()
    total = len(uncategorized)
    return f"Would match **{matches}** of {total} uncategorized transactions."


def _render_rules_grid(manager: CategoryRulesManager) -> None:
    rules = manager.get_rules()
    if not rules:
        st.info("No rules yet. Use the form to add one.")
        return

    data = []
    for rule in rules:
        # Format date range
        date_range = ""
        if rule.start_date or rule.end_date:
            start = rule.start_date or "all-time"
            end = rule.end_date or "all-time"
            date_range = f"{start} to {end}"
        else:
            date_range = "All-time"
        
        data.append({
            'Keyword': rule.keyword,
            'Category': rule.category,
            'Profiles': ", ".join(rule.profiles) if rule.profiles else 'All',
            'Case-Sensitive': rule.case_sensitive,
            'Whole-Word': rule.whole_word,
            'Date Range': date_range,
            'Enabled': rule.enabled,
        })
    df = pd.DataFrame(data)
    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Case-Sensitive': st.column_config.CheckboxColumn("Case"),
            'Whole-Word': st.column_config.CheckboxColumn("Whole"),
            'Enabled': st.column_config.CheckboxColumn("Enabled"),
        },
        key='rules_editor'
    )
    if not edited.equals(df):
        st.warning("Detected edits to the grid. Click 'Save changes' to persist.")
        if st.button("Save changes", type="primary"):
            _persist_grid_changes(manager, df, edited)
            st.success("Rules updated.")
            st.rerun()


def _persist_grid_changes(manager: CategoryRulesManager, original: pd.DataFrame, edited: pd.DataFrame) -> None:
    for orig, new in zip(original.to_dict('records'), edited.to_dict('records')):
        if orig == new:
            continue
        profiles = None if new['Profiles'] == 'All' else [p.strip() for p in new['Profiles'].split(',') if p.strip()]
        updates = {
            'keyword': new['Keyword'],
            'category': new['Category'],
            'profiles': profiles,
            'case_sensitive': new['Case-Sensitive'],
            'whole_word': new['Whole-Word'],
            'enabled': new['Enabled'],
        }
        manager.update_rule(
            keyword=orig['Keyword'],
            category=orig['Category'],
            **updates,
        )


def _render_insights(uncategorized: pd.DataFrame) -> None:
    st.subheader("Gaps & opportunities")
    if uncategorized.empty:
        st.success("No uncategorized transactions detected.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Top words in uncategorized descriptions")
        words = pd.DataFrame(_common_words(uncategorized, limit=20), columns=['Word', 'Count'])
        if not words.empty:
            st.dataframe(words, use_container_width=True, hide_index=True)
        else:
            st.info("Need more data to surface insights.")
    with col2:
        st.caption("Uncategorized by amount")
        chart_df = uncategorized.copy()
        chart_df['AbsAmount'] = chart_df['Amount'].abs()
        top = chart_df.nlargest(30, 'AbsAmount')
        fig = px.bar(top, x='Description', y='AbsAmount', color='profile_name', title='Largest uncategorized transactions')
        fig.update_layout(xaxis_title="Description", yaxis_title="Amount ($)", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)


def _common_words(uncategorized: pd.DataFrame, limit: int = 20) -> List[tuple[str, int]]:
    if uncategorized.empty or 'Description' not in uncategorized.columns:
        return []
    stop_words = {
        'the', 'and', 'for', 'with', 'this', 'that', 'from', 'your', 'you', 'payment',
        'purchase', 'card', 'debit', 'credit', 'withdrawal', 'online', 'bank', 'checking',
        'misc', 'transaction', 'fee', 'inc'
    }
    pattern = re.compile(r"[A-Za-z']+")
    counter: Counter[str] = Counter()
    for text in uncategorized['Description'].dropna().astype(str):
        for word in pattern.findall(text.lower()):
            if len(word) < 3 or word in stop_words:
                continue
            counter[word] += 1
    return counter.most_common(limit)


def _render_apply_rules(manager: CategoryRulesManager, uncategorized: pd.DataFrame) -> None:
    st.subheader("Apply rules to the database")
    registry = get_registry()
    profiles = ["All Profiles"] + list(registry._profiles.keys())
    selection = st.selectbox("Target profile", options=profiles)
    profile_name = None if selection == "All Profiles" else selection
    
    # Option to include already-categorized transactions
    include_categorized = st.checkbox(
        "Apply to all transactions (including already-categorized)",
        value=True,
        help="If checked, rules will update transactions even if they already have a category. If unchecked, only uncategorized transactions will be updated."
    )

    cols = st.columns(2)
    with cols[0]:
        if st.button("🔍 Dry run", use_container_width=True):
            with st.spinner("Previewing..."):
                res = apply_category_rules_to_transactions(
                    profile_name=profile_name, 
                    dry_run=True,
                    include_categorized=include_categorized
                )
            st.write(res)
    with cols[1]:
        if st.button("✅ Apply rules", type="primary", use_container_width=True):
            with st.spinner("Applying..."):
                res = apply_category_rules_to_transactions(
                    profile_name=profile_name, 
                    dry_run=False,
                    include_categorized=include_categorized
                )
            st.success(f"Updated {res['updated']} transactions")
            if res['updated']:
                st.balloons()
            st.rerun()


def _apply_rules_for_profiles(profiles: Optional[List[str]]) -> None:
    targets = profiles or [None]
    for profile_name in targets:
        apply_category_rules_to_transactions(
            profile_name=profile_name, 
            dry_run=False,
            include_categorized=True  # Apply to all transactions when rule is added
        )


if __name__ == '__main__':
    main()
