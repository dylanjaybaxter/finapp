"""Recurring Payments Page - detect and manage repeat transactions."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure package imports resolve when run as a Streamlit page
PARENT = Path(__file__).parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from shared_sidebar import render_shared_sidebar, apply_filters
from finance_dashboard import db
try:  # pragma: no cover
    from finance_dashboard.plan_storage import load_plan as _plan_load_impl, save_plan as _plan_save_impl
except ModuleNotFoundError:  # pragma: no cover
    _plan_load_impl = None
    _plan_save_impl = None
from finance_dashboard.recurring import (
    detect_recurring,
    normalize_payee,
    monthly_plan_candidates,
)


def main() -> None:
    st.set_page_config(page_title="Recurring Payments", page_icon="🔁", layout="wide")
    sidebar_data = render_shared_sidebar()
    base_df = sidebar_data['base_df']
    filters = sidebar_data.get('filters', {})
    filtered_df = apply_filters(base_df, filters)

    st.header("🔁 Recurring Payments")
    if filtered_df.empty:
        st.info("Load some transactions to analyze recurring patterns.")
        return

    try:
        recurring_df, recurring_details = detect_recurring(filtered_df, return_details=True)
    except TypeError:
        recurring_df = detect_recurring(filtered_df)
        recurring_details = _build_detail_fallback(filtered_df)
        if not recurring_df.empty and 'Group Key' not in recurring_df.columns:
            if 'Amount Bucket' in recurring_df.columns:
                recurring_df['Group Key'] = recurring_df['Amount Bucket']
            else:
                recurring_df['Group Key'] = 'ALL'
    if recurring_df.empty:
        st.success("No recurring payments detected with the current filters.")
        return

    filtered, detail_filtered = _render_filter_panel(recurring_df, recurring_details)

    if filtered.empty:
        st.info("No recurring patterns meet the selected criteria.")
        return

    _ensure_plan_state()
    _render_recurring_workbench(filtered, detail_filtered, filtered_df)


def _render_recurring_workbench(recurring_df: pd.DataFrame, detail_df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    st.subheader("📋 Recurring Payments Workbench")
    st.caption("Review detected patterns, inspect the underlying transactions, and manually build your confirmed recurring payments plan.")

    explorer_tab, plan_tab, category_tab = st.tabs([
        "📂 Recurring Explorer",
        "✅ Confirmed Plan",
        "🗂 Category Tools",
    ])

    with explorer_tab:
        _render_explorer(recurring_df, detail_df)

    with plan_tab:
        _render_plan_builder(recurring_df, detail_df)

    with category_tab:
        _render_category_tools(recurring_df, source_df)


def _render_filter_panel(recurring_df: pd.DataFrame, detail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    with st.expander("Filters", expanded=True):
        max_occ = int(recurring_df['Occurrences'].max())
        default_min = min(3, max_occ)
        col_occ, col_consistent, col_focus, col_conf = st.columns(4)
        min_count = col_occ.slider(
            "Minimum occurrences",
            min_value=2,
            max_value=max_occ,
            value=default_min,
        )
        only_consistent = col_consistent.checkbox("Consistent only", value=False, help="Hide patterns with irregular timing.")
        essentials_only = col_focus.checkbox(
            "Focus essentials",
            value=False,
            help="Limit the view to housing, utilities, subscriptions, and loans.",
        )
        min_confidence = 0
        if 'Confidence Score' in recurring_df.columns:
            min_confidence = col_conf.slider(
                "Min confidence",
                min_value=0,
                max_value=100,
                value=60,
                step=5,
            )

        monthly_abs = recurring_df['Monthly Estimate'].abs()
        max_monthly = float(monthly_abs.max()) if not monthly_abs.empty else 0.0
        monthly_step = max(5.0, round(max_monthly / 20, 2)) if max_monthly else 5.0
        upper_default = max(max_monthly, 50.0)
        amount_range = st.slider(
            "Monthly estimate range ($)",
            min_value=0.0,
            max_value=float(round(upper_default + monthly_step, 2)),
            value=(0.0, float(round(upper_default, 2))),
            step=monthly_step,
        )

        type_options = sorted(recurring_df['Recurring Type'].dropna().unique().tolist())
        freq_options = sorted(recurring_df['Frequency Label'].dropna().unique().tolist())
        col_type, col_freq = st.columns(2)
        selected_types = col_type.multiselect(
            "Recurring type",
            options=type_options,
            default=type_options,
        )
        selected_freq = col_freq.multiselect(
            "Frequency",
            options=freq_options,
            default=freq_options,
        )

        search_term = st.text_input(
            "Search payee or category",
            help="Filter results by typing part of a payee name or category.",
        ).strip()

    filtered = recurring_df[recurring_df['Occurrences'] >= min_count].copy()
    if only_consistent and 'Is Consistent' in filtered.columns:
        filtered = filtered[filtered['Is Consistent']]
    if essentials_only:
        essential_types = {'Rent/Mortgage', 'Student Loan', 'Utility', 'Essential', 'Subscription'}
        filtered = filtered[filtered['Recurring Type'].isin(essential_types)]
    if min_confidence and 'Confidence Score' in filtered.columns:
        filtered = filtered[filtered['Confidence Score'] >= min_confidence]
    if selected_types:
        filtered = filtered[filtered['Recurring Type'].isin(selected_types)]
    if selected_freq:
        filtered = filtered[filtered['Frequency Label'].isin(selected_freq)]
    if amount_range and 'Monthly Estimate' in filtered.columns:
        monthly_abs = filtered['Monthly Estimate'].abs()
        filtered = filtered[(monthly_abs >= amount_range[0]) & (monthly_abs <= amount_range[1])]
    if search_term:
        lowered = search_term.lower()
        filtered = filtered[
            filtered['Payee'].str.lower().str.contains(lowered, na=False)
            | filtered['Category'].str.lower().str.contains(lowered, na=False)
        ]

    detail_filtered = _filter_detail_rows(detail_df, filtered)
    st.caption(f"Showing {len(filtered)} recurring pattern(s) after filters")
    return filtered, detail_filtered


def _render_explorer(recurring_df: pd.DataFrame, detail_df: pd.DataFrame) -> None:
    st.markdown("### Explore detected recurring payments")
    cols = st.columns(3)
    cols[0].metric("Recurring payees", len(recurring_df))
    cols[1].metric("Consistent patterns", int(recurring_df['Is Consistent'].sum()))
    cols[2].metric("Total est. monthly", f"${abs(recurring_df['Monthly Estimate'].sum()):,.2f}")

    recommended = monthly_plan_candidates(recurring_df)
    if not recommended.empty:
        preview = ", ".join(recommended['Payee'].head(4))
        st.info(f"Suggested additions for your plan: {preview} …", icon="💡")

    type_summary = (
        recurring_df.groupby('Recurring Type')['Monthly Estimate']
        .sum()
        .abs()
        .sort_values(ascending=False)
    )
    if not type_summary.empty:
        st.bar_chart(type_summary, use_container_width=True, height=220)

    display_cols = [
        'Recurring Key', 'Payee', 'Recurring Type', 'Category', 'Occurrences',
        'Frequency Label', 'Average Amount', 'Amount Range', 'Monthly Estimate', 'Confidence Score', 'Date Range', 'Is Consistent'
    ]
    explorer_table = recurring_df[display_cols].copy()
    for col in ['Average Amount', 'Monthly Estimate']:
        if col in explorer_table:
            explorer_table[col] = explorer_table[col].abs()
    column_config = {
        'Average Amount': st.column_config.NumberColumn("Avg Payment", format="$%0.2f"),
        'Monthly Estimate': st.column_config.NumberColumn("Monthly Est.", format="$%0.2f"),
        'Confidence Score': st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d%%"),
        'Occurrences': st.column_config.NumberColumn("Count"),
        'Frequency Label': st.column_config.TextColumn("Frequency"),
        'Amount Range': st.column_config.TextColumn("Amount Range"),
    }
    st.data_editor(
        explorer_table,
        use_container_width=True,
        hide_index=True,
        disabled=True,
        column_config=column_config,
        key='recurring_explorer_table'
    )

    csv_data = explorer_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download filtered recurring data",
        data=csv_data,
        file_name="recurring_patterns.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("#### Inspect a recurring payment")
    selected_key = st.selectbox(
        "Choose a pattern to inspect",
        options=recurring_df['Recurring Key'],
        key='explorer_selected_key'
    )
    selected_row = recurring_df.loc[recurring_df['Recurring Key'] == selected_key].iloc[0]

    cols = st.columns(4)
    cols[0].metric("Payee", selected_row['Payee'])
    cols[1].metric("Frequency", selected_row['Frequency Label'])
    cols[2].metric("Occurrences", int(selected_row['Occurrences']))
    cols[3].metric("Avg Payment", f"${abs(selected_row['Average Amount']):,.2f}")

    in_plan = selected_key in set(st.session_state.get('confirmed_recurring', []))
    action_col1, action_col2 = st.columns([1, 3])
    if in_plan:
        action_col1.success("Already in plan")
        if action_col2.button("Remove from confirmed plan", key=f"remove_{_safe_widget_key(selected_key)}"):
            _remove_from_plan(selected_key)
            _rerun()
    else:
        action_col1.warning("Not yet in plan")
        if action_col2.button("Add to confirmed plan", key=f"add_{_safe_widget_key(selected_key)}"):
            _add_to_plan([selected_key])
            _rerun()

    transactions = _get_transactions_for_key(selected_row, detail_df)
    if transactions.empty:
        st.info("No underlying transactions available for this pattern.")
    else:
        st.markdown("##### Transaction history")
        st.caption("Amounts are shown as absolute values for clarity.")
        chart_data = transactions.set_index('Transaction Date')['Amount']
        st.line_chart(chart_data)
        st.dataframe(
            transactions,
            use_container_width=True,
            hide_index=True,
        )


def _render_plan_builder(recurring_df: pd.DataFrame, detail_df: pd.DataFrame) -> None:
    st.markdown("### Build your confirmed recurring payments")
    confirmed = set(st.session_state.get('confirmed_recurring', []))

    selection_cols = [
        'Recurring Key', 'Payee', 'Recurring Type', 'Monthly Estimate', 'Frequency Label', 'Occurrences', 'Is Consistent', 'Confidence Score', 'Group Key'
    ]
    selection_table = recurring_df[selection_cols].copy()
    selection_table['Monthly Estimate'] = selection_table['Monthly Estimate'].abs()
    selection_table['Add to Plan'] = selection_table['Recurring Key'].isin(confirmed)

    edited_selection = st.data_editor(
        selection_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            'Monthly Estimate': st.column_config.NumberColumn("Monthly Est. ($)", format="$%0.2f"),
            'Add to Plan': st.column_config.CheckboxColumn("Add to Plan"),
            'Confidence Score': st.column_config.NumberColumn("Confidence", format="%0.1f"),
        },
        disabled=['Recurring Key', 'Payee', 'Recurring Type', 'Monthly Estimate', 'Frequency Label', 'Occurrences', 'Is Consistent', 'Confidence Score', 'Group Key'],
        key='plan_selector_editor'
    )

    selected_keys = set(edited_selection.loc[edited_selection['Add to Plan'], 'Recurring Key'])
    if selected_keys != confirmed:
        st.session_state['confirmed_recurring'] = sorted(selected_keys)
        confirmed = selected_keys
        _persist_plan_state()

    plan_entries = recurring_df[recurring_df['Recurring Key'].isin(confirmed)]
    if plan_entries.empty:
        st.info("Select at least one recurring payment above to start your confirmed plan.")
        return

    st.markdown("#### Suggested recurring payments")
    suggestion_rows = recurring_df[~recurring_df['Recurring Key'].isin(confirmed)]
    if 'Confidence Score' in suggestion_rows.columns:
        suggestion_rows = suggestion_rows.sort_values('Confidence Score', ascending=False)
    suggestion_rows = suggestion_rows.head(5)
    if suggestion_rows.empty:
        st.caption("No remaining suggestions — great job!")
    else:
        suggestion_cols = st.columns(len(suggestion_rows))
        for idx, (col, (_, row)) in enumerate(zip(suggestion_cols, suggestion_rows.iterrows())):
            col.metric(row['Payee'], f"{row['Confidence Score']:.0f}%", delta=row['Recurring Type'])
            quick_key = f"quick_add_{_safe_widget_key(row['Recurring Key'])}_{idx}"
            if col.button("Quick add", key=quick_key):
                _add_to_plan([row['Recurring Key']])
                _rerun()

    st.divider()
    st.markdown("#### Confirm details & adjust amounts")
    plan_entries_safe = plan_entries.rename(columns=lambda c: c.replace(' ', '_'))
    summary = {'avg': 0.0, 'min': 0.0, 'max': 0.0}
    overrides = st.session_state.get('plan_overrides', {})
    overrides_changed = False
    adjusted_rows = []
    for entry in plan_entries_safe.itertuples(index=False):
        safe_key = _safe_widget_key(entry.Recurring_Key)
        header = f"{entry.Payee} • {entry.Recurring_Type}"
        with st.expander(header, expanded=False):
            payee_cols = st.columns(3)
            override_defaults = overrides.get(entry.Recurring_Key, {})
            base_avg = float(abs(entry.Average_Amount))
            base_min = min(abs(entry.Amount_Range_Min), abs(entry.Amount_Range_Max))
            base_max = max(abs(entry.Amount_Range_Min), abs(entry.Amount_Range_Max))
            avg_input = payee_cols[0].number_input(
                "Average payment ($)",
                min_value=0.0,
                value=float(override_defaults.get('avg', base_avg)),
                key=f"avg_{safe_key}"
            )
            min_input = payee_cols[1].number_input(
                "Minimum payment ($)",
                min_value=0.0,
                value=float(override_defaults.get('min', base_min)),
                key=f"min_{safe_key}"
            )
            max_input = payee_cols[2].number_input(
                "Maximum payment ($)",
                min_value=0.0,
                value=float(override_defaults.get('max', base_max)),
                key=f"max_{safe_key}"
            )

            adjusted_rows.append({
                'key': entry.Recurring_Key,
                'payee': entry.Payee,
                'recurring_type': entry.Recurring_Type,
                'frequency': entry.Frequency_Label,
                'category': entry.Category,
                'avg': avg_input,
                'min': min_input,
                'max': max_input,
                'confidence': getattr(entry, 'Confidence_Score', None),
                'monthly_estimate': abs(getattr(entry, 'Monthly_Estimate', 0.0)),
            })

            payload = {'avg': avg_input, 'min': min_input, 'max': max_input}
            if overrides.get(entry.Recurring_Key) != payload:
                overrides[entry.Recurring_Key] = payload
                overrides_changed = True

            transactions = _get_transactions_for_key(
                {'Payee': entry.Payee, 'Group Key': getattr(entry, 'Group_Key', None)},
                detail_df
            )
            if transactions.empty:
                st.info("No transaction details available.")
            else:
                st.markdown("Transactions")
                st.dataframe(
                    transactions,
                    use_container_width=True,
                    hide_index=True,
                )
            if st.button("Remove from plan", key=f"plan_remove_{safe_key}"):
                _remove_from_plan(entry.Recurring_Key)
                _rerun()

    summary['avg'] = sum(row['avg'] for row in adjusted_rows)
    summary['min'] = sum(row['min'] for row in adjusted_rows)
    summary['max'] = sum(row['max'] for row in adjusted_rows)

    if overrides_changed:
        st.session_state['plan_overrides'] = overrides
        _persist_plan_state()

    st.markdown("#### Plan summary")
    cols = st.columns(3)
    cols[0].metric("Avg monthly", f"${summary['avg']:,.2f}")
    cols[1].metric("Minimum exposure", f"${summary['min']:,.2f}")
    cols[2].metric("Maximum exposure", f"${summary['max']:,.2f}")

    if adjusted_rows:
        export_df = pd.DataFrame([
            {
                'Payee': row['payee'],
                'Recurring Type': row['recurring_type'],
                'Frequency': row['frequency'],
                'Avg Payment ($)': row['avg'],
                'Min Payment ($)': row['min'],
                'Max Payment ($)': row['max'],
                'Monthly Estimate ($)': row['monthly_estimate'],
                'Confidence Score': row['confidence'],
            }
            for row in adjusted_rows
        ])
        st.download_button(
            "Download confirmed plan",
            data=export_df.to_csv(index=False).encode('utf-8'),
            file_name="recurring_plan.csv",
            mime="text/csv",
            help="Save your confirmed recurring payments as a CSV snapshot.",
            use_container_width=True,
        )


def _render_category_tools(recurring_df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    st.markdown("### Quick category adjustments")
    st.caption("Batch reclassify recurring payees as Subscription for easier reporting.")
    selection = st.multiselect(
        "Select recurring entries to mark as Subscription",
        options=recurring_df['Recurring Key'].tolist(),
        key='category_selection'
    )
    if st.button("Update categories", key='update_category_button') and selection:
        payees = sorted({entry.split(' | ')[0] for entry in selection})
        update_category_to_subscription(payees, source_df)
        st.success(f"Updated {len(payees)} payees to Subscription. Refresh to see the changes.")
        st.rerun()
    elif selection:
        st.info("Click 'Update categories' to apply your selection.")


def _filter_detail_rows(detail_df: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame()
    if filtered_df is None or filtered_df.empty:
        return detail_df.copy()
    if not {'Payee', 'Group Bucket'}.issubset(detail_df.columns):
        return detail_df.copy()
    if 'Group Key' not in filtered_df.columns:
        return detail_df.copy()
    merged = detail_df.merge(
        filtered_df[['Payee', 'Group Key']].drop_duplicates(),
        left_on=['Payee', 'Group Bucket'],
        right_on=['Payee', 'Group Key'],
        how='inner'
    )
    return merged.drop(columns=['Group Key']) if 'Group Key' in merged.columns else merged


def _get_transactions_for_key(recurring_row: pd.Series | dict, detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()
    if isinstance(recurring_row, dict):
        payee = recurring_row.get('Payee')
        group_key = recurring_row.get('Group Key') or recurring_row.get('Group_Key')
    else:
        payee = recurring_row['Payee']
        group_key = recurring_row.get('Group Key')
    subset = detail_df[detail_df['Payee'] == payee]
    if 'Group Bucket' in detail_df.columns and group_key is not None:
        subset = subset[subset['Group Bucket'] == group_key]
    if subset.empty:
        return pd.DataFrame()
    subset = subset.copy()
    if 'Transaction Date' in subset.columns:
        subset['Transaction Date'] = pd.to_datetime(subset['Transaction Date']).dt.date
        subset = subset.sort_values('Transaction Date')
    if 'Amount' in subset.columns:
        subset['Amount'] = subset['Amount'].abs().round(2)
    rename_map = {'profile_name': 'Profile', 'Description': 'Original Description'}
    subset = subset.rename(columns=rename_map)
    display_cols = [col for col in ['Transaction Date', 'Amount', 'Category', 'Original Description', 'Profile', 'id'] if col in subset.columns]
    return subset[display_cols]


def _safe_widget_key(value: str) -> str:
    return str(abs(hash(value)))


def _build_detail_fallback(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    detail = df.copy()
    if 'Description' in detail.columns:
        detail['Payee'] = detail['Description'].apply(normalize_payee)
    if 'Transaction Date' in detail.columns:
        detail['Transaction Date'] = pd.to_datetime(detail['Transaction Date'], errors='coerce')
    if 'Amount' in detail.columns:
        detail['Amount'] = pd.to_numeric(detail['Amount'], errors='coerce')
    detail['Amount Bucket'] = detail['Amount'].round(2) if 'Amount' in detail.columns else 0.0
    detail['Group Bucket'] = detail['Amount Bucket']
    keep_cols = [
        col for col in ['id', 'Description', 'Transaction Date', 'Amount', 'Category', 'Payee', 'Amount Bucket', 'Group Bucket', 'profile_name']
        if col in detail.columns
    ]
    return detail[keep_cols]


def _rerun() -> None:
    rerun_fn = getattr(st, 'rerun', None) or getattr(st, 'experimental_rerun', None)
    if rerun_fn:
        rerun_fn()


DEFAULT_PLAN_STATE = {'selection': [], 'overrides': {}}


def _load_plan_data() -> dict:
    if _plan_load_impl is None:
        return DEFAULT_PLAN_STATE.copy()
    try:
        data = _plan_load_impl()
        if not isinstance(data, dict):
            return DEFAULT_PLAN_STATE.copy()
        selection = data.get('selection') if isinstance(data.get('selection'), list) else []
        overrides = data.get('overrides') if isinstance(data.get('overrides'), dict) else {}
        return {'selection': selection, 'overrides': overrides}
    except Exception:  # pragma: no cover - safeguard runtime issues
        return DEFAULT_PLAN_STATE.copy()


def _save_plan_data(selection: list[str], overrides: dict) -> None:
    if _plan_save_impl is None:
        return
    try:
        _plan_save_impl(selection, overrides)
    except Exception:  # pragma: no cover - safeguard runtime issues
        pass


def _ensure_plan_state() -> None:
    if st.session_state.get('plan_state_initialized'):
        return
    data = _load_plan_data()
    st.session_state['plan_state_initialized'] = True
    st.session_state['confirmed_recurring'] = data.get('selection', [])
    st.session_state['plan_overrides'] = data.get('overrides', {})


def _persist_plan_state() -> None:
    selection = st.session_state.get('confirmed_recurring', [])
    overrides = st.session_state.get('plan_overrides', {})
    _save_plan_data(selection, overrides)


def _add_to_plan(keys: list[str]) -> None:
    if not keys:
        return
    current = set(st.session_state.get('confirmed_recurring', []))
    new_keys = set(keys)
    if new_keys.issubset(current):
        return
    st.session_state['confirmed_recurring'] = sorted(current | new_keys)
    _persist_plan_state()


def _remove_from_plan(key: str) -> None:
    current = [k for k in st.session_state.get('confirmed_recurring', []) if k != key]
    st.session_state['confirmed_recurring'] = current
    overrides = st.session_state.get('plan_overrides', {})
    overrides.pop(key, None)
    st.session_state['plan_overrides'] = overrides
    _persist_plan_state()


def update_category_to_subscription(payees: list[str], df: pd.DataFrame) -> None:
    if not payees:
        return
    payees_upper = [p.upper() for p in payees]
    df_local = df.copy()
    df_local['Payee'] = df_local['Description'].apply(normalize_payee)
    target_ids = df_local.loc[df_local['Payee'].isin(payees_upper), 'id'].dropna().astype(int).tolist()
    if not target_ids:
        return

    with db.connect() as conn:
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE transactions SET category = ? WHERE id = ?",
            [('Subscription', txn_id) for txn_id in target_ids]
        )
        conn.commit()


if __name__ == '__main__':
    main()
