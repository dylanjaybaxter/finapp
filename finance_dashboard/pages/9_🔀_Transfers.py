"""Dedicated transfer review workspace."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure package imports resolve when run as a Streamlit page
PARENT = Path(__file__).parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from shared_sidebar import render_shared_sidebar
try:  # pragma: no cover
    from finance_dashboard import db
except ModuleNotFoundError:  # pragma: no cover
    import db


def main() -> None:
    st.set_page_config(page_title="Transfers", page_icon="🔀", layout="wide")
    sidebar_data = render_shared_sidebar()
    base_df = sidebar_data['base_df']
    transfer_details = sidebar_data.get('transfer_details') or {}
    transfers = transfer_details.get('transfers_df', pd.DataFrame())
    detection_mode = transfer_details.get('detection_mode', 'category-only')
    pairs = transfer_details.get('pairs', pd.DataFrame())

    st.header("🔀 Transfer Review")
    st.caption("Internal transfers are automatically removed from expense views. Use this workspace to audit them and relabel true payments.")

    if base_df.empty:
        st.info("Load transactions or switch to the database source to review transfers.")
        return

    if transfers.empty:
        st.success("No transfer transactions detected with the current data.")
        return

    _render_summary(transfers, detection_mode, pairs)

    overview_tab, review_tab, archive_tab = st.tabs([
        "✅ Likely Internal",
        "⚠️ Needs Review",
        "📂 All Transfers",
    ])

    with overview_tab:
        _render_transfer_table(transfers, classification='Likely Internal', info_text="These offsets were detected across your accounts and are hidden from expenses." )

    with review_tab:
        _render_reclassification_tool(transfers)

    with archive_tab:
        _render_transfer_table(transfers, classification=None, info_text="Complete log of transfer-tagged transactions.")


def _render_summary(transfers: pd.DataFrame, detection_mode: str, pairs: pd.DataFrame) -> None:
    total = len(transfers)
    hidden = int((transfers['Transfer Classification'] == 'Likely Internal').sum())
    review = total - hidden
    gross_flow = float(transfers['Amount'].abs().sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Detected transfers", f"{total:,}")
    col2.metric("Hidden from spend", f"{hidden:,}")
    col3.metric("Gross transfer volume", f"${gross_flow:,.2f}")

    mode_label = "pair matching" if detection_mode == 'pairing' else "category fallback"
    st.caption(f"Detection mode: **{mode_label}**")

    amount_by_class = (
        transfers.assign(AbsAmount=transfers['Amount'].abs())
        .groupby('Transfer Classification')['AbsAmount']
        .sum()
        .reset_index()
    )
    if not amount_by_class.empty:
        fig = px.bar(
            amount_by_class,
            x='Transfer Classification',
            y='AbsAmount',
            color='Transfer Classification',
            labels={'AbsAmount': 'Amount ($)'},
            title='Transfer volume by classification',
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if isinstance(pairs, pd.DataFrame) and not pairs.empty:
        _render_flow_sankey(pairs)

    csv_data = transfers.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download transfer log",
        data=csv_data,
        file_name="transfers.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _render_flow_sankey(pairs: pd.DataFrame) -> None:
    summary = (
        pairs.groupby(['profile_name_out', 'profile_name_in'])['__abs_amount__']
        .sum()
        .reset_index()
    )
    if summary.empty:
        return
    labels = sorted(set(summary['profile_name_out']).union(summary['profile_name_in']))
    index_map = {label: idx for idx, label in enumerate(labels)}
    sources = [index_map[row['profile_name_out']] for _, row in summary.iterrows()]
    targets = [index_map[row['profile_name_in']] for _, row in summary.iterrows()]
    values = summary['__abs_amount__'].tolist()

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, pad=20, thickness=18),
                link=dict(source=sources, target=targets, value=values)
            )
        ]
    )
    fig.update_layout(title='Flow between accounts (matched transfers)')
    st.plotly_chart(fig, use_container_width=True)


def _render_transfer_table(transfers: pd.DataFrame, *, classification: str | None, info_text: str) -> None:
    subset = transfers if classification is None else transfers[transfers['Transfer Classification'] == classification]
    st.markdown(info_text)
    if subset.empty:
        st.info("No transactions to display.")
        return
    table = subset.copy()
    if 'Amount' in table.columns:
        table['Amount'] = table['Amount'].abs().round(2)
    display_cols = [col for col in ['id', 'Transaction Date', 'Description', 'Amount', 'Category', 'profile_name', 'Counterparty Profile', 'Transfer Classification'] if col in table.columns]
    st.dataframe(
        table[display_cols],
        use_container_width=True,
        hide_index=True,
    )


def _render_reclassification_tool(transfers: pd.DataFrame) -> None:
    subset = transfers[transfers['Transfer Classification'] != 'Likely Internal']
    if subset.empty:
        st.success("All remaining transfers have been matched internally.")
        return

    if 'id' not in subset.columns:
        st.info("Transaction IDs are not available for this data source. Relabeling requires imported database entries.")
        _render_transfer_table(subset, classification=None, info_text="Transfers requiring manual review")
        return

    st.markdown("### Relabel suspected payments")
    st.caption("Check the rows you want to treat as expenses. They will be recategorized as **Payment** so they flow into your spending analytics.")

    editor_df = subset.copy()
    editor_df['Amount'] = editor_df['Amount'].abs().round(2)
    editor_df = editor_df.rename(columns={
        'id': 'ID',
        'profile_name': 'Profile',
    })
    editor_df = editor_df[['ID', 'Transaction Date', 'Description', 'Amount', 'Category', 'Profile', 'Counterparty Profile']]
    editor_df['Relabel as Payment'] = False

    edited = st.data_editor(
        editor_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            'Amount': st.column_config.NumberColumn("Amount ($)", format="$%0.2f"),
            'Relabel as Payment': st.column_config.CheckboxColumn("Relabel"),
        },
        key='transfer_relabel_editor'
    )

    selected_ids = edited.loc[edited['Relabel as Payment'], 'ID'].astype(int).tolist()
    disabled = len(selected_ids) == 0
    apply_col, note_col = st.columns([1, 3])
    with apply_col:
        if st.button("✅ Mark as Payment", disabled=disabled):
            for txn_id in selected_ids:
                db.update_transaction(txn_id, category='Payment', txn_type='Payment')
            st.success(f"Updated {len(selected_ids)} transaction(s). They will now appear in expense views.")
            st.rerun()
    with note_col:
        st.caption("Already-classified transfers stay hidden automatically. Only selected rows are reclassified.")

    _render_transfer_table(subset, classification=None, info_text="Transfers flagged for manual review")


if __name__ == '__main__':
    main()
