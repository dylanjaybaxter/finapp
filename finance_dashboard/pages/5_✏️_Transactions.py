"""Transactions Page - Edit and manage transactions."""

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

from shared_sidebar import render_shared_sidebar
import db


def main():
    """Render the Transactions editor page."""
    st.set_page_config(page_title="Transactions", page_icon="✏️", layout="wide")
    
    # Render shared sidebar
    render_shared_sidebar()
    
    st.header("✏️ Transaction Editor")
    st.markdown("Edit transfers, uncategorized transactions, and other transactions.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔄 Transfers", "❓ Uncategorized", "📋 All Transactions"])
    
    with tab1:
        _render_transfers_tab()
    
    with tab2:
        _render_uncategorized_tab()
    
    with tab3:
        _render_all_transactions_tab()


def _render_transfers_tab():
    """Render transfers tab."""
    st.subheader("🔀 Transfer Review")
    st.info("Transfers now live on the dedicated '🔀 Transfers' page for richer matching, flow visuals, and relabeling. Use the sidebar navigation to open it.")
    if hasattr(st, 'page_link'):
        st.page_link("pages/9_🔀_Transfers.py", label="Open Transfer Review", icon="🔀")
    else:
        st.markdown("[Open Transfer Review](#) — select the '🔀 Transfers' page from the sidebar navigation.")


def _render_uncategorized_tab():
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
            _render_transaction_edit_form(row)


def _render_all_transactions_tab():
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
                _render_transaction_edit_form(row)
    else:
        st.info("No transactions found in database.")


def _render_transaction_edit_form(transaction: pd.Series):
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


# Streamlit will execute this file directly
main()
