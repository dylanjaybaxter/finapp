"""Transactions Page - Edit and manage transactions with comprehensive search and detail view."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import json
from datetime import datetime
# Type hints are handled via __future__ annotations

# Import handling for Streamlit pages
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from shared_sidebar import render_shared_sidebar
import db

# Import budget/envelope functions
try:
    from pages.lib.budgets.envelopes import groups_from_envelopes
    from pages.lib.budgets.storage import load_saved_budgets
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(parent_dir / "pages"))
    from lib.budgets.envelopes import groups_from_envelopes
    from lib.budgets.storage import load_saved_budgets


def main():
    """Render the Transactions editor page."""
    st.set_page_config(page_title="Transactions", page_icon="✏️", layout="wide")
    
    # Render shared sidebar
    render_shared_sidebar()
    
    st.header("✏️ Transaction Editor")
    st.markdown("Search, view, and edit transactions with full metadata visibility.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search & View", "❓ Uncategorized", "🔄 Transfers"])
    
    with tab1:
        _render_search_and_view_tab()
    
    with tab2:
        _render_uncategorized_tab()
    
    with tab3:
        _render_transfers_tab()


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
    for _, row in uncategorized_df.iterrows():
        with st.expander(f"❓ {row.get('Description', 'Unknown')} - ${row.get('Amount', 0):,.2f} | {row.get('Transaction Date', 'N/A')}"):
            _render_transaction_edit_form(row)


def _render_search_and_view_tab():
    """Render comprehensive search and transaction detail view."""
    st.subheader("🔍 Search & View Transactions")
    st.markdown("Search the entire database and view comprehensive transaction details.")
    
    # Search filters
    with st.expander("🔍 Search Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input(
                "Search Text",
                placeholder="Description, memo, reference...",
                help="Searches in description, memo, and transaction references"
            )
            case_sensitive = st.checkbox(
                "Case Sensitive Search",
                value=False,
                help="If checked, search will be case-sensitive"
            )
            category = st.selectbox(
                "Category",
                options=[None] + db.fetch_distinct_categories(),
                format_func=lambda x: "All Categories" if x is None else x
            )
        
        with col2:
            profile_name = st.selectbox(
                "Profile",
                options=[None] + db.get_all_profiles(),
                format_func=lambda x: "All Profiles" if x is None else x
            )
            source_file = st.selectbox(
                "Source File",
                options=[None] + db.get_loaded_files(),
                format_func=lambda x: "All Files" if x is None else x
            )
        
        with col3:
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", value=None)
            with date_col2:
                end_date = st.date_input("End Date", value=None)
            
            amount_col1, amount_col2 = st.columns(2)
            with amount_col1:
                amount_min = st.number_input("Min Amount", value=None, min_value=0.0, step=0.01)
            with amount_col2:
                amount_max = st.number_input("Max Amount", value=None, min_value=0.0, step=0.01)
    
    # Search button
    if st.button("🔍 Search Transactions", type="primary", use_container_width=True):
        # Perform search
        results = db.search_transactions(
            search_term=search_term if search_term else None,
            category=category,
            profile_name=profile_name,
            source_file=source_file,
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None,
            amount_min=amount_min,
            amount_max=amount_max,
            limit=1000,  # Reasonable limit for performance
            case_sensitive=case_sensitive
        )
        
        st.session_state['search_results'] = results
        st.session_state['search_performed'] = True
        st.session_state['selected_transaction_id'] = None  # Reset selection
    
    # Initialize session state
    if 'search_performed' not in st.session_state:
        st.session_state['search_performed'] = False
    if 'search_results' not in st.session_state:
        st.session_state['search_results'] = pd.DataFrame()
    
    # Initialize session state
    if 'selected_transaction_id' not in st.session_state:
        st.session_state['selected_transaction_id'] = None
    
    # Display results
    if st.session_state.get('search_performed', False):
        results = st.session_state.get('search_results', pd.DataFrame())
        
        if results.empty:
            st.info("No transactions found matching your search criteria.")
        else:
            st.success(f"Found {len(results)} transaction(s)")
            
            # Display all results in a table
            st.markdown("### 📋 Search Results")
            st.caption("Select a transaction from the dropdown below to view detailed information")
            
            # Prepare display dataframe with key columns
            display_df = results.copy()
            display_df['Date'] = display_df['Transaction Date'].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and hasattr(x, 'strftime') else str(x)
            )
            display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:,.2f}")
            display_df['ID'] = display_df['id'].astype(int)
            
            # Select columns to display
            display_columns = ['ID', 'Date', 'Description', 'Category', 'Amount']
            if 'profile_name' in display_df.columns:
                display_columns.append('profile_name')
            
            display_df = display_df[display_columns].copy()
            if 'profile_name' in display_df.columns:
                display_df = display_df.rename(columns={'profile_name': 'Profile'})
            
            # Display table (read-only, for viewing all results)
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Transaction selector
            st.markdown("---")
            st.markdown("**Select Transaction to View Details:**")
            transaction_options = [
                f"ID {int(row['id'])}: {row.get('Description', 'Unknown')[:60]} - ${row.get('Amount', 0):,.2f} ({row.get('Transaction Date', 'N/A')})"
                for _, row in results.iterrows()
            ]
            
            # Get current selection index if available
            current_idx = None
            if st.session_state.get('selected_transaction_id'):
                current_id = st.session_state['selected_transaction_id']
                matching_idx = results[results['id'] == current_id].index
                if len(matching_idx) > 0:
                    current_idx = results.index.get_loc(matching_idx[0])
            
            selected_idx = st.selectbox(
                "Choose a transaction",
                options=range(len(results)),
                format_func=lambda i: transaction_options[i],
                key="transaction_selector",
                index=current_idx if current_idx is not None and current_idx < len(results) else 0
            )
            
            if selected_idx is not None and selected_idx < len(results):
                selected_transaction = results.iloc[selected_idx]
                st.session_state['selected_transaction_id'] = int(selected_transaction['id'])
                st.divider()
                _render_transaction_detail_view(selected_transaction)
    else:
        st.info("👆 Use the search filters above and click 'Search Transactions' to find transactions.")


def _render_transaction_detail_view(transaction: pd.Series):
    """Render comprehensive transaction detail view with all metadata."""
    transaction_id = transaction.get('id')
    if transaction_id is None:
        st.error("Transaction ID not found.")
        return
    
    st.divider()
    st.markdown("### 📋 Transaction Details")
    
    # Main transaction information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Amount", f"${transaction.get('Amount', 0):,.2f}")
    with col2:
        category = transaction.get('Category', 'Uncategorized')
        st.metric("Category", category if category else "❓ Uncategorized")
    with col3:
        txn_type = transaction.get('Type', 'N/A')
        st.metric("Type", txn_type if txn_type else "N/A")
    with col4:
        date = transaction.get('Transaction Date', 'N/A')
        if pd.notna(date):
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        else:
            date_str = 'N/A'
        st.metric("Date", date_str)
    
    # Detailed information sections
    detail_tabs = st.tabs([
        "📝 Transaction Info",
        "📁 Source & Processing",
        "🔧 Category Rules",
        "💼 Budget & Envelope",
        "✏️ Edit Transaction"
    ])
    
    with detail_tabs[0]:
        _render_transaction_info(transaction)
    
    with detail_tabs[1]:
        _render_source_info(transaction)
    
    with detail_tabs[2]:
        _render_rule_history(transaction)
    
    with detail_tabs[3]:
        _render_envelope_info(transaction)
    
    with detail_tabs[4]:
        _render_transaction_edit_form(transaction)


def _render_transaction_info(transaction: pd.Series):
    """Render basic transaction information."""
    st.markdown("#### Transaction Information")
    
    info_data = {
        "ID": transaction.get('id', 'N/A'),
        "Description": transaction.get('Description', 'N/A'),
        "Normalized Description": transaction.get('normalized_description', 'N/A'),
        "Amount": f"${transaction.get('Amount', 0):,.2f}",
        "Category": transaction.get('Category', 'Uncategorized'),
        "Type": transaction.get('Type', 'N/A'),
        "Transaction Date": str(transaction.get('Transaction Date', 'N/A')),
        "Post Date": str(transaction.get('Post Date', 'N/A')) if pd.notna(transaction.get('Post Date')) else 'N/A',
        "Memo": transaction.get('Memo', 'N/A') if pd.notna(transaction.get('Memo')) else 'N/A',
        "Currency": transaction.get('Currency', 'USD') if pd.notna(transaction.get('Currency')) else 'USD',
        "Account": transaction.get('account', 'N/A'),
    }
    
    # Additional fields if available
    if pd.notna(transaction.get('Transaction Reference')):
        info_data["Transaction Reference"] = transaction.get('Transaction Reference')
    if pd.notna(transaction.get('FI Transaction Reference')):
        info_data["FI Transaction Reference"] = transaction.get('FI Transaction Reference')
    if pd.notna(transaction.get('Original Amount')):
        info_data["Original Amount"] = f"${transaction.get('Original Amount'):,.2f}"
    
    for key, value in info_data.items():
        st.text(f"**{key}:** {value}")


def _render_source_info(transaction: pd.Series):
    """Render source file and processing information."""
    st.markdown("#### Source & Processing Information")
    
    source_file = transaction.get('source_file', 'N/A')
    profile_name = transaction.get('profile_name', 'N/A')
    imported_at = transaction.get('imported_at', 'N/A')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Source File:**")
        if source_file and source_file != 'N/A':
            st.info(f"📄 {source_file}")
        else:
            st.warning("No source file recorded")
        
        st.markdown(f"**Imported At:**")
        if imported_at and imported_at != 'N/A':
            try:
                if isinstance(imported_at, str):
                    dt = datetime.fromisoformat(imported_at.replace('Z', '+00:00'))
                    st.text(dt.strftime('%Y-%m-%d %H:%M:%S UTC'))
                else:
                    st.text(str(imported_at))
            except:
                st.text(str(imported_at))
        else:
            st.text("N/A")
    
    with col2:
        st.markdown(f"**Processing Profile:**")
        if profile_name and profile_name != 'N/A':
            st.success(f"🔧 {profile_name}")
        else:
            st.warning("No profile recorded")
        
        # Show profile details if available
        if profile_name and profile_name != 'N/A':
            try:
                from profile_manager import get_registry
                registry = get_registry()
                if profile_name in registry._profiles:
                    profile = registry._profiles[profile_name]
                    st.caption(f"Profile: {profile.description or profile_name}")
            except:
                pass


def _render_rule_history(transaction: pd.Series):
    """Render category rule application history."""
    st.markdown("#### Category Rule History")
    
    rule_history_str = transaction.get('rule_history')
    
    if rule_history_str and pd.notna(rule_history_str) and rule_history_str.strip():
        try:
            rule_history = json.loads(rule_history_str)
            
            if rule_history and isinstance(rule_history, list):
                st.success(f"✅ {len(rule_history)} rule(s) applied")
                
                for i, rule in enumerate(rule_history, 1):
                    with st.container():
                        st.markdown(f"**Rule #{i}** (Order: {rule.get('order', i)})")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text(f"Keyword: `{rule.get('keyword', 'N/A')}`")
                            st.text(f"Category: **{rule.get('category', 'N/A')}**")
                        with col2:
                            st.text(f"Case Sensitive: {rule.get('case_sensitive', False)}")
                            st.text(f"Whole Word: {rule.get('whole_word', False)}")
                        if rule.get('profiles'):
                            st.caption(f"Profiles: {', '.join(rule.get('profiles', []))}")
                        st.divider()
            else:
                st.info("Rule history is empty.")
        except json.JSONDecodeError:
            st.warning(f"⚠️ Could not parse rule history: {rule_history_str}")
    else:
        st.info("ℹ️ No category rules were applied to this transaction.")


def _render_envelope_info(transaction: pd.Series):
    """Render envelope/budget information."""
    st.markdown("#### Budget & Envelope Information")
    
    category = transaction.get('Category', '')
    
    if not category:
        st.warning("⚠️ Transaction has no category assigned. Cannot determine envelope.")
        return
    
    # Get current budget configuration
    try:
        saved_budgets = load_saved_budgets()
        default_env = None
        try:
            from pages.lib.budgets.envelopes import get_default_envelopes
            default_env = get_default_envelopes()
        except:
            pass
        
        envelopes = st.session_state.get('envelope_groups_config', default_env)
        
        if envelopes:
            groups_map = groups_from_envelopes(envelopes)
            
            # Find which envelope this category belongs to
            envelope_name = None
            for env_name, categories in groups_map.items():
                if category in categories:
                    envelope_name = env_name
                    break
            
            if envelope_name:
                st.success(f"📦 **Envelope:** {envelope_name}")
                
                # Show envelope details
                env_info = next((e for e in envelopes if e.get('Group') == envelope_name), None)
                if env_info:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(f"Type: {env_info.get('Type', 'N/A')}")
                        if 'Target %' in env_info:
                            st.text(f"Target: {env_info.get('Target %', 0):.1f}% of income")
                    with col2:
                        st.text(f"Categories in envelope: {len(categories)}")
            else:
                st.warning(f"⚠️ Category '{category}' is not assigned to any envelope in the current budget.")
        else:
            st.info("ℹ️ No budget configuration found. Create a budget to see envelope assignments.")
    except Exception as e:
        st.error(f"Error loading envelope information: {e}")


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
if __name__ == "__main__":
    main()
