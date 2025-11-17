"""Shared sidebar components for multi-page dashboard.

This module provides common sidebar functionality that should be available
across all pages of the dashboard.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Any, Dict, Optional

try:
    from . import db
    from . import data_processing as dp
    from .personal_finance_ui import PersonalFinanceUI
    from .transfer_utils import annotate_transfers
    from .category_rules import (
        apply_rules_to_dataframe as _apply_rules_to_dataframe,
        apply_category_rules_to_transactions as _apply_rules_to_db,
        get_rules_hash as _get_rules_hash,
    )
    from .persistent_cache import load_cache as load_persistent_cache, save_cache as save_persistent_cache
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    import db
    import data_processing as dp
    from personal_finance_ui import PersonalFinanceUI
    from transfer_utils import annotate_transfers
    from category_rules import (
        apply_rules_to_dataframe as _apply_rules_to_dataframe,
        apply_category_rules_to_transactions as _apply_rules_to_db,
        get_rules_hash as _get_rules_hash,
    )
    from persistent_cache import load_cache as load_persistent_cache, save_cache as save_persistent_cache


def render_shared_sidebar() -> Dict:
    """Render shared sidebar elements available on all pages.
    
    Returns:
        Dict with keys: 'base_df', 'source_mode', 'filters', 'analytics'
    """
    ui = PersonalFinanceUI()
    cache = _get_persistent_cache()
    
    # Render header and dark mode toggle
    ui.render_header()
    ui.render_dark_mode_toggle()
    
    # Auto-load section
    st.sidebar.subheader("📂 Auto-Load Files")
    if st.sidebar.button("🔄 Scan data/raw directory"):
        status_box = st.sidebar.container()
        with status_box.status("Scanning raw files...", expanded=True) as status:
            results = db.auto_load_raw_files()

            if results['loaded_files']:
                status.write(f"✅ Loaded {len(results['loaded_files'])} new files")
                for filename in results['loaded_files']:
                    status.write(f"  • {filename}")

            if results['skipped_files']:
                status.write(f"⏭️ Skipped {len(results['skipped_files'])} already loaded files")

            if results['error_files']:
                status.write(f"❌ {len(results['error_files'])} files had errors:")
                for error in results['error_files']:
                    status.write(f"  • {error}")

            if results.get('issues'):
                status.write("⚠️ Normalization notes:")
                for fname, notes in results['issues'].items():
                    status.write(f"**{fname}**")
                    for note in notes:
                        status.write(f"  - {note}")

            if results['total_inserted'] > 0:
                status.write(f"📊 Imported {results['total_inserted']} new transactions, skipped {results['total_skipped']} duplicates")

            if not any([results['loaded_files'], results['skipped_files'], results['error_files']]):
                status.write("No files found in data/raw directory")

            status.update(label="Scan complete", state="complete")
        st.rerun()

    # Database management
    st.sidebar.subheader("🗄️ Database Management")
    if st.sidebar.button("🗑️ Clear Database", help="Delete all transactions from the database"):
        if 'confirm_clear' not in st.session_state:
            st.session_state.confirm_clear = False
        st.session_state.confirm_clear = True
    
    if st.session_state.get('confirm_clear', False):
        st.sidebar.warning("⚠️ This will delete ALL transactions!")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("✅ Confirm", key="confirm_clear_btn"):
                if db.clear_database():
                    st.session_state.confirm_clear = False
                    st.sidebar.success("Database cleared successfully!")
                    st.rerun()
        with col2:
            if st.button("❌ Cancel", key="cancel_clear_btn"):
                st.session_state.confirm_clear = False
                st.rerun()

    # File upload
    st.sidebar.subheader("📁 Data Import")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Bank Statement",
        type=["csv", "xls", "xlsx"],
        help="Upload your bank statement in CSV or Excel format"
    )

    # Data source selection
    source_options = ["Database (all history)", "Current upload only"]
    default_source = cache.get('source_mode') if cache.get('source_mode') in source_options else source_options[0]
    source_mode = st.sidebar.radio(
        "Data Source",
        options=source_options,
        index=source_options.index(default_source),
        help="Choose whether to analyze all persisted data or only the current upload",
    )
    if cache.get('source_mode') != source_mode:
        cache['source_mode'] = source_mode
        _persist_cache(cache)

    # Handle file upload
    current_upload = None
    if uploaded_file is not None:
        current_upload = _load_uploaded_file(uploaded_file)

    # Determine base dataset according to selected source
    if source_mode == "Database (all history)":
        base_df = db.fetch_transactions()
    else:
        base_df = current_upload.copy() if isinstance(current_upload, pd.DataFrame) else pd.DataFrame()

    transfer_details = annotate_transfers(base_df) if not base_df.empty else {
        'transfers_df': pd.DataFrame(),
        'internal_indices': set(),
        'internal_ids': set(),
        'pairs': pd.DataFrame(),
        'detection_mode': 'category-only',
    }

    _ensure_rules_synced(cache)

    # Render filters if data is available
    filters = {}
    if not base_df.empty:
        defaults = _sanitize_filter_defaults(cache.get('filters'))
        base_filters = ui.render_sidebar_filters(base_df, defaults=defaults)
        serialized = _serialize_filters_for_cache(base_filters)
        if serialized != defaults:
            cache['filters'] = serialized
            _persist_cache(cache)
        filters = base_filters
        filters['internal_transfer_indices'] = list(transfer_details.get('internal_indices', []))
        filters['internal_transfer_ids'] = list(transfer_details.get('internal_ids', []))
    
    return {
        'base_df': base_df,
        'source_mode': source_mode,
        'filters': filters,
        'current_upload': current_upload,
        'transfer_details': transfer_details,
    }


def _load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load and process uploaded file."""
    try:
        # Read the file
        df = dp.read_file(uploaded_file)
        
        if df.empty:
            st.sidebar.error("The uploaded file appears to be empty.")
            return None
        
        # Detect format and normalize columns
        normalized_df, profile_label = dp.detect_format_and_normalize(df, uploaded_file.name)
        profile_internal = normalized_df.attrs.get('profile_name') or profile_label
        
        # Show format detection result
        st.sidebar.info(f"📋 Detected format: **{profile_label}**")
        
        # Apply category rules to uncategorized transactions
        apply_rules_func = globals().get('_apply_rules_to_dataframe')
        if apply_rules_func is not None:
            normalized_df = apply_rules_func(normalized_df, profile_name=profile_internal)
        
        # Store in database with profile name
            inserted, skipped = db.upsert_transactions(
                normalized_df,
                account='manual_upload',
                source_file=uploaded_file.name,
                profile_name=profile_internal,
            )

            apply_db_rules = globals().get('_apply_rules_to_db')
            if apply_db_rules is not None:
                apply_db_rules(profile_internal)
        
        if inserted > 0:
            st.sidebar.success(f"✅ Imported {inserted} transactions")
        if skipped > 0:
            st.sidebar.info(f"⏭️ Skipped {skipped} duplicates")
        
        st.rerun()
        return normalized_df

    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")
        return None


def apply_filters(data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to data."""
    filtered_data = data.copy()

    date_col = filters.get('date_col') if filters else None
    start_date = pd.to_datetime(filters.get('start_date')) if filters and filters.get('start_date') else None
    end_date = pd.to_datetime(filters.get('end_date')) if filters and filters.get('end_date') else None

    if date_col and date_col in filtered_data.columns:
        filtered_data[date_col] = pd.to_datetime(filtered_data[date_col], errors='coerce')
        if start_date is not None:
            filtered_data = filtered_data[filtered_data[date_col] >= start_date]
        if end_date is not None:
            filtered_data = filtered_data[filtered_data[date_col] <= end_date]

    # Ensure amount is numeric for downstream calculations
    if 'Amount' in filtered_data.columns:
        filtered_data['Amount'] = pd.to_numeric(filtered_data['Amount'], errors='coerce')
        filtered_data = filtered_data.dropna(subset=['Amount'])

    # Category filter
    if filters and filters.get('selected_categories'):
        filtered_data = filtered_data[filtered_data['Category'].isin(filters['selected_categories'])]

    # Transaction type filter
    if filters and filters.get('transaction_types') and 'Type' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['Type'].isin(filters['transaction_types'])]

    # Amount range filter
    if filters and filters.get('amount_range'):
        min_amount, max_amount = filters['amount_range']
        filtered_data = filtered_data[
            (filtered_data['Amount'] >= min_amount) &
            (filtered_data['Amount'] <= max_amount)
        ]

    if filters and filters.get('expenses_only') and 'Amount' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['Amount'] < 0]

    internal_indices = set(filters.get('internal_transfer_indices', [])) if filters else set()
    if internal_indices:
        filtered_data = filtered_data[~filtered_data.index.isin(internal_indices)]

    search_text = (filters.get('search_text') or '').strip().lower() if filters else ''
    if search_text:
        search_cols = [col for col in ['Description', 'Category', 'Payee', 'Memo'] if col in filtered_data.columns]
        if search_cols:
            combined_mask = pd.Series(False, index=filtered_data.index)
            for col in search_cols:
                combined_mask = combined_mask | filtered_data[col].astype(str).str.lower().str.contains(search_text, na=False)
            filtered_data = filtered_data[combined_mask]

    return filtered_data


def _get_persistent_cache() -> Dict[str, Any]:
    cache = st.session_state.get('_persistent_cache_store')
    if cache is None:
        try:
            cache = load_persistent_cache()
        except Exception:  # pragma: no cover - defensive loading
            cache = {'source_mode': 'Database (all history)', 'filters': {}}
        if not isinstance(cache, dict):
            cache = {'source_mode': 'Database (all history)', 'filters': {}}
        st.session_state['_persistent_cache_store'] = cache
    cache.setdefault('source_mode', 'Database (all history)')
    if not isinstance(cache.get('filters'), dict):
        cache['filters'] = {}
    return cache


def _persist_cache(cache: Dict[str, Any]) -> None:
    try:  # pragma: no cover - disk IO
        save_persistent_cache(cache)
    except Exception:
        pass


def _serialize_filters_for_cache(filters: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in filters.items():
        if key in {'internal_transfer_indices', 'internal_transfer_ids'}:
            continue
        if key == 'date_range':
            if isinstance(value, (list, tuple)) and len(value) == 2:
                payload[key] = [_iso_date(value[0]), _iso_date(value[1])]
            else:
                payload[key] = []
        else:
            payload[key] = value
    return payload


def _sanitize_filter_defaults(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    cleaned = dict(raw)
    cleaned.pop('internal_transfer_indices', None)
    cleaned.pop('internal_transfer_ids', None)
    return cleaned


def _iso_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return str(value)


def _ensure_rules_synced(cache: Dict[str, Any]) -> None:
    rules_hash = _get_rules_hash() if '_get_rules_hash' in globals() else ''
    if not rules_hash:
        return
    cached_hash = cache.get('rules_hash')
    if cached_hash == rules_hash:
        return
    apply_func = globals().get('_apply_rules_to_db')
    if apply_func is None:
        return
    try:
        results = apply_func()
    except Exception:
        return
    cache['rules_hash'] = rules_hash
    _persist_cache(cache)
    updated = results.get('updated') if isinstance(results, dict) else None
    if updated:
        st.sidebar.success(f"Applied category rules to {updated} transaction(s)")
