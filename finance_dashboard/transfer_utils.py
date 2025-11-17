"""Utility helpers for identifying transfer transactions across accounts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd

TRANSFER_CATEGORY_LABELS = {'transfer', 'transfers', 'internal transfer'}
TRANSFER_TYPES = {'transfer'}


def annotate_transfers(df: pd.DataFrame, *, window_days: int = 3) -> Dict[str, object]:
    """Return metadata about transfer transactions within ``df``.

    The result contains:
        ``transfers_df``: filtered DataFrame of candidate transfers with labels.
        ``internal_indices``: row indices to hide from expense views.
        ``internal_ids``: database ids (when available) for quick access.
        ``pairs``: paired transfer summary rows for visualizations.
        ``detection_mode``: 'pairing' when matching succeeds, otherwise 'category-only'.
    """

    result = {
        'transfers_df': pd.DataFrame(),
        'internal_indices': set(),
        'internal_ids': set(),
        'pairs': pd.DataFrame(),
        'detection_mode': 'category-only',
    }

    if df is None or df.empty:
        return result

    working = df.copy()
    if 'Amount' not in working.columns:
        return result

    working['Amount'] = pd.to_numeric(working['Amount'], errors='coerce')
    working = working.dropna(subset=['Amount'])
    if working.empty:
        return result

    transfer_mask = _is_transfer_mask(working)
    transfers = working[transfer_mask].copy()
    if transfers.empty:
        return result

    transfers['Transaction Date'] = _preferred_date_series(transfers)
    transfers['profile_name'] = transfers.get('profile_name', 'Unknown').fillna('Unknown')
    transfers['__row_index__'] = transfers.index
    transfers['__abs_amount__'] = transfers['Amount'].abs().round(2)
    transfers['__sign__'] = np.where(transfers['Amount'] >= 0, 1, -1)
    transfers['__transfer_id__'] = (
        transfers['id'].astype(int)
        if 'id' in transfers.columns and transfers['id'].notna().all()
        else np.arange(len(transfers))
    )

    pairs = _match_transfer_pairs(transfers, window_days=window_days)
    result['pairs'] = pairs

    detection_mode = 'pairing' if not pairs.empty else 'category-only'
    result['detection_mode'] = detection_mode

    internal_row_indices: Set[int] = set()
    internal_ids: Set[int] = set()
    counterparty: Dict[int, str] = {}

    if not pairs.empty:
        for _, row in pairs.iterrows():
            src_idx = int(row['__row_index__out'])
            dst_idx = int(row['__row_index__in'])
            internal_row_indices.update([src_idx, dst_idx])
            counterparty[int(row['__transfer_id__out'])] = row.get('profile_name_in', 'Unknown')
            counterparty[int(row['__transfer_id__in'])] = row.get('profile_name_out', 'Unknown')
            if 'id' in transfers.columns:
                internal_ids.update([int(row['id_out']), int(row['id_in'])])
    else:
        # Fall back to excluding all transfer rows when we cannot match pairs
        internal_row_indices = set(transfers.index)
        if 'id' in transfers.columns:
            internal_ids = set(transfers['id'].dropna().astype(int).tolist())

    classification = transfers['__row_index__'].apply(
        lambda idx: 'Likely Internal' if idx in internal_row_indices else 'Needs Review'
    )
    transfers['Transfer Classification'] = classification
    transfers['Counterparty Profile'] = transfers['__transfer_id__'].map(counterparty).fillna('')

    clean_cols = [col for col in transfers.columns if not col.startswith('__')]
    transfers = transfers[clean_cols]

    result['transfers_df'] = transfers
    result['internal_indices'] = internal_row_indices
    result['internal_ids'] = internal_ids
    return result


def _is_transfer_mask(df: pd.DataFrame) -> pd.Series:
    category_series = df.get('Category', '').astype(str).str.strip().str.lower()
    type_series = df.get('Type', '').astype(str).str.strip().str.lower()
    return category_series.isin(TRANSFER_CATEGORY_LABELS) | type_series.isin(TRANSFER_TYPES)


def _preferred_date_series(df: pd.DataFrame) -> pd.Series:
    if 'Transaction Date' in df.columns:
        dates = pd.to_datetime(df['Transaction Date'], errors='coerce')
    elif 'Post Date' in df.columns:
        dates = pd.to_datetime(df['Post Date'], errors='coerce')
    else:
        dates = pd.Series(pd.NaT, index=df.index)
    if 'Post Date' in df.columns:
        fallback = pd.to_datetime(df['Post Date'], errors='coerce')
        dates = dates.fillna(fallback)
    return dates


def _match_transfer_pairs(transfers: pd.DataFrame, *, window_days: int) -> pd.DataFrame:
    negatives = transfers[transfers['__sign__'] < 0]
    positives = transfers[transfers['__sign__'] > 0]
    if negatives.empty or positives.empty:
        return pd.DataFrame()

    candidates = negatives.merge(
        positives,
        on='__abs_amount__',
        suffixes=('_out', '_in'),
        how='inner'
    )

    if not candidates.empty:
        candidates = _normalize_suffix_columns(candidates)

    if candidates.empty:
        return pd.DataFrame()

    date_diff = (candidates['Transaction Date_out'] - candidates['Transaction Date_in']).abs()
    within_window = date_diff <= pd.Timedelta(days=window_days)
    different_entry = candidates['__row_index__out'] != candidates['__row_index__in']
    different_profile = candidates['profile_name_out'] != candidates['profile_name_in']
    valid = within_window & different_entry & different_profile
    candidates = candidates[valid].copy()

    if candidates.empty:
        return pd.DataFrame()

    candidates = candidates.sort_values(['Transaction Date_out', 'Transaction Date_in'])
    used_out: Set[int] = set()
    used_in: Set[int] = set()
    rows: List[Dict[str, object]] = []
    for _, record in candidates.iterrows():
        idx_out = int(record['__row_index__out'])
        idx_in = int(record['__row_index__in'])
        if idx_out in used_out or idx_in in used_in:
            continue
        used_out.add(idx_out)
        used_in.add(idx_in)
        rows.append(record.to_dict())

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _normalize_suffix_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if '___' in col:
            rename_map[col] = col.replace('___', '__')
    if rename_map:
        df = df.rename(columns=rename_map)
    return df
