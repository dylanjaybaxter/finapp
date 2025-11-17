"""Data ingestion and financial analysis helpers.

This module contains pure functions for reading CSV/Excel files into
pandas DataFrames, detecting column types and computing a variety of
financial analyses.  These functions are designed to operate
independently of any user interface so that they can be unit tested and
reused in other contexts (e.g. command‑line scripts or REST APIs).

The ingestion layer integrates with profile definitions stored under
``finance_dashboard/profiles`` to provide a consistent normalization
workflow for heterogeneous bank exports.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # Allow both package and script execution contexts
    from .profile_manager import DEFAULT_STANDARD_COLUMNS, ProfileMatch, get_registry
except ImportError:  # pragma: no cover - fallback for direct execution
    from profile_manager import DEFAULT_STANDARD_COLUMNS, ProfileMatch, get_registry

# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def read_file(path_or_buffer) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame with enhanced heuristics."""
    csv_kwargs = {"index_col": False}

    if hasattr(path_or_buffer, "read"):
        name = getattr(path_or_buffer, "name", "uploaded_file.csv").lower()
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(path_or_buffer)
        return pd.read_csv(path_or_buffer, **csv_kwargs)

    path = Path(path_or_buffer)
    ext = path.suffix.lower()
    if ext in {".csv", ""}:
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        delimiters = [',', ';', '\t', '|']
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(path, encoding=encoding, delimiter=delimiter, **csv_kwargs)
                    if df.shape[1] > 1 or df.shape[0] > 1:
                        if not isinstance(df.index, pd.RangeIndex):
                            df = df.reset_index(drop=True)
                        return df
                except Exception:
                    continue
        # Final fallback
        df = pd.read_csv(path, **csv_kwargs)
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index(drop=True)
        return df
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file extension '{ext}'.")


# ---------------------------------------------------------------------------
# Profile driven normalization
# ---------------------------------------------------------------------------


def detect_format_and_normalize(df: pd.DataFrame, file_path: str = None) -> tuple[pd.DataFrame, str]:
    """Detect the best matching profile and normalize the dataset."""
    if df is None or df.empty:
        return df, "Unknown"

    registry = get_registry()
    filename = Path(file_path).name if file_path else None
    match = registry.match(df, filename)

    normalized_df, metadata = _apply_profile(match, df, filename)
    normalized_df.attrs.update(metadata)

    display_name = metadata.get('profile_description') or metadata.get('profile_name') or match.profile.name
    return normalized_df, display_name


def _apply_profile(match: ProfileMatch, df: pd.DataFrame, filename: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    profile = match.profile
    mapping = match.mapping  # normalized -> original
    issues: List[str] = []
    applied_transforms: List[str] = []

    normalized_df = df.copy()
    normalized_df.attrs['original_columns'] = list(df.columns)

    # Ensure normalized columns are present based on the mapping
    for normalized_col, original_col in mapping.items():
        if normalized_col in normalized_df.columns and original_col == normalized_col:
            continue
        if original_col in normalized_df.columns:
            normalized_df[normalized_col] = normalized_df[original_col]

    expected_columns = set(DEFAULT_STANDARD_COLUMNS)
    expected_columns.update(mapping.keys())
    expected_columns.update(profile.fields.get('column_mappings', {}).keys())
    for column in expected_columns:
        if column not in normalized_df.columns:
            normalized_df[column] = pd.NA

    # Apply default overrides if specified
    for column, value in profile.fields.get('default_values', {}).items():
        if column not in normalized_df.columns:
            normalized_df[column] = value
        else:
            normalized_df[column] = normalized_df[column].fillna(value)

    _sanitize_string_columns(normalized_df)
    issues.extend(_coerce_core_columns(normalized_df))

    issues.extend(_apply_category_groups(normalized_df, profile.fields.get('category_groups', {})))

    keyword_issues, keyword_hits = _apply_description_rules(
        normalized_df,
        profile.fields.get('description_keywords', {}),
    )
    issues.extend(keyword_issues)

    transform_issues, applied_transforms = _apply_transformations(normalized_df, profile.transformations)
    issues.extend(transform_issues)
    issues.extend(_apply_category_mappings(normalized_df, profile.fields.get('category_mappings', {})))
    issues.extend(_finalize_dataframe(normalized_df))

    if match.missing_required:
        issues.append(
            "Profile matched but required fields were missing in source: "
            + ", ".join(sorted(set(match.missing_required)))
        )

    metadata: Dict[str, Any] = {
        'profile_name': profile.name,
        'profile_description': profile.description,
        'profile_score': match.score,
        'profile_coverage': match.coverage,
        'column_mapping': mapping,
        'unmapped_columns': sorted(col for col in df.columns if col not in mapping.values()),
        'normalization_issues': issues,
        'applied_transformations': applied_transforms,
        'missing_required_fields': match.missing_required,
        'source_filename': filename,
        'category_mappings': profile.fields.get('category_mappings', {}),
        'consolidated_categories': profile.fields.get('category_groups', {}),
        'description_keyword_assignments': keyword_hits,
        'processed_at': datetime.utcnow().isoformat(),
    }
    profile_identity = metadata['profile_name']
    normalized_df['profile_name'] = profile_identity
    return normalized_df, metadata


def _sanitize_string_columns(df: pd.DataFrame) -> None:
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].apply(_clean_string_value)


def _clean_string_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else pd.NA
    return value


def _coerce_core_columns(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []
    for column in ['Transaction Date', 'Post Date']:
        if column in df.columns:
            parsed = pd.to_datetime(df[column], errors='coerce')
            invalid = parsed.isna() & df[column].notna()
            if invalid.any():
                issues.append(f"Unable to parse {invalid.sum()} values in '{column}'. They were set to NaT.")
            df[column] = parsed
    if 'Amount' in df.columns:
        numeric = pd.to_numeric(df['Amount'], errors='coerce')
        invalid = numeric.isna() & df['Amount'].notna()
        if invalid.any():
            issues.append(f"Unable to parse {invalid.sum()} amount values; fallback handling applied.")
        df['Amount'] = numeric
    return issues


def _apply_transformations(df: pd.DataFrame, transformations: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    if not transformations:
        return [], []
    issues: List[str] = []
    applied: List[str] = []

    amount_cfg = transformations.get('Amount')
    if amount_cfg:
        ttype = amount_cfg.get('type')
        if ttype == 'debit_credit':
            debit_fields = _ensure_list(amount_cfg.get('debit_fields'))
            credit_fields = _ensure_list(amount_cfg.get('credit_fields'))
            debit_total = pd.Series(0.0, index=df.index, dtype=float)
            credit_total = pd.Series(0.0, index=df.index, dtype=float)
            for field in debit_fields:
                if field in df.columns:
                    debit_total = debit_total.add(pd.to_numeric(df[field], errors='coerce').fillna(0.0), fill_value=0.0)
            for field in credit_fields:
                if field in df.columns:
                    credit_total = credit_total.add(pd.to_numeric(df[field], errors='coerce').fillna(0.0), fill_value=0.0)
            df['Amount'] = credit_total - debit_total
            applied.append('Amount:debit_credit')
        elif ttype == 'indicator_sign':
            indicator_field = amount_cfg.get('indicator_field')
            if indicator_field and indicator_field in df.columns:
                indicator_series = df[indicator_field].apply(lambda v: str(v).strip().upper() if pd.notna(v) else v)
                amount_series = pd.to_numeric(df['Amount'], errors='coerce')
                credit_values = {str(v).upper() for v in _ensure_list(amount_cfg.get('credit_values'))}
                debit_values = {str(v).upper() for v in _ensure_list(amount_cfg.get('debit_values'))}
                amount_series = amount_series.where(~indicator_series.isin(debit_values), -amount_series.abs())
                amount_series = amount_series.where(~indicator_series.isin(credit_values), amount_series.abs())
                df['Amount'] = amount_series
                applied.append('Amount:indicator_sign')
            else:
                issues.append("Indicator sign transformation skipped: indicator field missing in dataset.")
        else:
            issues.append(f"Unknown Amount transformation type '{ttype}'.")

    memo_cfg = transformations.get('Memo')
    if memo_cfg and memo_cfg.get('type') == 'append_fields':
        fields = _ensure_list(memo_cfg.get('fields'))
        separator = memo_cfg.get('separator', ' - ')
        base = df['Memo'].astype('string') if 'Memo' in df.columns else pd.Series(pd.NA, index=df.index, dtype='string')
        new_values: List[Optional[str]] = []
        for idx, row in df.iterrows():
            parts: List[str] = []
            base_value = base.at[idx]
            if pd.notna(base_value):
                parts.append(str(base_value).strip())
            for field in fields:
                if field in df.columns:
                    value = row[field]
                    if pd.notna(value) and str(value).strip():
                        parts.append(str(value).strip())
            new_values.append(separator.join(parts) if parts else pd.NA)
        df['Memo'] = pd.Series(new_values, index=df.index, dtype='string')
        applied.append('Memo:append_fields')

    return issues, applied


def _apply_category_mappings(df: pd.DataFrame, mappings: Dict[str, Dict[str, str]]) -> List[str]:
    issues: List[str] = []
    if not mappings:
        return issues
    for column, mapping in mappings.items():
        if column not in df.columns:
            continue
        mapped_col = f"{column} (Mapped)"
        upper_map = {str(k).upper(): v for k, v in mapping.items()}
        df[mapped_col] = df[column].apply(
            lambda v: upper_map.get(str(v).upper(), mapping.get(v, v)) if pd.notna(v) else v
        )
    return issues


def _apply_category_groups(df: pd.DataFrame, groups: Dict[str, Dict[str, Any]]) -> List[str]:
    """Map raw categories to consolidated categories per profile definition."""
    issues: List[str] = []
    if not groups:
        return issues

    for column, consolidated_map in groups.items():
        if column not in df.columns:
            continue

        raw_col = f"{column} (Raw)"
        if raw_col not in df.columns:
            df[raw_col] = df[column]

        value_map: Dict[str, str] = {}
        for consolidated, variants in consolidated_map.items():
            for variant in _ensure_list(variants):
                if variant is None:
                    continue
                value_map[str(variant).strip().lower()] = consolidated

        series = df[column].astype('string')
        consolidated_values: List[str] = []
        uncategorized = 0
        for value in series:
            if value is None or pd.isna(value):
                consolidated_values.append('Uncategorized')
                uncategorized += 1
                continue

            value_str = str(value).strip()
            if not value_str or value_str.lower() == 'none':
                consolidated_values.append('Uncategorized')
                uncategorized += 1
                continue

            key = value_str.lower()
            consolidated = value_map.get(key)
            if consolidated is None:
                consolidated_values.append('Uncategorized')
                uncategorized += 1
            else:
                consolidated_values.append(consolidated)

        df[column] = pd.Series(consolidated_values, index=df.index, dtype='string')
        if uncategorized:
            issues.append(
                f"{uncategorized} '{column}' values not mapped; defaulted to 'Uncategorized'."
            )

    return issues


def _apply_description_rules(df: pd.DataFrame, rules: Dict[str, Any]) -> Tuple[List[str], int]:
    if not rules or 'Description' not in df.columns:
        return [], 0

    issues: List[str] = []
    descriptions = df['Description'].astype('string').fillna('')
    raw_categories = df['Category'] if 'Category' in df.columns else pd.Series(pd.NA, index=df.index)
    assigned = 0

    normalized_rules: Dict[str, List[str]] = {
        category: [str(keyword).upper() for keyword in _ensure_list(keywords) if keyword]
        for category, keywords in rules.items()
    }

    for idx, description in descriptions.items():
        if description is None or pd.isna(description):
            continue
        description_str = str(description).strip()
        if not description_str:
            continue

        current_category = raw_categories.loc[idx] if idx in raw_categories.index else pd.NA
        current_category = _coerce_scalar(current_category)
        if current_category is not None and pd.notna(current_category):
            normalized_category = str(current_category).strip().lower()
            if normalized_category and normalized_category != 'uncategorized':
                continue
        description_upper = description_str.upper()
        for category, keywords in normalized_rules.items():
            if any(keyword in description_upper for keyword in keywords):
                df.at[idx, 'Category'] = category
                assigned += 1
                break

    if assigned:
        issues.append(f"Assigned categories to {assigned} transactions using description keywords.")
    return issues, assigned


def _finalize_dataframe(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []

    if 'Category' not in df.columns or df['Category'].isna().all():
        fallback_sources = [
            'Category (Mapped)',
            'Category (Raw)'
        ]
        for source in fallback_sources:
            if source in df.columns and df[source].notna().any():
                df['Category'] = df[source].fillna('Uncategorized')
                break
        else:
            df['Category'] = pd.Series(['Uncategorized'] * len(df), index=df.index, dtype='string')
    df['Category'] = df['Category'].astype('string')

    if 'Description' in df.columns:
        desc = df['Description'].astype('string')
        mask = desc.isna() | desc.str.strip().eq('')
        if mask.any():
            placeholders = []
            counter = 1
            for _ in df.index[mask]:
                placeholders.append(f"Unknown Transaction {counter}")
                counter += 1
            df.loc[mask, 'Description'] = placeholders
            issues.append(f"Filled {mask.sum()} missing descriptions with placeholders.")
        df['Description'] = df['Description'].astype('string')

    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        missing = df['Amount'].isna()
        if missing.any():
            df.loc[missing, 'Amount'] = 0.0
            issues.append(f"Replaced {missing.sum()} missing amounts with 0.0 for database compatibility.")
            memo = df['Memo'].astype('string') if 'Memo' in df.columns else pd.Series(pd.NA, index=df.index, dtype='string')
            memo = memo.fillna('')
            memo.loc[missing] = memo.loc[missing].apply(_add_amount_missing_note)
            df['Memo'] = memo

        # Harmonize amount sign based on mapped category labels when available
        classification_columns = [col for col in df.columns if col.endswith('(Mapped)')]
        if classification_columns:
            adjustments = {'Income': 'positive', 'Expense': 'negative'}
            adjusted_rows = 0
            for mapped_col in classification_columns:
                classifications = df[mapped_col]
                if classifications.isna().all():
                    continue
                for idx, label in classifications.items():
                    if pd.isna(label):
                        continue
                    desired = adjustments.get(str(label))
                    if not desired:
                        continue
                    current = df.at[idx, 'Amount']
                    if pd.isna(current):
                        continue
                    if desired == 'positive' and current < 0:
                        df.at[idx, 'Amount'] = abs(current)
                        adjusted_rows += 1
                    elif desired == 'negative' and current > 0:
                        df.at[idx, 'Amount'] = -abs(current)
                        adjusted_rows += 1
            if adjusted_rows:
                issues.append(f"Adjusted signs for {adjusted_rows} transactions based on mapped categories.")
    else:
        df['Amount'] = 0.0
        issues.append("Amount column missing in source. Defaulted to 0.0 for all rows.")

    return issues


def _add_amount_missing_note(value: str) -> str:
    note = 'Amount missing in source file'
    if not value:
        return note
    if note.lower() in value.lower():
        return value
    return f"{value} | {note}"


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_scalar(value: Any) -> Any:
    """Best-effort conversion of pandas objects to a scalar value."""
    if isinstance(value, pd.Series):
        if value.empty:
            return pd.NA
        return value.iloc[0]
    if isinstance(value, (np.ndarray, pd.Index)):
        if len(value) == 0:
            return pd.NA
        return value[0]
    return value


# ---------------------------------------------------------------------------
# Analytical helpers (unchanged)
# ---------------------------------------------------------------------------


def detect_date_column(df: pd.DataFrame, threshold: float = 0.8) -> Optional[str]:
    """Detect the first column containing mostly date values."""
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
        if df[col].dtype.kind in {"O", "i", "f"}:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            non_na = parsed.notna().sum()
            if len(df[col]) > 0 and non_na / len(df[col]) >= threshold:
                return col
    return None


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return all numeric column names, excluding datetimes."""
    return df.select_dtypes(include=["number"]).columns.tolist()


def detect_category_columns(df: pd.DataFrame, max_unique: int = 20) -> List[str]:
    """Detect object columns with a small number of unique values."""
    cats: List[str] = []
    for col in df.columns:
        if df[col].dtype == object:
            uniq = df[col].nunique(dropna=True)
            if 0 < uniq <= max_unique:
                cats.append(col)
    return cats


def aggregate_by_period(df: pd.DataFrame, date_col: str, numeric_col: str, freq: str = "D") -> pd.DataFrame:
    """Sum a numeric column by period based on a date column."""
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found")
    if numeric_col not in df.columns:
        raise ValueError(f"Numeric column '{numeric_col}' not found")
    tmp = df[[date_col, numeric_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    grouped = tmp.set_index(date_col).groupby(pd.Grouper(freq=freq))[numeric_col].sum()
    return grouped.to_frame(name=numeric_col)


def aggregate_by_category(df: pd.DataFrame, category_col: str, numeric_col: str) -> pd.Series:
    """Sum a numeric column for each category."""
    if category_col not in df.columns or numeric_col not in df.columns:
        raise ValueError(f"Columns '{category_col}' and/or '{numeric_col}' not found")
    return df.groupby(category_col)[numeric_col].sum().sort_values(ascending=False)


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for numeric columns."""
    return df.describe(include="number")


def compute_financial_ratios(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate a suite of financial ratios."""
    normalized = {col.lower().replace(" ", "").replace("_", ""): col for col in df.columns}
    ratios: Dict[str, float] = {}

    def get_value(keys: Sequence[str]) -> Optional[float]:
        for k in keys:
            if k in normalized:
                return float(df[normalized[k]].sum())
        return None

    current_assets = get_value(["currentassets", "currentasset"])
    current_liabilities = get_value(["currentliabilities", "currentliability"])
    if current_assets is not None and current_liabilities not in (None, 0):
        ratios["Current Ratio"] = current_assets / current_liabilities

    cash = get_value(["cash", "cashandcashequivalents", "cashandequivalents"])
    receivables = get_value(["accountsreceivable", "accountreceivable", "receivables"])
    marketable = get_value(["marketablesecurities", "shortterminvestments", "marketablesecurity"])
    if current_liabilities not in (None, 0):
        quick_num = sum(v for v in [cash, receivables, marketable] if v is not None)
        if quick_num > 0:
            ratios["Quick Ratio"] = quick_num / current_liabilities

    net_income = get_value(["netincome", "netprofit", "earnings"])
    revenue = get_value(["totalrevenue", "revenue", "sales", "netsales"])
    if net_income is not None and revenue not in (None, 0):
        ratios["Net Profit Margin"] = net_income / revenue

    equity = get_value(["totalequity", "shareholdersequity", "equity"])
    if net_income is not None and equity not in (None, 0):
        ratios["Return on Equity"] = net_income / equity

    assets = get_value(["totalassets", "assets"])
    if revenue not in (None, 0) and assets not in (None, 0):
        ratios["Asset Turnover"] = revenue / assets

    liabilities = get_value(["totalliabilities", "liabilities"])
    if liabilities not in (None, 0) and equity not in (None, 0):
        ratios["Debt to Equity"] = liabilities / equity

    gross_profit = get_value(["grossprofit", "grossincome"])
    if gross_profit is not None and revenue not in (None, 0):
        ratios["Gross Margin"] = gross_profit / revenue

    cogs = get_value(["costofgoodssold", "cogs", "costofsales"])
    inventory = get_value(["inventory", "totalinventory", "inventories"])
    if cogs not in (None, 0) and inventory not in (None, 0):
        ratios["Inventory Turnover"] = cogs / inventory

    return ratios


def horizontal_analysis(df: pd.DataFrame, date_col: str, numeric_cols: Sequence[str], freq: str = "M") -> pd.DataFrame:
    """Compute period‑over‑period changes for numeric columns."""
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
    tmp = tmp.dropna(subset=[date_col])
    grouped = tmp.set_index(date_col).groupby(pd.Grouper(freq=freq))[list(numeric_cols)].sum()
    return grouped.pct_change()


def vertical_analysis(df: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.DataFrame:
    """Compute vertical (common size) analysis for numeric columns."""
    sub = df[list(numeric_cols)].copy()
    row_sums = sub.sum(axis=1)
    return sub.div(row_sums, axis=0)


def dupont_analysis(df: pd.DataFrame) -> Dict[str, float]:
    """Perform DuPont decomposition of ROE."""
    normalized = {col.lower().replace(" ", "").replace("_", ""): col for col in df.columns}

    def get(keys: Sequence[str]) -> Optional[float]:
        for k in keys:
            if k in normalized:
                return float(df[normalized[k]].sum())
        return None

    net_income = get(["netincome", "netprofit", "earnings"])
    revenue = get(["totalrevenue", "revenue", "sales", "netsales"])
    assets = get(["totalassets", "assets"])
    equity = get(["totalequity", "shareholdersequity", "equity"])
    if None in (net_income, revenue, assets, equity) or revenue == 0 or assets == 0 or equity == 0:
        return {}
    net_margin = net_income / revenue
    asset_turnover = revenue / assets
    leverage = assets / equity
    roe = net_margin * asset_turnover * leverage
    return {
        "Net Profit Margin": net_margin,
        "Asset Turnover": asset_turnover,
        "Financial Leverage": leverage,
        "Return on Equity": roe,
    }
