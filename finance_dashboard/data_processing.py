"""Data ingestion and financial analysis helpers.

This module contains pure functions for reading CSV/Excel files into
pandas DataFrames, detecting column types and computing a variety of
financial analyses.  These functions are designed to operate
independently of any user interface so that they can be unit tested and
reused in other contexts (e.g. command‑line scripts or REST APIs).

New analyses can be added here as standalone functions.  Each
function should accept one or more pandas objects and return a
DataFrame, Series or scalar.  Avoid mutating input arguments and
document any assumptions or requirements in the docstring.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def read_file(path_or_buffer) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame with enhanced format detection.

    Accepts either a filesystem path or a file‑like object.  The
    extension is used to infer the file type when a path is given.  For
    in‑memory uploads (e.g. from Streamlit), pass a bytes buffer.  In
    that case the format defaults to CSV unless an ``.xlsx`` suffix is
    present on the name attribute of the buffer.

    Parameters
    ----------
    path_or_buffer : str or file‑like
        The path or file object to read.

    Returns
    -------
    pandas.DataFrame
        The loaded data.
    """
    if hasattr(path_or_buffer, "read"):
        # Assume a file‑like object
        name = getattr(path_or_buffer, "name", "uploaded_file.csv").lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(path_or_buffer)
        else:
            return pd.read_csv(path_or_buffer)
    else:
        # Assume a filesystem path
        _, ext = os.path.splitext(str(path_or_buffer))
        ext = ext.lower()
        if ext in {".csv", ""}:
            # Try different encodings and delimiters for CSV files
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            delimiters = [',', ';', '\t', '|']
            
            df = None
            for encoding in encodings:
                for delimiter in delimiters:
                    try:
                        df = pd.read_csv(path_or_buffer, encoding=encoding, delimiter=delimiter)
                        if len(df.columns) > 1:  # Success if we have multiple columns
                            break
                    except:
                        continue
                if df is not None and len(df.columns) > 1:
                    break
            
            if df is None or len(df.columns) <= 1:
                # Fallback to default
                df = pd.read_csv(path_or_buffer)
            return df
        elif ext in {".xls", ".xlsx"}:
            return pd.read_excel(path_or_buffer)
        else:
            raise ValueError(f"Unsupported file extension '{ext}'.")


def detect_format_and_normalize(df: pd.DataFrame, file_path: str = None) -> tuple[pd.DataFrame, str]:
    """Detect file format and normalize column names.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The raw data to normalize
    file_path : str, optional
        Path to the file for context
        
    Returns
    -------
    tuple
        (normalized_dataframe, format_name)
    """
    original_columns = df.columns.tolist()
    
    # Format 1: Standard format (Transaction Date, Description, Amount, etc.)
    standard_format = {
        'Transaction Date': ['transaction date', 'date', 'transaction_date', 'posting date', 'post date'],
        'Description': ['description', 'desc', 'memo', 'details', 'payee'],
        'Amount': ['amount', 'transaction amount', 'value'],
        'Category': ['category', 'cat', 'type', 'classification'],
        'Type': ['type', 'transaction type', 'debit/credit'],
        'Memo': ['memo', 'notes', 'reference', 'transaction reference']
    }

    # Format 2: Bank statement format (POSTED DATE, DESCRIPTION, AMOUNT, etc.)
    bank_format = {
        'Transaction Date': ['posted date', 'post date', 'transaction date', 'date'],
        'Description': ['description', 'desc', 'payee', 'merchant'],
        'Amount': ['amount', 'transaction amount', 'value'],
        'Category': ['category', 'cat', 'type', 'classification'],
        'Type': ['type', 'transaction type', 'credit/debit', 'debit/credit'],
        'Currency': ['currency', 'curr'],
        'Transaction Reference': ['transaction reference number', 'ref', 'reference', 'transaction ref'],
        'FI Transaction Reference': ['fi transaction reference', 'fi ref', 'bank ref'],
        'Original Amount': ['original amount', 'orig amount']
    }

    # Format 3: Bank of America export (Date, Debit/Credit, Balance, etc.)
    boa_format = {
        'Transaction Date': ['date'],
        'Description': ['description'],
        'Category': ['category'],
        'Type': ['type'],
        'Memo': ['memo'],
        'Debit': ['debit'],
        'Credit': ['credit'],
        'Balance': ['balance'],
        'Check Number': ['check no', 'check number'],
        'Sub Category': ['sub category', 'subcategory']
    }

    # Try to detect format by matching columns
    format_name = "Unknown"
    column_mapping = {}

    lower_columns = [col.lower() for col in original_columns]

    # Detect Bank of America format
    boa_required = {'date', 'description', 'debit', 'credit'}
    if boa_required.issubset(set(lower_columns)):
        format_name = "BOA"
        target_format = boa_format
    # Check for bank format next (more specific than standard)
    elif any(col.lower() in ['posted date', 'post date'] for col in original_columns):
        format_name = "Bank Statement Format"
        target_format = bank_format
    else:
        format_name = "Standard Format"
        target_format = standard_format
    
    # Create column mapping
    for target_col, possible_names in target_format.items():
        for original_col in original_columns:
            if any(name.lower() in original_col.lower() for name in possible_names):
                column_mapping[original_col] = target_col
                break
    
    # Normalize the dataframe
    normalized_df = df.copy()
    
    # Rename columns
    normalized_df = normalized_df.rename(columns=column_mapping)
    
    # Add missing columns with default values
    required_columns = ['Transaction Date', 'Description', 'Amount']
    for col in required_columns:
        if col not in normalized_df.columns:
            normalized_df[col] = None
    
    # Add optional columns if missing
    optional_columns = ['Category', 'Type', 'Memo', 'Currency', 'Transaction Reference', 
                       'FI Transaction Reference', 'Original Amount']
    for col in optional_columns:
        if col not in normalized_df.columns:
            normalized_df[col] = None
    
    # Add BOA-specific normalization
    if format_name == "BOA":
        debit_series = pd.to_numeric(normalized_df.get('Debit'), errors='coerce') if 'Debit' in normalized_df.columns else pd.Series(dtype=float)
        credit_series = pd.to_numeric(normalized_df.get('Credit'), errors='coerce') if 'Credit' in normalized_df.columns else pd.Series(dtype=float)

        if debit_series.empty:
            debit_series = pd.Series(0.0, index=normalized_df.index)
        else:
            debit_series = debit_series.fillna(0.0).apply(lambda x: -abs(x) if x != 0 else 0.0)

        if credit_series.empty:
            credit_series = pd.Series(0.0, index=normalized_df.index)
        else:
            credit_series = credit_series.fillna(0.0).apply(lambda x: abs(x))

        # Prefer debit values when present, otherwise fall back to credit
        amount_values = np.where(debit_series != 0, debit_series, credit_series)
        normalized_df['Amount'] = amount_values.astype(float)

        # Preserve additional context by appending sub-category to memo when available
        if 'Sub Category' in normalized_df.columns:
            memo_series = normalized_df.get('Memo')
            subcat_series = normalized_df.get('Sub Category')
            if memo_series is not None and subcat_series is not None:
                normalized_df['Memo'] = memo_series.fillna('').astype(str)
                normalized_df['Sub Category'] = subcat_series.fillna('').astype(str)
                normalized_df['Memo'] = (
                    normalized_df[['Memo', 'Sub Category']]
                    .apply(lambda parts: ' - '.join([p for p in parts if p and p.lower() != 'nan']).strip(' -') or None, axis=1)
                )

    # Add format metadata
    normalized_df.attrs['original_format'] = format_name
    normalized_df.attrs['original_columns'] = original_columns
    normalized_df.attrs['column_mapping'] = column_mapping

    return normalized_df, format_name


def detect_date_column(df: pd.DataFrame, threshold: float = 0.8) -> Optional[str]:
    """Detect the first column containing mostly date values.

    A column is considered date‑like if at least ``threshold`` fraction
    of its values can be parsed by :func:`pandas.to_datetime`.

    Parameters
    ----------
    df : pandas.DataFrame
        The data to inspect.
    threshold : float, optional
        Fraction of parseable values required to declare a column as
        a date column.  Defaults to 0.8.

    Returns
    -------
    Optional[str]
        The name of the detected date column, or ``None`` if none is
        found.
    """
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
    """Detect object columns with a small number of unique values.

    Parameters
    ----------
    df : pandas.DataFrame
        The data to inspect.
    max_unique : int, optional
        Maximum number of unique, non‑null values allowed for a column
        to be considered categorical.  Defaults to 20.

    Returns
    -------
    List[str]
        Names of detected categorical columns.
    """
    cats: List[str] = []
    for col in df.columns:
        if df[col].dtype == object:
            uniq = df[col].nunique(dropna=True)
            if 0 < uniq <= max_unique:
                cats.append(col)
    return cats


def aggregate_by_period(df: pd.DataFrame, date_col: str, numeric_col: str, freq: str = "D") -> pd.DataFrame:
    """Sum a numeric column by period based on a date column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    date_col : str
        Column containing date values (will be converted to datetime).
    numeric_col : str
        Column to sum.
    freq : str, optional
        Pandas offset alias (e.g. 'D', 'M', 'A') controlling the
        aggregation frequency.  Defaults to daily.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by period with a single column of sums.
    """
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
    """Sum a numeric column for each category.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    category_col : str
        The categorical column.
    numeric_col : str
        The numeric column to sum.

    Returns
    -------
    pandas.Series
        Series indexed by category with summed values.
    """
    if category_col not in df.columns or numeric_col not in df.columns:
        raise ValueError(f"Columns '{category_col}' and/or '{numeric_col}' not found")
    return df.groupby(category_col)[numeric_col].sum().sort_values(ascending=False)


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for numeric columns."""
    return df.describe(include="number")


def compute_financial_ratios(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate a suite of financial ratios.

    The function attempts to find key line items and computes
    liquidity, profitability, leverage and efficiency ratios.  See
    ``README.md`` for a full list of supported ratios.
    """
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
    """Compute period‑over‑period changes for numeric columns.

    Groups the data by the given frequency and calculates the
    percentage change from one period to the next.
    """
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    grouped = tmp.set_index(date_col).groupby(pd.Grouper(freq=freq))[list(numeric_cols)].sum()
    return grouped.pct_change()


def vertical_analysis(df: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.DataFrame:
    """Compute vertical (common size) analysis for numeric columns.

    Returns a DataFrame where each value is divided by the row‑wise sum
    across the selected numeric columns.
    """
    sub = df[list(numeric_cols)].copy()
    row_sums = sub.sum(axis=1)
    return sub.div(row_sums, axis=0)


def dupont_analysis(df: pd.DataFrame) -> Dict[str, float]:
    """Perform DuPont decomposition of ROE.

    Returns a dictionary containing net profit margin, asset turnover,
    financial leverage and the resulting ROE.  If any component is
    missing or zero, an empty dictionary is returned.
    """
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
