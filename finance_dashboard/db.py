from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any
import re
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'finance.db')
DB_PATH = os.path.abspath(DB_PATH)

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT,
    transaction_date TEXT,
    post_date TEXT,
    description TEXT,
    normalized_description TEXT,
    category TEXT,
    type TEXT,
    amount REAL,
    memo TEXT,
    currency TEXT,
    transaction_reference TEXT,
    fi_transaction_reference TEXT,
    original_amount REAL,
    source_file TEXT,
    imported_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_txn_dedup
ON transactions (account, transaction_date, post_date, normalized_description, amount, transaction_reference);

CREATE INDEX IF NOT EXISTS ix_txn_date ON transactions (transaction_date);
CREATE INDEX IF NOT EXISTS ix_txn_category ON transactions (category);
CREATE INDEX IF NOT EXISTS ix_txn_amount ON transactions (amount);
"""


def _ensure_dirs() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def normalize_description(text: Optional[str]) -> str:
    if not text:
        return ''
    # Uppercase, strip, collapse whitespace and punctuation spacing
    t = text.strip().upper()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^A-Z0-9 \-\&\*\#\']", "", t)
    return t


@contextmanager
def connect() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        # Run migrations to add new columns if they don't exist
        _migrate_database(conn)


def _migrate_database(conn) -> None:
    """Add new columns to existing database if they don't exist."""
    cursor = conn.cursor()
    
    # Check if new columns exist and add them if they don't
    cursor.execute("PRAGMA table_info(transactions)")
    existing_columns = [row[1] for row in cursor.fetchall()]
    
    new_columns = [
        ('currency', 'TEXT'),
        ('transaction_reference', 'TEXT'),
        ('fi_transaction_reference', 'TEXT'),
        ('original_amount', 'REAL')
    ]
    
    for column_name, column_type in new_columns:
        if column_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE transactions ADD COLUMN {column_name} {column_type}")
                print(f"Added column {column_name} to transactions table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    print(f"Error adding column {column_name}: {e}")
    
    conn.commit()


def _to_iso_date(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    try:
        # pandas Timestamp or datetime
        if hasattr(value, 'to_pydatetime'):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            return value.date().isoformat()
        # strings like MM/DD/YYYY or YYYY-MM-DD
        ts = pd.to_datetime(value, errors='coerce')
        if pd.isna(ts):
            return None
        return ts.date().isoformat()
    except Exception:
        return None


def upsert_transactions(df: pd.DataFrame, account: str = 'default', source_file: Optional[str] = None) -> Tuple[int, int]:
    """Upsert transactions with deduplication.

    Returns (inserted_count, skipped_count).
    """
    if df.empty:
        return (0, 0)

    records: List[Tuple] = []
    imported_at = datetime.utcnow().isoformat()

    renamed = df.rename(columns={
        'Transaction Date': 'transaction_date',
        'Post Date': 'post_date',
        'Description': 'description',
        'Category': 'category',
        'Type': 'type',
        'Amount': 'amount',
        'Memo': 'memo',
        'Currency': 'currency',
        'Transaction Reference': 'transaction_reference',
        'FI Transaction Reference': 'fi_transaction_reference',
        'Original Amount': 'original_amount',
    })

    for _, row in renamed.iterrows():
        td = _to_iso_date(row.get('transaction_date'))
        pd_ = _to_iso_date(row.get('post_date'))
        desc = row.get('description')
        nd = normalize_description(desc)
        cat = row.get('category')
        typ = row.get('type')
        amt = float(row.get('amount')) if pd.notna(row.get('amount')) else None
        memo = row.get('memo')
        currency = row.get('currency')
        txn_ref = row.get('transaction_reference')
        fi_ref = row.get('fi_transaction_reference')
        orig_amt = float(row.get('original_amount')) if pd.notna(row.get('original_amount')) else None
        records.append((account, td, pd_, desc, nd, cat, typ, amt, memo, currency, txn_ref, fi_ref, orig_amt, source_file, imported_at))

    insert_sql = (
        "INSERT OR IGNORE INTO transactions (account, transaction_date, post_date, description, "
        "normalized_description, category, type, amount, memo, currency, transaction_reference, "
        "fi_transaction_reference, original_amount, source_file, imported_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    with connect() as conn:
        cur = conn.cursor()
        cur.executemany(insert_sql, records)
        conn.commit()
        inserted = cur.rowcount if cur.rowcount is not None else 0

        # Count duplicates skipped by comparing total vs unique constraint
        # Approximate skipped as total - inserted
        skipped = len(records) - inserted
        return (inserted, skipped)


def fetch_transactions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    amount_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    where: List[str] = []
    params: List[Any] = []

    if start_date:
        where.append("transaction_date >= ?")
        params.append(start_date)
    if end_date:
        where.append("transaction_date <= ?")
        params.append(end_date)
    if categories:
        where.append("category IN ({})".format(
            ",".join(["?" for _ in categories])
        ))
        params.extend(list(categories))
    if amount_range:
        where.append("amount BETWEEN ? AND ?")
        params.extend([amount_range[0], amount_range[1]])

    sql = "SELECT account, transaction_date AS 'Transaction Date', post_date AS 'Post Date', description AS 'Description', category AS 'Category', type AS 'Type', amount AS 'Amount', memo AS 'Memo', currency AS 'Currency', transaction_reference AS 'Transaction Reference', fi_transaction_reference AS 'FI Transaction Reference', original_amount AS 'Original Amount', source_file FROM transactions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY transaction_date ASC, id ASC"

    with connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    # Convert dates back to datetime for UI
    if not df.empty and 'Transaction Date' in df.columns:
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
        df['Post Date'] = pd.to_datetime(df['Post Date'])
    return df


def fetch_distinct_categories() -> List[str]:
    with connect() as conn:
        rows = conn.execute("SELECT DISTINCT category FROM transactions WHERE category IS NOT NULL ORDER BY category").fetchall()
    return [r[0] for r in rows if r[0] is not None]


def monthly_aggregates() -> pd.DataFrame:
    sql = """
    SELECT substr(transaction_date,1,7) AS Month,
           SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS Income,
           SUM(CASE WHEN amount < 0 THEN -amount ELSE 0 END) AS Expenses
    FROM transactions
    GROUP BY substr(transaction_date,1,7)
    ORDER BY Month
    """
    with connect() as conn:
        df = pd.read_sql_query(sql, conn)
    df['Net'] = df['Income'] - df['Expenses']
    df['SavingsRate'] = df.apply(lambda r: (r['Net']/r['Income']*100) if r['Income']>0 else 0.0, axis=1)
    return df


def get_loaded_files() -> List[str]:
    """Get list of files that have been imported into the database."""
    with connect() as conn:
        rows = conn.execute("SELECT DISTINCT source_file FROM transactions WHERE source_file IS NOT NULL").fetchall()
    return [row[0] for row in rows if row[0]]


def auto_load_raw_files(raw_dir: str = "data/raw") -> Dict[str, Any]:
    """Auto-load new files from data/raw directory.
    
    Returns dict with keys: loaded_files, skipped_files, error_files, total_inserted, total_skipped
    """
    import os
    import glob
    
    # Handle both relative and absolute imports
    try:
        from . import data_processing as dp
    except ImportError:
        # Fallback for direct execution
        import data_processing as dp
    
    results = {
        'loaded_files': [],
        'skipped_files': [],
        'error_files': [],
        'total_inserted': 0,
        'total_skipped': 0
    }
    
    if not os.path.exists(raw_dir):
        return results
    
    # Get already loaded files
    loaded_files = set(get_loaded_files())
    
    # Find all CSV and Excel files in raw directory
    file_patterns = [os.path.join(raw_dir, "*.csv"), os.path.join(raw_dir, "*.xlsx"), os.path.join(raw_dir, "*.xls")]
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern))
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # Skip if already loaded
        if filename in loaded_files:
            results['skipped_files'].append(filename)
            continue
        
        try:
            # Try to load the file
            df = dp.read_file(file_path)
            
            if df.empty:
                results['error_files'].append(f"{filename}: File is empty or contains no valid data")
                continue
            
            # Detect format and normalize columns
            normalized_df, format_name = dp.detect_format_and_normalize(df, file_path)
            
            # Check for required columns after normalization
            required_cols = ['Transaction Date', 'Description', 'Amount']
            missing_cols = [col for col in required_cols if col not in normalized_df.columns or normalized_df[col].isna().all()]
            
            if missing_cols:
                results['error_files'].append(f"{filename}: Missing required columns after format detection: {', '.join(missing_cols)}")
                continue
            
            # Try to upsert the data
            inserted, skipped = upsert_transactions(normalized_df, account='auto_load', source_file=filename)
            results['loaded_files'].append(f"{filename} ({format_name})")
            results['total_inserted'] += inserted
            results['total_skipped'] += skipped
            
        except Exception as e:
            error_msg = f"{filename}: {str(e)}"
            if "Unsupported file extension" in str(e):
                error_msg = f"{filename}: Unsupported file format. Expected CSV, XLS, or XLSX"
            elif "No columns to parse" in str(e):
                error_msg = f"{filename}: File appears to be empty or corrupted"
            elif "Error tokenizing data" in str(e):
                error_msg = f"{filename}: CSV parsing error - check delimiter and encoding"
            elif "Excel file format cannot be determined" in str(e):
                error_msg = f"{filename}: Excel file appears to be corrupted or password-protected"
            
            results['error_files'].append(error_msg)
    
    return results
