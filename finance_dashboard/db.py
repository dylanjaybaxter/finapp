from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any
import re

import pandas as pd

# Import configuration
try:
    from .config import DB_PATH, RAW_DATA_DIR, ensure_data_directories
except ImportError:
    from config import DB_PATH, RAW_DATA_DIR, ensure_data_directories

# Convert Path to string for backward compatibility with existing code
DB_PATH_STR = str(DB_PATH)

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
    ensure_data_directories()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalize_description(text: Optional[str]) -> str:
    if text is None or (isinstance(text, str) and not text.strip()):
        return ''
    if pd.isna(text):
        return ''
    # Uppercase, strip, collapse whitespace and punctuation spacing
    t = str(text).strip().upper()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^A-Z0-9 \-\&\*\#\']", "", t)
    return t


@contextmanager
def connect() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH_STR)
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
        ('original_amount', 'REAL'),
        ('profile_name', 'TEXT')
    ]
    
    for column_name, column_type in new_columns:
        if column_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE transactions ADD COLUMN {column_name} {column_type}")
                print(f"Added column {column_name} to transactions table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    print(f"Error adding column {column_name}: {e}")
    
    # Normalize historical profile names that stored descriptions instead of canonical IDs
    try:
        from .profile_manager import get_registry  # type: ignore
    except ImportError:
        from profile_manager import get_registry  # type: ignore

    registry = get_registry()
    for profile in registry._profiles.values():
        description = profile.description
        if description and description != profile.name:
            cursor.execute(
                "UPDATE transactions SET profile_name = ? WHERE profile_name = ?",
                (profile.name, description),
            )

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


def _parse_amount(value: Any) -> Optional[float]:
    """Convert different textual amount representations into floats."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        # Handle accounting negatives e.g. (123.45)
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = f"-{cleaned[1:-1]}"
        # Remove common currency markers
        cleaned = cleaned.replace("$", "").replace(",", "")
        cleaned = cleaned.replace("CR", "").replace("cr", "")
        value = cleaned
    parsed = pd.to_numeric([value], errors='coerce')
    number = parsed.iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def _sanitize_db_value(value: Any) -> Any:
    """Convert pandas NA/NaT and empty strings to SQLite-friendly values."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if pd.isna(value):
        return None
    return value


def upsert_transactions(df: pd.DataFrame, account: str = 'default', source_file: Optional[str] = None, profile_name: Optional[str] = None) -> Tuple[int, int]:
    """Upsert transactions with deduplication.

    Returns (inserted_count, skipped_count).
    """
    if df.empty:
        return (0, 0)

    records: List[Tuple] = []
    imported_at = datetime.utcnow().isoformat()
    skipped_invalid = 0

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

    for idx, row in renamed.iterrows():
        td = _to_iso_date(row.get('transaction_date'))
        post_d = _to_iso_date(row.get('post_date'))
        desc = row.get('description')
        if pd.isna(desc) or (isinstance(desc, str) and not desc.strip()):
            desc = f"Unknown Transaction #{idx + 1}"
        amt = _parse_amount(row.get('amount'))
        # Reject transactions missing mandatory fields to avoid corrupt records
        if amt is None:
            skipped_invalid += 1
            continue
        nd = normalize_description(desc)
        cat = _sanitize_db_value(row.get('category'))
        typ = _sanitize_db_value(row.get('type'))
        memo = _sanitize_db_value(row.get('memo'))
        currency = _sanitize_db_value(row.get('currency'))
        txn_ref = _sanitize_db_value(row.get('transaction_reference'))
        fi_ref = _sanitize_db_value(row.get('fi_transaction_reference'))
        orig_amt = _parse_amount(row.get('original_amount'))
        records.append((
            account,
            td,
            post_d,
            desc,
            nd,
            cat,
            typ,
            amt,
            memo,
            currency,
            txn_ref,
            fi_ref,
            orig_amt,
            _sanitize_db_value(source_file),
            _sanitize_db_value(profile_name),
            imported_at,
        ))

    insert_sql = (
        "INSERT OR IGNORE INTO transactions (account, transaction_date, post_date, description, "
        "normalized_description, category, type, amount, memo, currency, transaction_reference, "
        "fi_transaction_reference, original_amount, source_file, profile_name, imported_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    if not records:
        return (0, len(df) if skipped_invalid == 0 else skipped_invalid)

    with connect() as conn:
        cur = conn.cursor()
        before_changes = conn.total_changes
        cur.executemany(insert_sql, records)
        conn.commit()
        inserted = conn.total_changes - before_changes

    skipped_duplicates = len(records) - inserted
    total_skipped = skipped_invalid + skipped_duplicates
    return inserted, total_skipped


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

    sql = "SELECT id, account, transaction_date AS 'Transaction Date', post_date AS 'Post Date', description AS 'Description', category AS 'Category', type AS 'Type', amount AS 'Amount', memo AS 'Memo', currency AS 'Currency', transaction_reference AS 'Transaction Reference', fi_transaction_reference AS 'FI Transaction Reference', original_amount AS 'Original Amount', source_file, profile_name AS 'profile_name' FROM transactions"
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


def update_transaction(
    transaction_id: int,
    category: Optional[str] = None,
    description: Optional[str] = None,
    amount: Optional[float] = None,
    memo: Optional[str] = None,
    transaction_date: Optional[str] = None,
    txn_type: Optional[str] = None,
) -> bool:
    """Update a transaction in the database.
    
    Returns True if update was successful, False otherwise.
    """
    updates = []
    params = []
    
    if category is not None:
        updates.append("category = ?")
        params.append(category)
        # Update normalized description if category changes
        if description is None:
            # Get current description to update normalized_description
            with connect() as conn:
                row = conn.execute("SELECT description FROM transactions WHERE id = ?", (transaction_id,)).fetchone()
                if row:
                    desc = row[0]
                    updates.append("normalized_description = ?")
                    params.append(normalize_description(desc))
    
    if description is not None:
        updates.append("description = ?")
        params.append(description)
        updates.append("normalized_description = ?")
        params.append(normalize_description(description))
    
    if amount is not None:
        updates.append("amount = ?")
        params.append(amount)
    
    if memo is not None:
        updates.append("memo = ?")
        params.append(memo)
    
    if transaction_date is not None:
        updates.append("transaction_date = ?")
        params.append(_to_iso_date(transaction_date))

    if txn_type is not None:
        updates.append("type = ?")
        params.append(txn_type)
    
    if not updates:
        return False
    
    params.append(transaction_id)
    sql = f"UPDATE transactions SET {', '.join(updates)} WHERE id = ?"
    
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
        return cursor.rowcount > 0


def fetch_transactions_by_category(category: str, include_null: bool = False) -> pd.DataFrame:
    """Fetch transactions by category. If include_null is True, also includes NULL categories."""
    if include_null:
        sql = "SELECT id, account, transaction_date AS 'Transaction Date', post_date AS 'Post Date', description AS 'Description', category AS 'Category', type AS 'Type', amount AS 'Amount', memo AS 'Memo', currency AS 'Currency', transaction_reference AS 'Transaction Reference', fi_transaction_reference AS 'FI Transaction Reference', original_amount AS 'Original Amount', source_file FROM transactions WHERE category = ? OR category IS NULL ORDER BY transaction_date DESC, id DESC"
        params = [category]
    else:
        sql = "SELECT id, account, transaction_date AS 'Transaction Date', post_date AS 'Post Date', description AS 'Description', category AS 'Category', type AS 'Type', amount AS 'Amount', memo AS 'Memo', currency AS 'Currency', transaction_reference AS 'Transaction Reference', fi_transaction_reference AS 'FI Transaction Reference', original_amount AS 'Original Amount', source_file FROM transactions WHERE category = ? ORDER BY transaction_date DESC, id DESC"
        params = [category]
    
    with connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    if not df.empty and 'Transaction Date' in df.columns:
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
        if 'Post Date' in df.columns:
            df['Post Date'] = pd.to_datetime(df['Post Date'])
    return df


def fetch_uncategorized_transactions() -> pd.DataFrame:
    """Fetch all transactions with NULL or empty category."""
    sql = "SELECT id, account, transaction_date AS 'Transaction Date', post_date AS 'Post Date', description AS 'Description', category AS 'Category', type AS 'Type', amount AS 'Amount', memo AS 'Memo', currency AS 'Currency', transaction_reference AS 'Transaction Reference', fi_transaction_reference AS 'FI Transaction Reference', original_amount AS 'Original Amount', source_file, profile_name FROM transactions WHERE category IS NULL OR category = '' ORDER BY transaction_date DESC, id DESC"
    
    with connect() as conn:
        df = pd.read_sql_query(sql, conn)
    if not df.empty and 'Transaction Date' in df.columns:
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
        if 'Post Date' in df.columns:
            df['Post Date'] = pd.to_datetime(df['Post Date'])
    return df


def clear_database() -> bool:
    """Clear all transactions from the database. Returns True if successful."""
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transactions")
        conn.commit()
        return True


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


def get_files_by_profile(profile_name: str) -> List[str]:
    """Get list of files that were processed with a specific profile.
    
    Args:
        profile_name: Name of the profile to filter by
    
    Returns:
        List of source file names
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT source_file FROM transactions WHERE profile_name = ? AND source_file IS NOT NULL",
            (profile_name,)
        ).fetchall()
    return [row[0] for row in rows if row[0]]


def get_all_profiles() -> List[str]:
    """Get list of all profile names that have been used."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT profile_name FROM transactions WHERE profile_name IS NOT NULL ORDER BY profile_name"
        ).fetchall()
    return [row[0] for row in rows if row[0]]


def auto_load_raw_files(raw_dir: Optional[str] = None) -> Dict[str, Any]:
    """Auto-load new files from data/raw directory.
    
    Args:
        raw_dir: Optional path to raw data directory. If None, uses config default.
    
    Returns dict with keys: loaded_files, skipped_files, error_files, total_inserted, total_skipped
    """
    import os
    import glob
    
    # Use config default if not provided
    if raw_dir is None:
        raw_dir = str(RAW_DATA_DIR)
    
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
        'total_skipped': 0,
        'issues': {}
    }
    
    if not os.path.exists(raw_dir):
        return results
    
    # Get already loaded files
    loaded_files = set(get_loaded_files())
    
    # Find all CSV and Excel files in raw directory
    all_entries = glob.glob(os.path.join(raw_dir, "*"))
    allowed_exts = {".csv", ".xlsx", ".xls"}
    all_files = [
        path for path in all_entries
        if os.path.isfile(path) and Path(path).suffix.lower() in allowed_exts
    ]
    
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
            profile_identity = normalized_df.attrs.get('profile_name') or format_name

            issues = normalized_df.attrs.get('normalization_issues', [])
            if issues:
                results['issues'][filename] = issues
            if normalized_df['Amount'].dropna().empty:
                results['error_files'].append(
                    f"{filename}: Could not determine transaction amounts from profile '{format_name}'."
                )
                continue

            # Apply category rules to uncategorized transactions
            try:
                from .category_rules import apply_rules_to_dataframe, apply_category_rules_to_transactions
            except ImportError:
                from category_rules import apply_rules_to_dataframe, apply_category_rules_to_transactions
            normalized_df = apply_rules_to_dataframe(normalized_df, profile_name=profile_identity)

            # Try to upsert the data with profile name
            inserted, skipped = upsert_transactions(
                normalized_df,
                account='auto_load',
                source_file=filename,
                profile_name=profile_identity,
            )
            apply_category_rules_to_transactions(profile_identity)
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
