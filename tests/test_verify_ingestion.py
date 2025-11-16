#!/usr/bin/env python3
"""Verify that every raw transaction row is ingested into the database."""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

# Import config - handle both package and script execution
try:
    from finance_dashboard.config import RAW_DATA_DIR, DB_PATH
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from finance_dashboard.config import RAW_DATA_DIR, DB_PATH


def count_rows(path: Path) -> int:
    if path.suffix.lower() == '.csv':
        with path.open('r', newline='', encoding='utf-8') as handle:
            reader = csv.reader(handle)
            next(reader, None)  # skip header
            return sum(1 for row in reader if any(cell.strip() for cell in row))
    raise ValueError(f"Unsupported test file extension: {path.suffix}")


def main() -> int:
    if not RAW_DATA_DIR.exists():
        print(f"Raw directory not found: {RAW_DATA_DIR}")
        return 1

    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return 1

    mismatches = []
    conn = sqlite3.connect(str(DB_PATH))
    try:
        for path in sorted(RAW_DATA_DIR.glob('**/*')):
            if path.is_dir() or path.suffix.lower() not in {'.csv'}:
                continue
            expected = count_rows(path)
            actual = conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE source_file = ?",
                (path.name,),
            ).fetchone()[0]
            if expected != actual:
                mismatches.append((path.name, expected, actual))
    finally:
        conn.close()

    if mismatches:
        print('Mismatch detected between raw rows and database records:')
        for name, expected, actual in mismatches:
            print(f"  - {name}: expected {expected} rows, found {actual}")
        return 1

    print('All raw transactions are present in the database for account auto_load.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

