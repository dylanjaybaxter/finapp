#!/usr/bin/env python3
"""Show top uncategorized transactions to aid rule creation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_dashboard import db


def main(limit: int = 200) -> None:
    df = db.fetch_uncategorized_transactions()
    if df.empty:
        print("All transactions are categorized. 🎉")
        return

    print(f"Total uncategorized: {len(df)}")
    freq = df['Description'].value_counts().head(limit)
    print("\nTop descriptions:")
    print(freq.to_string())

    print("\nBy profile:")
    print(df['profile_name'].value_counts().to_string())

    sample_columns: List[str] = [
        col for col in ['Transaction Date', 'Description', 'Amount', 'profile_name', 'source_file']
        if col in df.columns
    ]
    print("\nSample rows:")
    print(df[sample_columns].head(20).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show uncategorized transaction stats.')
    parser.add_argument('--limit', type=int, default=200, help='How many top descriptions to show')
    args = parser.parse_args()
    main(limit=args.limit)
