"""Integration-style tests that validate raw bank exports ingest cleanly.

Each test spins up a temporary SQLite database, normalizes a specific
raw export using the profile registry, writes the transactions, and
confirms both the row counts and the presence of a handful of known
anchor transactions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finance_dashboard import data_processing as dp
from finance_dashboard import db as db_mod


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# (filename, expected_row_count, [(iso_date, description_fragment, amount), ...])
RAW_FILE_CASES: Tuple[Tuple[str, int, Iterable[Tuple[str, str, float]]], ...] = (
    (
        "Chase2997_Activity_20250926.CSV",
        9,
        (
            ("2025-09-12", "ANDURIL INDUSTRI PAYROLL", 4026.01),
            ("2025-09-09", "Extra Bonus", 400.00),
        ),
    ),
    (
        "Chase1337_Activity20220815_20240815_20240815.CSV",
        105,
        (
            ("2024-08-11", "TACO BELL #026126", -3.89),
            ("2024-07-23", "SPRUCE CAFE", -8.50),
        ),
    ),
    (
        "Chase1337_Activity20230926_20250926_20250926.CSV",
        951,
        (
            ("2025-09-21", "TACO BELL #027949", -3.64),
            ("2025-05-17", "BONFIRE.COM", -66.55),
        ),
    ),
    (
        "statements_0322_0624.csv",
        1748,
        (
            ("2024-06-28", "JOSEPH WARDELL", -4.00),
            ("2023-12-04", "MICROSOFT PC", -9.99),
        ),
    ),
    (
        "transactions_112120_021923.csv",
        1222,
        (
            ("2021-12-31", "MEET FRESH SAN JOSE", -7.66),
            ("2022-08-08", "VENMO 8558124430", -16.00),
        ),
    ),
    (
        "transactions_417937924.csv",
        1128,
        (
            ("2025-09-19", "MARK ATTAALLA", -18.00),
            ("2024-06-25", "CALPOLY ONTHEHUB", -29.99),
        ),
    ),
    (
        "transactions_832388047.csv",
        144,
        (
            ("2024-06-28", "JOSEPH WARDELL", -4.00),
            ("2024-05-01", "VILLAGE BAKER", -17.67),
        ),
    ),
)


@pytest.mark.parametrize("filename, expected_rows, anchors", RAW_FILE_CASES)
def test_raw_file_ingests_successfully(
    filename: str,
    expected_rows: int,
    anchors: Iterable[Tuple[str, str, float]],
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure each raw export fully loads into a temporary database."""

    db_path = tmp_path / "finance.db"
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path))
    db_mod.init_db()

    raw_path = RAW_DIR / filename
    assert raw_path.exists(), f"Missing raw fixture: {raw_path}"

    source_df = dp.read_file(raw_path)
    normalized_df, _ = dp.detect_format_and_normalize(source_df, str(raw_path))
    assert len(normalized_df) == expected_rows

    inserted, skipped = db_mod.upsert_transactions(normalized_df, account="test", source_file=filename)
    assert inserted == expected_rows
    assert skipped == 0

    with db_mod.connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        assert total == expected_rows
        by_source = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE source_file = ?",
            (filename,),
        ).fetchone()[0]
        assert by_source == expected_rows

        for iso_date, fragment, amount in anchors:
            row = conn.execute(
                """
                SELECT transaction_date, description, amount
                FROM transactions
                WHERE transaction_date = ? AND amount = ? AND description LIKE ?
                """,
                (iso_date, amount, f"%{fragment}%"),
            ).fetchone()
            assert row is not None, (
                f"Expected transaction not found for {filename}: {iso_date}, {fragment}, {amount}"
            )
