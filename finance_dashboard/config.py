"""Configuration management for the finance dashboard.

This module centralizes all configuration values including paths,
defaults, and environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Base project root - assumes this file is in finance_dashboard/
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directories
DATA_DIR = Path(os.getenv("FINAPP_DATA_DIR", _PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
BUDGETS_DIR = DATA_DIR / "budgets"
GOALS_DIR = DATA_DIR / "goals"
REPORTS_DIR = DATA_DIR / "reports"

# Database
DB_PATH = Path(
    os.getenv("FINAPP_DB_PATH", DATA_DIR / "finance.db")
).resolve()

# Profiles directory
PROFILES_DIR = Path(
    os.getenv("FINAPP_PROFILES_DIR", Path(__file__).parent / "profiles")
).resolve()


def ensure_data_directories() -> None:
    """Create all required data directories if they don't exist."""
    for directory in [DATA_DIR, RAW_DATA_DIR, BUDGETS_DIR, GOALS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_raw_data_dir() -> str:
    """Get the raw data directory as a string (for backward compatibility)."""
    return str(RAW_DATA_DIR)


def get_db_path() -> str:
    """Get the database path as a string (for backward compatibility)."""
    return str(DB_PATH)

