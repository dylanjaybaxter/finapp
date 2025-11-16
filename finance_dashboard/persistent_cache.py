"""Lightweight persistent cache for user-facing preferences/filters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

CACHE_PATH = Path(__file__).resolve().parents[1] / 'data' / 'persistent_cache.json'
DEFAULT_CACHE: Dict[str, Any] = {
    'source_mode': 'Database (all history)',
    'filters': {},
    'rules_hash': '',
}


def load_cache(path: Path | None = None) -> Dict[str, Any]:
    target = path or CACHE_PATH
    if not target.exists():
        return DEFAULT_CACHE.copy()
    try:
        with target.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return DEFAULT_CACHE.copy()
    if not isinstance(data, dict):
        return DEFAULT_CACHE.copy()
    merged = DEFAULT_CACHE.copy()
    merged.update({k: v for k, v in data.items() if k in DEFAULT_CACHE})
    return merged


def save_cache(cache: Dict[str, Any], path: Path | None = None) -> None:
    target = path or CACHE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
