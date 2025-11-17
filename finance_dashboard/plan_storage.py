"""Persistence helpers for confirmed recurring payment selections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

PLAN_FILE = Path(__file__).resolve().parents[1] / 'data' / 'recurring_plan.json'
DEFAULT_PLAN = {
    'selection': [],
    'overrides': {},
}


def load_plan(path: Path | None = None) -> Dict[str, object]:
    target = path or PLAN_FILE
    if not target.exists():
        return DEFAULT_PLAN.copy()
    try:
        with target.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return DEFAULT_PLAN.copy()
    selection = data.get('selection') or []
    overrides = data.get('overrides') or {}
    if not isinstance(selection, list):
        selection = []
    if not isinstance(overrides, dict):
        overrides = {}
    return {
        'selection': selection,
        'overrides': overrides,
    }


def save_plan(selection: List[str], overrides: Dict[str, Dict[str, float]], path: Path | None = None) -> None:
    target = path or PLAN_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'selection': sorted(set(selection or [])),
        'overrides': overrides or {},
    }
    with target.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
