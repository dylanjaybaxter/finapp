#!/usr/bin/env python3
"""Lightweight validator for profile JSON definitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

PROFILE_DIR = Path(__file__).resolve().parents[1] / "finance_dashboard" / "profiles"


def validate_profile(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    errors = []
    name = data.get("name", path.stem)

    if "fields" not in data:
        errors.append("missing 'fields' block")
    else:
        field_block = data["fields"]
        if "column_mappings" not in field_block:
            errors.append("fields.column_mappings missing")
        elif not isinstance(field_block["column_mappings"], dict):
            errors.append("fields.column_mappings must be a dictionary")

    if errors:
        return {"name": name, "errors": "; ".join(errors)}
    return {}


def main() -> int:
    if not PROFILE_DIR.exists():
        print(f"Profile directory not found: {PROFILE_DIR}")
        return 1

    issues = []
    for path in sorted(PROFILE_DIR.glob('*.json')):
        result = validate_profile(path)
        if result:
            issues.append((path.name, result['errors']))

    if issues:
        print("Profile validation failed:")
        for filename, message in issues:
            print(f"  - {filename}: {message}")
        return 1

    print("All profiles validated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
