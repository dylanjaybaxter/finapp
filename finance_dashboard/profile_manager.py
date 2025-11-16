"""Profile management and normalization utilities for financial data ingestion.

This module centralises loading profile definitions, selecting the best match for a
raw dataset, and applying the associated column mappings and transformations to
produce a normalized DataFrame ready for database insertion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_header(name: str) -> str:
    """Normalise a header for comparison (lowercase alphanumerics only)."""
    return ''.join(ch for ch in str(name).lower() if ch.isalnum())


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _deep_merge(base: Any, incoming: Any) -> Any:
    """Recursively merge two profile fragments."""
    if isinstance(base, dict) and isinstance(incoming, dict):
        merged: Dict[str, Any] = {key: value for key, value in base.items()}
        for key, value in incoming.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    if isinstance(base, list) and isinstance(incoming, list):
        merged_list = list(base)
        for item in incoming:
            if item not in merged_list:
                merged_list.append(item)
        return merged_list
    return incoming


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Profile:
    name: str
    description: str
    match: Dict[str, Any]
    requirements: Dict[str, Any]
    fields: Dict[str, Any]
    transformations: Dict[str, Any]
    quality: Dict[str, Any]
    source_path: Path
    raw: Dict[str, Any]

    def required_fields(self, fallback: List[str]) -> List[str]:
        return _ensure_list(
            self.requirements.get('required_normalized_fields')
            or self.raw.get('required_normalized_fields')
            or fallback
        )


@dataclass
class ProfileMatch:
    profile: Profile
    mapping: Dict[str, str]  # normalized -> original column name
    score: float
    coverage: Dict[str, Any]
    missing_required: List[str]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ProfileRegistry:
    """Loads and matches profile definitions against raw DataFrames."""

    def __init__(self, profiles_dir: Optional[Path] = None) -> None:
        self.profiles_dir = profiles_dir or Path(__file__).with_name('profiles')
        self._profiles: Dict[str, Profile] = {}
        self._load_profiles()

    # Public API -------------------------------------------------------------

    def match(self, df: pd.DataFrame, filename: Optional[str]) -> ProfileMatch:
        best: Optional[ProfileMatch] = None
        for profile in self._profiles.values():
            match = self._score_profile(profile, df, filename)
            if best is None or match.score > best.score:
                best = match
        if best is None:
            raise ValueError("No profiles available for matching.")
        if not best.mapping:
            raise ValueError(
                "Unable to map source columns to a known profile. Extend profile definitions for this file."
            )
        return best

    # Internal ----------------------------------------------------------------

    def _load_profiles(self) -> None:
        raw_profiles: Dict[str, Dict[str, Any]] = {}
        for path in sorted(self.profiles_dir.glob('*.json')):
            with path.open('r', encoding='utf-8') as handle:
                data = json.load(handle)
            name = data.get('name') or path.stem
            data['name'] = name
            raw_profiles[name] = {'__path__': path, **data}

        resolved: Dict[str, Profile] = {}

        def resolve(name: str, stack: Optional[List[str]] = None) -> Profile:
            if stack is None:
                stack = []
            if name in resolved:
                return resolved[name]
            if name in stack:
                cycle = ' -> '.join(stack + [name])
                raise ValueError(f"Circular profile inheritance detected: {cycle}")
            if name not in raw_profiles:
                raise ValueError(f"Profile '{name}' referenced but not defined")
            raw = raw_profiles[name]
            base: Dict[str, Any] = {}
            parent_name = raw.get('extends')
            if parent_name:
                parent = resolve(parent_name, stack + [name])
                base = parent.raw
            merged_raw = json.loads(json.dumps(base)) if base else {}
            merged_raw = self._merge_dicts(merged_raw, {k: v for k, v in raw.items() if k != '__path__'})
            profile = Profile(
                name=merged_raw.get('name', name),
                description=merged_raw.get('description', merged_raw.get('name', name)),
                match=merged_raw.get('match', {}),
                requirements=merged_raw.get('requirements', {}),
                fields=merged_raw.get('fields', {}),
                transformations=merged_raw.get('transformations', {}),
                quality=merged_raw.get('quality', {}),
                source_path=raw.get('__path__'),
                raw=merged_raw,
            )
            resolved[name] = profile
            return profile

        for name in raw_profiles:
            resolve(name)
        self._profiles = resolved

    def _merge_dicts(self, base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base)
        for key, value in incoming.items():
            if key == 'extends':
                continue
            if key in result:
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _score_profile(self, profile: Profile, df: pd.DataFrame, filename: Optional[str]) -> ProfileMatch:
        column_lookup = {_normalise_header(col): col for col in df.columns}
        mapping: Dict[str, str] = {}
        column_mappings: Dict[str, Any] = profile.fields.get('column_mappings', {})
        coverage_details: Dict[str, Any] = {
            'total_columns': len(df.columns),
            'candidate_fields': len(column_mappings),
            'matched_fields': 0,
            'matched_required': 0,
        }

        required_fields = profile.required_fields(DEFAULT_REQUIRED_FIELDS)

        header_requirements = profile.match.get('header_contains', [])
        header_requirements = _ensure_list(header_requirements)
        missing_headers = [
            header for header in header_requirements
            if _normalise_header(header) not in column_lookup
        ]

        if missing_headers:
            coverage_details['missing_headers'] = missing_headers
            return ProfileMatch(
                profile=profile,
                mapping={},
                score=-1000,
                coverage=coverage_details,
                missing_required=required_fields,
            )

        for normalized, candidates in column_mappings.items():
            candidate_list = list(dict.fromkeys(_ensure_list(candidates) + [normalized]))
            for candidate in candidate_list:
                key = _normalise_header(candidate)
                if key in column_lookup:
                    mapping[normalized] = column_lookup[key]
                    break

        coverage_details['matched_fields'] = len(mapping)
        coverage_details['matched_required'] = len([f for f in required_fields if f in mapping])

        score = coverage_details['matched_fields']
        score += 4 * coverage_details['matched_required']

        patterns = _ensure_list(profile.match.get('filename_patterns'))
        if not patterns:
            patterns = _ensure_list(profile.raw.get('filename_patterns'))
        if filename and patterns:
            lowered = filename.lower()
            if any(pattern.lower() in lowered for pattern in patterns):
                score += 5

        missing_required = [field for field in required_fields if field not in mapping]
        if missing_required:
            score -= 10 * len(missing_required)

        return ProfileMatch(
            profile=profile,
            mapping=mapping,
            score=score,
            coverage=coverage_details,
            missing_required=missing_required,
        )


# Defaults used when profiles do not override them
DEFAULT_REQUIRED_FIELDS = ["Transaction Date", "Description", "Amount"]
DEFAULT_STANDARD_COLUMNS = [
    "Transaction Date",
    "Post Date",
    "Description",
    "Category",
    "Type",
    "Amount",
    "Memo",
    "Currency",
    "Transaction Reference",
    "FI Transaction Reference",
    "Original Amount",
]


# Convenience factory ---------------------------------------------------------

_registry: Optional[ProfileRegistry] = None


def get_registry() -> ProfileRegistry:
    global _registry
    if _registry is None:
        _registry = ProfileRegistry()
    return _registry
