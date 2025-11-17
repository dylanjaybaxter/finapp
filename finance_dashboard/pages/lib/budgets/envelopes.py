"""Envelope management and configuration utilities.

This module handles envelope configuration normalization, default envelope
creation, savings entry management, and conversion between envelope formats.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Import config - handle both package and direct execution
try:
    from ...config import get_budget_config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path as _Path
    parent_dir = _Path(__file__).resolve().parents[3]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from pages.config import get_budget_config

# Import db for category fetching
try:
    from ... import db
except ImportError:
    import sys
    from pathlib import Path as _Path
    parent_dir = _Path(__file__).resolve().parents[3]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from finance_dashboard import db


def _get_config() -> Dict[str, Any]:
    """Get budget configuration."""
    return get_budget_config()


def _get_constants() -> Dict[str, Any]:
    """Get constants from configuration."""
    return _get_config()['constants']


def _get_default_envelopes_config() -> List[Dict[str, Any]]:
    """Get default envelopes from configuration."""
    return _get_config()['default_envelopes']


def _fetch_categories() -> List[str]:
    """Fetch all distinct categories from the database.
    
    Returns:
        List of category names
    """
    records = db.fetch_distinct_categories()
    return [c for c in records if c]


def get_default_envelopes() -> List[Dict[str, Any]]:
    """Get default envelope configuration from config file.
    
    Returns:
        List of default envelope dictionaries with Group, Type, Target %, and Categories
    """
    config_envelopes = _get_default_envelopes_config()
    
    # Convert config format to internal format
    envelopes = []
    for env in config_envelopes:
        envelopes.append({
            'Group': env['group'],
            'Type': env['type'],
            'Target %': env['target_percent'],
            'Categories': ', '.join(env.get('categories', [])),
        })
    
    return envelopes


def get_default_envelopes_with_all_categories() -> List[Dict[str, Any]]:
    """Get default envelopes with all categories from database assigned.
    
    Unassigned categories are distributed to Needs or Wants based on their names.
    
    Returns:
        List of envelope dictionaries with all categories assigned
    """
    from .categorization import categorize_expense
    
    all_categories = set(_fetch_categories())
    if not all_categories:
        # No categories in database yet, return defaults as-is
        return get_default_envelopes()
    
    # Get default envelopes
    default_envelopes = get_default_envelopes()
    
    # Get categories already assigned in default envelopes
    assigned_in_defaults = set()
    needs_env = None
    wants_env = None
    
    for env in default_envelopes:
        if env['Group'] == 'Essentials (Needs)':
            needs_env = env
            default_cats = [c.strip() for c in env.get('Categories', '').split(',') if c.strip()]
            assigned_in_defaults.update(default_cats)
        elif env['Group'] == 'Lifestyle (Wants)':
            wants_env = env
            default_cats = [c.strip() for c in env.get('Categories', '').split(',') if c.strip()]
            assigned_in_defaults.update(default_cats)
    
    # Find unassigned categories and distribute them
    unassigned = all_categories - assigned_in_defaults
    
    if unassigned and needs_env and wants_env:
        needs_cats = [c.strip() for c in needs_env.get('Categories', '').split(',') if c.strip()]
        wants_cats = [c.strip() for c in wants_env.get('Categories', '').split(',') if c.strip()]
        
        for cat in sorted(unassigned):
            target = categorize_expense(cat)
            if target == 'Essentials (Needs)':
                needs_cats.append(cat)
            else:
                wants_cats.append(cat)
        
        needs_env['Categories'] = ', '.join(sorted(set(needs_cats)))
        wants_env['Categories'] = ', '.join(sorted(set(wants_cats)))
    
    return default_envelopes


def normalize_envelope_config(entries: Any) -> List[Dict[str, Any]]:
    """Normalize envelope configuration, ensuring Type field is present and all categories are assigned.
    
    Args:
        entries: List of envelope dictionaries, or None/invalid to get defaults
        
    Returns:
        Normalized list of envelope dictionaries with all required fields
    """
    constants = _get_constants()
    savings_group_name = constants['savings_group_name']
    envelope_types = constants['envelope_types']
    expenses_type = envelope_types['expenses']
    savings_type = envelope_types['savings']
    
    if not isinstance(entries, list):
        return get_default_envelopes_with_all_categories()
    
    normalized = []
    all_categories = set(_fetch_categories())
    assigned_categories = set()
    
    for row in entries:
        if not isinstance(row, dict):
            continue
        
        # Determine type - default to expenses if not specified
        envelope_type = row.get('Type') or row.get('type', expenses_type)
        # If it's the savings group, ensure type is savings
        if row.get('Group') == savings_group_name:
            envelope_type = savings_type
        
        categories = row.get('Categories') or row.get('categories') or ''
        cat_list = [c.strip() for c in categories.split(',') if c.strip()]
        assigned_categories.update(cat_list)
        
        normalized.append({
            'Group': row.get('Group') or row.get('group') or 'Envelope',
            'Type': envelope_type,
            'Target %': float(row.get('Target %', row.get('target_pct', 0.0))),
            'Categories': categories,
        })
    
    # If no valid entries, return defaults with all categories
    if not normalized:
        return get_default_envelopes_with_all_categories()
    
    # Find unassigned categories and distribute them to Needs or Wants
    unassigned = all_categories - assigned_categories
    if unassigned:
        from .categorization import categorize_expense
        
        needs_env = next((e for e in normalized if e.get('Group') == 'Essentials (Needs)'), None)
        wants_env = next((e for e in normalized if e.get('Group') == 'Lifestyle (Wants)'), None)
        
        if needs_env and wants_env:
            needs_cats = [c.strip() for c in needs_env.get('Categories', '').split(',') if c.strip()]
            wants_cats = [c.strip() for c in wants_env.get('Categories', '').split(',') if c.strip()]
            
            for cat in sorted(unassigned):
                target = categorize_expense(cat)
                if target == 'Essentials (Needs)':
                    needs_cats.append(cat)
                else:
                    wants_cats.append(cat)
            
            needs_env['Categories'] = ', '.join(sorted(set(needs_cats)))
            wants_env['Categories'] = ', '.join(sorted(set(wants_cats)))
    
    return normalized


def ensure_savings_entry(config: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Ensure savings entry exists with proper Type field. General Savings has no categories.
    
    Args:
        config: List of envelope configuration dictionaries
        
    Returns:
        Updated configuration with savings entry guaranteed to exist
    """
    from .categorization import categorize_expense
    
    constants = _get_constants()
    savings_group_name = constants['savings_group_name']
    envelope_types = constants['envelope_types']
    expenses_type = envelope_types['expenses']
    savings_type = envelope_types['savings']
    
    config_copy = [dict(row) for row in config]
    
    # Get all categories and find which are assigned
    all_categories = set(_fetch_categories())
    assigned_categories = set()
    for row in config_copy:
        if row.get('Group') != savings_group_name:  # Don't count savings categories
            cats = [c.strip() for c in row.get('Categories', '').split(',') if c.strip()]
            assigned_categories.update(cats)
    
    unassigned = all_categories - assigned_categories
    
    found = False
    for row in config_copy:
        if row.get('Group') == savings_group_name:
            # Ensure Type is set to savings and categories are empty
            row['Type'] = savings_type
            row['Categories'] = ''  # General Savings has no categories
            found = True
            break
    
    if not found:
        # Calculate remainder for savings
        expenses_total = sum(
            float(row.get('Target %', 0.0)) 
            for row in config_copy 
            if row.get('Type', expenses_type) == expenses_type
        )
        savings_target = round(max(0.0, 100.0 - expenses_total), 2)
        config_copy.append({
            'Group': savings_group_name,
            'Type': savings_type,
            'Target %': savings_target,
            'Categories': '',  # General Savings has no categories
        })
    
    # Distribute any unassigned categories to Needs or Wants
    if unassigned:
        needs_env = next((e for e in config_copy if e.get('Group') == 'Essentials (Needs)'), None)
        wants_env = next((e for e in config_copy if e.get('Group') == 'Lifestyle (Wants)'), None)
        
        if needs_env and wants_env:
            needs_cats = [c.strip() for c in needs_env.get('Categories', '').split(',') if c.strip()]
            wants_cats = [c.strip() for c in wants_env.get('Categories', '').split(',') if c.strip()]
            
            for cat in sorted(unassigned):
                target = categorize_expense(cat)
                if target == 'Essentials (Needs)':
                    needs_cats.append(cat)
                else:
                    wants_cats.append(cat)
            
            needs_env['Categories'] = ', '.join(sorted(set(needs_cats)))
            wants_env['Categories'] = ', '.join(sorted(set(wants_cats)))
    
    return config_copy


def finalize_envelope_config(entries: List[Dict[str, str]], savings_template: Dict[str, str]) -> List[Dict[str, str]]:
    """Finalize envelope configuration, calculating savings as remainder.
    
    Args:
        entries: List of envelope configuration dictionaries
        savings_template: Template dictionary for savings entry (typically empty Categories)
        
    Returns:
        Finalized configuration with savings entry calculated as remainder
    """
    constants = _get_constants()
    savings_group_name = constants['savings_group_name']
    envelope_types = constants['envelope_types']
    expenses_type = envelope_types['expenses']
    savings_type = envelope_types['savings']
    
    normalized = []
    # Only count expenses envelopes for total allocation
    expenses_total = 0.0
    for row in entries:
        row_copy = dict(row)
        envelope_type = row_copy.get('Type', expenses_type)
        
        # Only expenses envelopes contribute to allocation
        if envelope_type == expenses_type:
            row_copy['Target %'] = max(0.0, min(100.0, float(row_copy.get('Target %', 0.0))))
            expenses_total += row_copy['Target %']
        else:
            # Savings envelopes don't have editable Target %
            row_copy['Target %'] = 0.0
        
        # Ensure Type field is present
        if 'Type' not in row_copy:
            row_copy['Type'] = expenses_type
        
        normalized.append(row_copy)
    
    # Calculate savings as remainder (only count expenses)
    expenses_total = min(expenses_total, 100.0)
    savings_target = round(max(0.0, 100.0 - expenses_total), 2)
    
    # Add or update savings entry
    savings_entry = {
        'Group': savings_group_name,
        'Type': savings_type,
        'Target %': savings_target,  # Auto-calculated, not editable
        'Categories': savings_template.get('Categories', '')  # General Savings has no categories
    }
    
    # Remove any existing savings entry and add the new one
    normalized = [e for e in normalized if e.get('Group') != savings_group_name]
    return normalized + [savings_entry]


def groups_from_envelopes(config: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Extract groups mapping from envelope configuration.
    
    Args:
        config: List of envelope configuration dictionaries
        
    Returns:
        Dictionary mapping group names to lists of category names
    """
    groups: Dict[str, List[str]] = {}
    for row in ensure_savings_entry(config):
        name = row.get('Group') or 'Lifestyle (Wants)'
        cats = [c.strip() for c in row.get('Categories', '').split(',') if c.strip()]
        groups[name] = cats
    return groups


def targets_from_envelopes(config: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract targets mapping from envelope configuration.
    
    Args:
        config: List of envelope configuration dictionaries
        
    Returns:
        Dictionary mapping group names to target percentages (as decimals, e.g., 0.5 for 50%)
    """
    targets: Dict[str, float] = {}
    for row in ensure_savings_entry(config):
        name = row.get('Group') or 'Lifestyle (Wants)'
        targets[name] = float(row.get('Target %', 0.0)) / 100.0
    return targets

