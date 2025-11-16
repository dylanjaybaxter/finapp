"""Category categorization utilities.

This module provides functions to categorize expenses into Needs vs Wants
and to infer which envelope group a category belongs to.
"""

from __future__ import annotations

from typing import Dict, List, Optional

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


def _get_keywords() -> Dict[str, List[str]]:
    """Get keywords from configuration."""
    config = get_budget_config()
    return config['keywords']


def categorize_expense(category_name: str) -> str:
    """Categorize an expense category into 'Needs' or 'Wants' based on keywords.
    
    Args:
        category_name: The category name to categorize
        
    Returns:
        'Essentials (Needs)' or 'Lifestyle (Wants)'
        
    Example:
        >>> categorize_expense('Groceries')
        'Essentials (Needs)'
        >>> categorize_expense('Entertainment')
        'Lifestyle (Wants)'
    """
    keywords = _get_keywords()
    needs_keywords = set(kw.lower() for kw in keywords['needs'])
    wants_keywords = set(kw.lower() for kw in keywords['wants'])
    
    cat_lower = category_name.lower()
    
    # Check for needs keywords
    for keyword in needs_keywords:
        if keyword in cat_lower:
            return 'Essentials (Needs)'
    
    # Check for wants keywords
    for keyword in wants_keywords:
        if keyword in cat_lower:
            return 'Lifestyle (Wants)'
    
    # Default to Wants if unclear (better to over-budget wants than needs)
    return 'Lifestyle (Wants)'


def infer_group(category: Optional[str], groups_map: Dict[str, List[str]]) -> str:
    """Infer which envelope group a category belongs to.
    
    Args:
        category: Category name to look up (may be None)
        groups_map: Dictionary mapping group names to lists of categories
        
    Returns:
        Group name that contains the category, or first group in map if not found
        
    Example:
        >>> groups = {'Essentials (Needs)': ['Groceries'], 'Lifestyle (Wants)': ['Entertainment']}
        >>> infer_group('Groceries', groups)
        'Essentials (Needs)'
        >>> infer_group('Unknown', groups)
        'Essentials (Needs)'  # Returns first group as default
    """
    if not category:
        return next(iter(groups_map.keys()), 'Lifestyle (Wants)')
    
    for group, cats in groups_map.items():
        if category in cats:
            return group
    
    # Default to first group (or Wants if empty)
    return next(iter(groups_map.keys()), 'Lifestyle (Wants)')

