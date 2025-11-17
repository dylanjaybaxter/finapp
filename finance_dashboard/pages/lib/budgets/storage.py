"""Budget storage and file I/O operations.

This module handles all file operations for budgets including loading,
saving, and deleting budget files from disk.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..common.file_operations import safe_filename, ensure_directory

# Import config - handle both package and direct execution
try:
    from ...config import BUDGETS_DIR, ensure_data_directories
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path as _Path
    parent_dir = _Path(__file__).resolve().parents[3]
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from finance_dashboard.config import BUDGETS_DIR, ensure_data_directories


class BudgetStorage:
    """Handles budget file storage operations."""
    
    def __init__(self, budgets_dir: Optional[Path] = None):
        """Initialize budget storage.
        
        Args:
            budgets_dir: Optional custom directory for budget files.
                        Defaults to BUDGETS_DIR from config.
        """
        self.budgets_dir = budgets_dir or BUDGETS_DIR
        ensure_data_directories()
        ensure_directory(self.budgets_dir)
    
    def get_path(self, name: str) -> Path:
        """Get the file path for a budget by name.
        
        Args:
            name: Budget name
            
        Returns:
            Path object for the budget file
        """
        return self.budgets_dir / f"{safe_filename(name)}.json"
    
    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load all saved budgets from disk.
        
        Returns:
            Dictionary mapping budget names to budget data
            
        Note:
            Budgets with invalid or missing data are skipped with a warning.
        """
        budgets: Dict[str, Dict[str, Any]] = {}
        
        if not self.budgets_dir.exists():
            return budgets
        
        for budget_file in self.budgets_dir.glob('*.json'):
            name = budget_file.stem
            
            try:
                with budget_file.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, dict):
                    continue
                
                # Extract budget values
                entries = data.get('budgets', {})
                if entries is None:
                    entries = {}
                elif not isinstance(entries, dict):
                    entries = {}
                
                budget_dict: Dict[str, float] = {}
                for k, v in entries.items():
                    if v is not None:
                        try:
                            budget_dict[k] = float(v)
                        except (ValueError, TypeError):
                            continue
                
                # Extract and normalize envelopes
                from .envelopes import (
                    normalize_envelope_config,
                    groups_from_envelopes,
                    targets_from_envelopes,
                )
                envelopes = normalize_envelope_config(data.get('envelopes'))
                
                # Build complete budget entry
                budgets[name] = {
                    'name': name,
                    'budgets': budget_dict,
                    'envelopes': envelopes,
                    'targets': targets_from_envelopes(envelopes),
                    'groups': groups_from_envelopes(envelopes),
                    'saved_at': data.get('saved_at'),
                    'version': data.get('version', 1),
                }
            except (json.JSONDecodeError, OSError) as e:
                # Skip corrupted files
                try:
                    import streamlit as st
                    st.warning(f"⚠️ Could not load budget '{name}': {e}")
                except (AttributeError, RuntimeError, ImportError):
                    # Not in streamlit context or streamlit not available
                    pass
                continue
        
        return budgets
    
    def save(
        self,
        name: str,
        budgets: Dict[str, float],
        envelopes: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Save a budget to disk.
        
        Args:
            name: Budget name
            budgets: Dictionary mapping category names to budget amounts
            envelopes: Optional envelope configuration
            
        Raises:
            ValueError: If budget name is empty
            OSError: If file cannot be written
        """
        if not name or not name.strip():
            raise ValueError("Budget name cannot be empty")
        
        # Normalize budget values to floats
        normalized_budgets = {k: float(v) for k, v in budgets.items()}
        
        # Normalize envelopes
        from .envelopes import (
            normalize_envelope_config,
            groups_from_envelopes,
            targets_from_envelopes,
        )
        normalized_env = normalize_envelope_config(envelopes) if envelopes else []
        
        payload = {
            'name': name.strip(),
            'budgets': normalized_budgets,
            'envelopes': normalized_env,
            'groups': groups_from_envelopes(normalized_env),
            'targets': targets_from_envelopes(normalized_env),
            'saved_at': datetime.utcnow().isoformat(),
            'version': 1,
        }
        
        target = self.get_path(name)
        ensure_directory(target.parent)
        
        try:
            with target.open('w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        except OSError as e:
            raise OSError(f"Failed to save budget to {target}: {e}") from e
    
    def delete(self, name: str) -> None:
        """Delete a budget file from disk.
        
        Args:
            name: Budget name to delete
            
        Raises:
            ValueError: If budget name is empty
            OSError: If file cannot be deleted (but silently ignores missing files)
        """
        if not name or not name.strip():
            raise ValueError("Budget name cannot be empty")
        
        target = self.get_path(name)
        
        if not target.exists():
            return  # Silently ignore non-existent files
        
        try:
            target.unlink()
        except OSError as e:
            raise OSError(f"Failed to delete budget file {target}: {e}") from e


# Convenience functions using default storage
_default_storage = BudgetStorage()


def get_budget_path(name: str) -> Path:
    """Get the file path for a budget by name.
    
    Args:
        name: Budget name
        
    Returns:
        Path object for the budget file
    """
    return _default_storage.get_path(name)


def load_saved_budgets() -> Dict[str, Dict[str, Any]]:
    """Load all saved budgets from disk.
    
    Returns:
        Dictionary mapping budget names to budget data
    """
    return _default_storage.load_all()


def save_budget(
    name: str,
    budgets: Dict[str, float],
    envelopes: Optional[List[Dict[str, Any]]] = None
) -> None:
    """Save a budget to disk.
    
    Args:
        name: Budget name
        budgets: Dictionary mapping category names to budget amounts
        envelopes: Optional envelope configuration
        
    Raises:
        ValueError: If budget name is empty
        OSError: If file cannot be written
    """
    _default_storage.save(name, budgets, envelopes)


def delete_budget(name: str) -> None:
    """Delete a budget file from disk.
    
    Args:
        name: Budget name to delete
        
    Raises:
        ValueError: If budget name is empty
        OSError: If file cannot be deleted
    """
    _default_storage.delete(name)

