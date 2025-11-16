"""Configuration loader for page-specific settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

# Configuration directory
CONFIG_DIR = Path(__file__).parent


def load_config(config_name: str) -> Dict[str, Any]:
    """Load a configuration file by name.
    
    Args:
        config_name: Name of the config file (without .json extension)
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the configuration file is invalid JSON
        
    Example:
        >>> config = load_config('budgets')
        >>> config['constants']['savings_group_name']
        'General Savings'
    """
    config_path = CONFIG_DIR / f"{config_name}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_budget_config() -> Dict[str, Any]:
    """Get the budget page configuration.
    
    Returns:
        Budget configuration dictionary with constants, keywords, defaults, and UI settings
        
    Example:
        >>> config = get_budget_config()
        >>> config['constants']['savings_group_name']
        'General Savings'
    """
    return load_config('budgets')


def get_config_value(config_name: str, *keys: str, default: Any = None) -> Any:
    """Get a nested configuration value by key path.
    
    Args:
        config_name: Name of the config file
        *keys: Path to the nested value (e.g., 'constants', 'savings_group_name')
        default: Default value if key path doesn't exist
        
    Returns:
        The configuration value at the specified path, or default if not found
        
    Example:
        >>> get_config_value('budgets', 'constants', 'savings_group_name')
        'General Savings'
    """
    try:
        config = load_config(config_name)
        value = config
        for key in keys:
            value = value[key]
        return value
    except (KeyError, FileNotFoundError):
        return default

