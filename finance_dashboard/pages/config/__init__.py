"""Page configuration files and loaders.

This module provides configuration management for page-specific settings,
constants, and UI labels. Configuration is stored in JSON files for easy
modification without code changes.
"""

from .defaults import load_config, get_budget_config

__all__ = ['load_config', 'get_budget_config']

