"""File operation utilities for safe filename handling and path management."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def safe_filename(name: str, default: str = 'file', max_length: Optional[int] = None) -> str:
    """Create a safe filename from a user-provided name.
    
    Removes or replaces characters that are not safe for filenames and
    normalizes the result. Preserves alphanumeric characters, spaces, underscores,
    and hyphens. Converts spaces to underscores.
    
    Args:
        name: The original filename or name to sanitize
        default: Default name to use if sanitization results in empty string
        max_length: Optional maximum length for the filename (truncates if provided)
        
    Returns:
        Sanitized filename safe for use in file systems
        
    Example:
        >>> safe_filename("My Budget 2024!")
        'My_Budget_2024'
        >>> safe_filename("", default="budget")
        'budget'
        >>> safe_filename("a" * 300, max_length=50)
        'a' * 50
    """
    if not name:
        return default
    
    # Keep only alphanumeric, spaces, underscores, and hyphens
    cleaned = ''.join(c for c in name if c.isalnum() or c in {' ', '_', '-'})
    cleaned = cleaned.strip().replace(' ', '_')
    
    # Remove consecutive underscores
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    
    # Apply length limit if specified
    if max_length is not None and len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    
    # Remove trailing underscores
    cleaned = cleaned.rstrip('_')
    
    return cleaned if cleaned else default


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: The directory path to ensure exists
        
    Returns:
        The path object (for chaining)
        
    Raises:
        OSError: If the directory cannot be created
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

