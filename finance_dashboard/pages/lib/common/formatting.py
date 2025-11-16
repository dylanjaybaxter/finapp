"""Formatting utilities for currency and text display."""

from __future__ import annotations

from typing import Union


def escape_dollar_for_markdown(amount: float) -> str:
    """Format a dollar amount and escape the dollar sign for markdown rendering.
    
    This function formats a dollar amount with proper comma separators and
    escapes the dollar sign to prevent markdown from interpreting it as a
    LaTeX math delimiter (which causes unintended italics).
    
    Args:
        amount: The dollar amount to format (e.g., 1234.56)
        
    Returns:
        Formatted string with escaped dollar sign (e.g., "\\$1,234.56")
        
    Example:
        >>> escape_dollar_for_markdown(1234.56)
        '\\$1,234.56'
    """
    return f"${amount:,.2f}".replace("$", "\\$")


def format_currency(amount: Union[float, int], include_sign: bool = True) -> str:
    """Format a currency amount with proper formatting.
    
    Args:
        amount: The amount to format
        include_sign: Whether to include the dollar sign
        
    Returns:
        Formatted currency string (e.g., "$1,234.56" or "1,234.56")
        
    Example:
        >>> format_currency(1234.56)
        '$1,234.56'
        >>> format_currency(1234.56, include_sign=False)
        '1,234.56'
    """
    formatted = f"{amount:,.2f}"
    return f"${formatted}" if include_sign else formatted

