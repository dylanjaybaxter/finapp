"""Budget-specific utilities and business logic.

This module provides all budget-related functionality including:
- Budget storage and file I/O operations
- Envelope management and configuration
- Budget calculations and suggestions
- Category categorization
- UI components for budget pages
"""

from .storage import (
    BudgetStorage,
    load_saved_budgets,
    save_budget,
    delete_budget,
    get_budget_path,
)
from .envelopes import (
    normalize_envelope_config,
    get_default_envelopes,
    get_default_envelopes_with_all_categories,
    ensure_savings_entry,
    finalize_envelope_config,
    groups_from_envelopes,
    targets_from_envelopes,
)
from .categorization import (
    categorize_expense,
    infer_group,
)
from .calculations import (
    suggested_amounts,
    suggested_group_amounts,
    weighted_income_estimate,
    aggregate_budgets_by_group,
    expand_group_budgets,
    budget_performance_snapshot,
    budget_trend_dataframe,
    monthly_income_vs_spend,
    monthly_group_targets_actuals,
    category_percent_income,
    percent_income_summary,
)

__all__ = [
    # Storage
    'BudgetStorage',
    'load_saved_budgets',
    'save_budget',
    'delete_budget',
    'get_budget_path',
    # Envelopes
    'normalize_envelope_config',
    'get_default_envelopes',
    'get_default_envelopes_with_all_categories',
    'ensure_savings_entry',
    'finalize_envelope_config',
    'groups_from_envelopes',
    'targets_from_envelopes',
    # Categorization
    'categorize_expense',
    'infer_group',
    # Calculations
    'suggested_amounts',
    'suggested_group_amounts',
    'weighted_income_estimate',
    'aggregate_budgets_by_group',
    'expand_group_budgets',
    'budget_performance_snapshot',
    'budget_trend_dataframe',
    'monthly_income_vs_spend',
    'monthly_group_targets_actuals',
    'category_percent_income',
    'percent_income_summary',
]

