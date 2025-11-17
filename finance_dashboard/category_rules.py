"""Category Rules Management - Keyword-based categorization rules.

This module handles user-defined rules for automatically categorizing transactions
based on keywords in their descriptions. Rules can be associated with specific profiles
or applied globally.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
import re
import pandas as pd

try:
    from . import config
    from .profile_manager import get_registry
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import config
    from profile_manager import get_registry


@dataclass
class CategoryRule:
    """A rule for categorizing transactions based on keywords."""
    keyword: str
    category: str
    profiles: List[str]  # List of profile names, empty means all profiles
    case_sensitive: bool = False
    whole_word: bool = False  # If True, match whole words only
    enabled: bool = True
    start_date: Optional[str] = None  # YYYY-MM-DD format, None means no start limit
    end_date: Optional[str] = None  # YYYY-MM-DD format, None means no end limit


GLOBAL_PROFILE_KEY = "__all_profiles__"


class CategoryRulesManager:
    """Manages category rules for automatic transaction categorization."""
    
    def __init__(self, rules_file: Optional[Path] = None):
        """Initialize the rules manager.
        
        Args:
            rules_file: Path to JSON file storing rules. If None, uses default location.
        """
        if rules_file is None:
            rules_file = config.DATA_DIR / "category_rules.json"
        self.rules_file = Path(rules_file)
        self.rules: List[CategoryRule] = []
        self._load_rules()
    
    def _load_rules(self) -> None:
        """Load rules from JSON file."""
        if not self.rules_file.exists():
            self.rules = []
            return
        
        try:
            with self.rules_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            stored_rules = data.get('rules', [])
            if isinstance(stored_rules, dict):
                self.rules = self._deserialize_compact_rules(stored_rules)
            else:  # legacy list format
                self.rules = [CategoryRule(**rule_data) for rule_data in stored_rules]
                self._save_rules()  # rewrite in compact format
        except Exception as e:
            print(f"Error loading category rules: {e}")
            self.rules = []
    
    def _save_rules(self) -> None:
        """Save rules to JSON file."""
        self.rules_file.parent.mkdir(parents=True, exist_ok=True)

        mapping: Dict[str, Dict[str, List[Dict[str, object]]]] = {}

        for rule in sorted(self.rules, key=lambda r: (r.category.lower(), r.keyword.lower())):
            category_bucket = mapping.setdefault(rule.category, {})
            profile_targets = rule.profiles or [None]
            for profile in profile_targets:
                profile_key = profile if profile else GLOBAL_PROFILE_KEY
                entries = category_bucket.setdefault(profile_key, [])

                if not rule.case_sensitive and not rule.whole_word and rule.enabled and not rule.start_date and not rule.end_date:
                    entries.append(rule.keyword)
                else:
                    rule_dict = {
                        'keyword': rule.keyword,
                        'case_sensitive': rule.case_sensitive,
                        'whole_word': rule.whole_word,
                        'enabled': rule.enabled,
                    }
                    if rule.start_date:
                        rule_dict['start_date'] = rule.start_date
                    if rule.end_date:
                        rule_dict['end_date'] = rule.end_date
                    entries.append(rule_dict)

        with self.rules_file.open('w', encoding='utf-8') as f:
            json.dump({'rules': mapping}, f, indent=2, ensure_ascii=False)

    def _deserialize_compact_rules(self, mapping: Dict[str, Dict[str, List]]) -> List[CategoryRule]:
        rules: List[CategoryRule] = []
        for category, profile_map in mapping.items():
            for profile_key, entries in profile_map.items():
                profile_list = [] if profile_key == GLOBAL_PROFILE_KEY else [profile_key]
                for entry in entries:
                    if isinstance(entry, str):
                        keyword = entry
                        case_sensitive = False
                        whole_word = False
                        enabled = True
                        start_date = None
                        end_date = None
                    else:
                        keyword = entry.get('keyword', '')
                        case_sensitive = entry.get('case_sensitive', False)
                        whole_word = entry.get('whole_word', False)
                        enabled = entry.get('enabled', True)
                        start_date = entry.get('start_date')
                        end_date = entry.get('end_date')

                    if not keyword:
                        continue
                    rules.append(
                        CategoryRule(
                            keyword=keyword,
                            category=category,
                            profiles=profile_list.copy(),
                            case_sensitive=case_sensitive,
                            whole_word=whole_word,
                            enabled=enabled,
                            start_date=start_date,
                            end_date=end_date
                        )
                    )
        return rules
    
    def add_rule(self, keyword: str, category: str, profiles: List[str] = None, 
                 case_sensitive: bool = False, whole_word: bool = False,
                 start_date: Optional[str] = None, end_date: Optional[str] = None) -> CategoryRule:
        """Add a new category rule.
        
        Args:
            keyword: Keyword to match in transaction descriptions
            category: Category to assign when keyword is found
            profiles: List of profile names to apply rule to. None/empty means all profiles.
            case_sensitive: Whether keyword matching is case-sensitive
            whole_word: Whether to match whole words only
            start_date: Start date for rule validity (YYYY-MM-DD), None means no start limit
            end_date: End date for rule validity (YYYY-MM-DD), None means no end limit
        
        Returns:
            The created CategoryRule
        """
        if profiles is None:
            profiles = []
        
        rule = CategoryRule(
            keyword=keyword,
            category=category,
            profiles=profiles,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
            enabled=True,
            start_date=start_date,
            end_date=end_date
        )
        
        self.rules.append(rule)
        self._save_rules()
        return rule
    
    def remove_rule(self, keyword: str, category: str) -> bool:
        """Remove a rule matching keyword and category.
        
        Returns:
            True if rule was removed, False if not found
        """
        original_count = len(self.rules)
        self.rules = [
            r for r in self.rules 
            if not (r.keyword == keyword and r.category == category)
        ]
        
        if len(self.rules) < original_count:
            self._save_rules()
            return True
        return False
    
    def update_rule(self, keyword: str, category: str, **kwargs) -> bool:
        """Update an existing rule.
        
        Args:
            keyword: Original keyword
            category: Original category
            **kwargs: Fields to update (keyword, category, profiles, case_sensitive, whole_word, enabled)
        
        Returns:
            True if rule was updated, False if not found
        """
        for rule in self.rules:
            if rule.keyword == keyword and rule.category == category:
                for key, value in kwargs.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                self._save_rules()
                return True
        return False
    
    def get_rules(self, profile_name: Optional[str] = None) -> List[CategoryRule]:
        """Get rules, optionally filtered by profile.
        
        Args:
            profile_name: If provided, only return rules that apply to this profile
                          (or rules that apply to all profiles)
        
        Returns:
            List of CategoryRule objects
        """
        if profile_name is None:
            return self.rules.copy()
        
        # Return rules that either:
        # 1. Apply to all profiles (empty profiles list)
        # 2. Apply to the specific profile
        return [
            r for r in self.rules 
            if r.enabled and (not r.profiles or profile_name in r.profiles)
        ]
    
    def apply_rules(self, description: str, profile_name: Optional[str] = None, track_history: bool = False, transaction_date: Optional[str] = None) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Apply rules to a transaction description and return category if matched.
        
        Args:
            description: Transaction description text
            profile_name: Profile name to filter rules by
            track_history: If True, return list of applied rules in order
            transaction_date: Transaction date in YYYY-MM-DD format for date range checking
        
        Returns:
            Tuple of (category_name, rule_history)
            - category_name: Category name if a rule matches, None otherwise
            - rule_history: List of dicts with rule info (keyword, category, order) if track_history=True, else []
        """
        if not description:
            return None, []
        
        rules = self.get_rules(profile_name)
        rule_history = []
        
        for order, rule in enumerate(rules, start=1):
            if not rule.enabled:
                continue
            
            # Check date range if transaction_date is provided
            if transaction_date and (rule.start_date or rule.end_date):
                try:
                    from datetime import datetime as dt
                    txn_dt = dt.strptime(transaction_date, '%Y-%m-%d').date()
                    
                    if rule.start_date:
                        start_dt = dt.strptime(rule.start_date, '%Y-%m-%d').date()
                        if txn_dt < start_dt:
                            continue
                    
                    if rule.end_date:
                        end_dt = dt.strptime(rule.end_date, '%Y-%m-%d').date()
                        if txn_dt > end_dt:
                            continue
                except (ValueError, AttributeError):
                    # If date parsing fails, skip date check (apply rule anyway)
                    pass
            
            # Build pattern
            keyword = rule.keyword
            if not rule.case_sensitive:
                keyword = keyword.lower()
                desc = description.lower()
            else:
                desc = description
            
            if rule.whole_word:
                # Match whole words only
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                # Match anywhere in string
                pattern = re.escape(keyword)
            
            if re.search(pattern, desc):
                if track_history:
                    rule_history.append({
                        'order': order,
                        'keyword': rule.keyword,
                        'category': rule.category,
                        'case_sensitive': rule.case_sensitive,
                        'whole_word': rule.whole_word,
                        'profiles': rule.profiles,
                        'start_date': rule.start_date,
                        'end_date': rule.end_date
                    })
                return rule.category, rule_history
        
        return None, rule_history
    
    def get_all_keywords(self) -> Set[str]:
        """Get all unique keywords from rules."""
        return {rule.keyword for rule in self.rules}
    
    def get_all_categories(self) -> Set[str]:
        """Get all unique categories from rules."""
        return {rule.category for rule in self.rules}


def apply_category_rules_to_transactions(profile_name: Optional[str] = None, 
                                        dry_run: bool = False,
                                        include_categorized: bool = True) -> Dict[str, int]:
    """Apply category rules to transactions in the database.
    
    Args:
        profile_name: Profile name to filter rules by. None means use all applicable rules.
        dry_run: If True, don't actually update transactions, just return counts
        include_categorized: If True, also apply rules to already-categorized transactions (default: True)
    
    Returns:
        Dict with 'updated', 'skipped', 'total_checked' counts
    """
    try:
        from . import db
    except ImportError:
        import db
    
    manager = CategoryRulesManager()
    
    # Fetch transactions - include categorized if requested
    if include_categorized:
        transactions = db.fetch_transactions()
    else:
        transactions = db.fetch_uncategorized_transactions()
    
    results = {
        'updated': 0,
        'skipped': 0,
        'total_checked': len(transactions)
    }
    
    if transactions.empty:
        return results
    
    # Get profile_name from transaction if available (from database)
    for _, transaction in transactions.iterrows():
        description = transaction.get('Description', '')
        if not description:
            results['skipped'] += 1
            continue
        
        # Get transaction date for date range checking
        transaction_date = None
        txn_date = transaction.get('Transaction Date')
        if pd.notna(txn_date):
            if hasattr(txn_date, 'strftime'):
                transaction_date = txn_date.strftime('%Y-%m-%d')
            elif isinstance(txn_date, str):
                transaction_date = txn_date
        
        # Use transaction's profile_name if available, otherwise use parameter
        txn_profile = transaction.get('profile_name') if 'profile_name' in transaction else None
        rule_profile = profile_name or txn_profile
        
        category, rule_history = manager.apply_rules(
            description, 
            rule_profile, 
            track_history=True,
            transaction_date=transaction_date
        )
        
        if category:
            transaction_id = transaction.get('id')
            if transaction_id and not dry_run:
                # Store rule history as JSON
                import json
                rule_history_json = json.dumps(rule_history) if rule_history else None
                db.update_transaction(transaction_id, category=category, rule_history=rule_history_json)
                results['updated'] += 1
            elif dry_run:
                results['updated'] += 1
        else:
            results['skipped'] += 1
    
    return results


def apply_rules_to_dataframe(df: pd.DataFrame, profile_name: Optional[str] = None) -> pd.DataFrame:
    """Apply category rules to a DataFrame of transactions.
    
    This is useful for applying rules during import before saving to database.
    
    Args:
        df: DataFrame with 'Description' and optionally 'Category' columns
        profile_name: Profile name to filter rules by
    
    Returns:
        DataFrame with updated categories
    """
    manager = CategoryRulesManager()
    df = df.copy()
    
    # Only apply rules to rows without categories
    if 'Category' not in df.columns:
        df['Category'] = None

    category_series = df['Category'].astype('string')
    normalized = category_series.fillna('').str.strip()
    mask = normalized.eq('') | normalized.str.lower().eq('uncategorized')

    for idx in df[mask].index:
        description = df.loc[idx, 'Description']
        if pd.notna(description) and description:
            # Get transaction date if available
            transaction_date = None
            if 'Transaction Date' in df.columns:
                txn_date = df.loc[idx, 'Transaction Date']
                if pd.notna(txn_date):
                    if hasattr(txn_date, 'strftime'):
                        transaction_date = txn_date.strftime('%Y-%m-%d')
                    elif isinstance(txn_date, str):
                        transaction_date = txn_date
            
            category, rule_history = manager.apply_rules(
                str(description), 
                profile_name, 
                track_history=True,
                transaction_date=transaction_date
            )
            if category:
                df.loc[idx, 'Category'] = category
                # Store rule history in dataframe for later database insertion
                import json
                df.loc[idx, 'rule_history'] = json.dumps(rule_history) if rule_history else None
    
    return df


def get_rules_hash(rules_file: Optional[Path] = None) -> str:
    """Return a stable hash of the stored rules for sync tracking."""
    path = rules_file or config.DATA_DIR / "category_rules.json"
    if not Path(path).exists():
        return ''
    try:
        data = Path(path).read_bytes()
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return ''
