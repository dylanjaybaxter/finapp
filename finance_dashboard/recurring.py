"""Helpers for detecting recurring payments like subscriptions or rent."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict

import numpy as np
import pandas as pd

# Categories that should never be treated as recurring expenses
EXCLUDED_RECURRING_CATEGORIES = {'Dining', 'Groceries'}
NON_EXPENSE_CATEGORIES = {'Income', 'Transfer', 'Transfers'}

# Categories that almost always represent fixed, recurring costs
ESSENTIAL_RECURRING_CATEGORIES = {
    'Housing',
    'Utilities',
    'Mortgage',
    'Rent',
    'Insurance',
    'Phone',
    'Internet',
    'Student Loan',
    'Loans',
    'Loan Payment',
}

HOUSING_CATEGORIES = {'Housing', 'Rent', 'Mortgage', 'Rent/Mortgage'}
SUBSCRIPTION_CATEGORIES = {'Subscription', 'Subscriptions', 'Streaming', 'Entertainment', 'Software'}
UTILITY_CATEGORIES = {'Utilities', 'Internet', 'Phone', 'Mobile', 'Insurance', 'Electric', 'Water'}
STUDENT_LOAN_CATEGORIES = {'Student Loan', 'Loans', 'Loan Payment', 'Education'}

SUBSCRIPTION_KEYWORDS = {
    'SUBSCRIPTION', 'MEMBERSHIP', 'PLAN', 'BILLING', 'BILLPAY', 'SERVICE', 'STREAMING',
    'NETFLIX', 'HULU', 'DISNEY', 'HBO', 'MAX', 'SPOTIFY', 'APPLE', 'GOOGLE', 'YOUTUBE',
    'ADOBE', 'DROPBOX', 'PRIME', 'SIRIUS', 'XFINITY', 'PEACOCK', 'NYTIMES', 'WSJ', 'GITHUB',
    'PELOTON', 'GYM', 'STRAVA'
}
RENT_KEYWORDS = {
    'RENT', 'APT', 'APAR', 'APTS', 'APARTMENTS', 'APARTMENT', 'LEASE', 'MORTGAGE', 'LOAN SERV',
    'PROPERTY', 'REALTY', 'RESIDENCE', 'RESIDENT', 'ENCLAVE', 'PMTS', 'PMT', 'RPS'
}
STUDENT_LOAN_KEYWORDS = {
    'STUDENT LOAN', 'LOAN SERV', 'LOAN SERVICE', 'SERVICING', 'MOHELA', 'NELNET', 'NAVIENT',
    'AIDVANTAGE', 'GREAT LAKES', 'STUDENTLN', 'EDUCATION LOAN'
}
UTILITY_KEYWORDS = {
    'UTILITY', 'UTIL', 'POWER', 'ENERGY', 'ELECTRIC', 'GAS', 'WATER', 'SEWER', 'WASTE',
    'TRASH', 'COMCAST', 'XFINITY', 'SPECTRUM', 'AT&T', 'ATT', 'VERIZON', 'T-MOBILE', 'TMOBILE',
    'INSURANCE', 'GEICO', 'STATE FARM', 'USAA'
}

ESSENTIAL_GROUP_KEY = '__ESSENTIAL__'
MONTHLY_FREQUENCIES = {'Monthly', 'Semi-Monthly', 'Biweekly'}
PRIORITY_RECURRING_TYPES = {'Rent/Mortgage', 'Utility', 'Subscription', 'Student Loan', 'Essential'}

FREQUENCY_WINDOWS: Dict[str, Dict[str, float]] = {
    'Weekly': {'min': 5, 'max': 9, 'std': 2.5, 'monthly_multiplier': 52 / 12},
    'Biweekly': {'min': 10, 'max': 20, 'std': 4.0, 'monthly_multiplier': 26 / 12},
    'Semi-Monthly': {'min': 13, 'max': 17, 'std': 3.5, 'monthly_multiplier': 2.0},
    'Monthly': {'min': 25, 'max': 40, 'std': 6.0, 'monthly_multiplier': 1.0},
    'Quarterly': {'min': 80, 'max': 110, 'std': 12.0, 'monthly_multiplier': 1 / 3},
    'Annual': {'min': 330, 'max': 400, 'std': 20.0, 'monthly_multiplier': 1 / 12},
}


@dataclass
class FrequencyStats:
    avg_days: float
    median_days: float
    std_days: float
    samples: int


def normalize_payee(value: Any) -> str:
    """Normalize description text so recurring detection groups similar payees."""
    if not isinstance(value, str):
        return 'UNKNOWN'
    text = value.upper()
    text = text.replace('*', ' ')
    text = re.sub(
        r"\b(POS|PPD|WEB|ACH|CCD|ATM|SQ|SP|CARDMEMBER SERV|PAYPAL|VENMO|DEBIT CARD|CREDIT CARD|ACH CREDIT|ACH DEBIT|RPS|PMT|PMTS|PYMT|PAYMENT|PAYMENTS)\b\s*",
        '',
        text,
    )
    text = re.sub(r'^RPS', '', text)
    text = re.sub(r'\s+ON\s+\d{6}.*$', '', text)
    text = re.sub(r'\s+#?\d{2,}$', '', text)
    text = re.sub(r'\b(APARTMENTS?|APTS?|APT|APAR|APA|RESIDENCE|RESIDENT)\b', '', text)
    text = re.sub(r'\b(CD|CO|LLC|INC)\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text or 'UNKNOWN'


def detect_recurring(df: pd.DataFrame, *, return_details: bool = False):
    """Identify recurring outgoing payments such as rent or subscriptions."""

    def _empty_result():
        empty = pd.DataFrame()
        return (empty, empty) if return_details else empty

    if df is None or df.empty or 'Description' not in df.columns:
        return _empty_result()
    working = df.copy()
    working['Payee'] = working['Description'].apply(normalize_payee)

    if 'Transaction Date' not in working.columns:
        return _empty_result()
    working['Transaction Date'] = pd.to_datetime(working['Transaction Date'], errors='coerce')
    working = working.dropna(subset=['Transaction Date'])
    if working.empty:
        return _empty_result()

    if 'Amount' not in working.columns:
        return _empty_result()
    working['Amount'] = pd.to_numeric(working['Amount'], errors='coerce')
    working = working.dropna(subset=['Amount'])
    working = working[working['Amount'] < 0]
    if working.empty:
        return _empty_result()

    if 'Category' not in working.columns:
        working['Category'] = 'Uncategorized'
    working['Category'] = working['Category'].fillna('Uncategorized')

    mask_non_expense = working['Category'].isin(NON_EXPENSE_CATEGORIES)
    working = working[~mask_non_expense]
    if working.empty:
        return _empty_result()
    mask_excluded = working['Category'].isin(EXCLUDED_RECURRING_CATEGORIES)
    mask_essential = working['Category'].isin(ESSENTIAL_RECURRING_CATEGORIES)
    working = working[~mask_excluded | mask_essential]
    if working.empty:
        return _empty_result()

    if 'profile_name' not in working.columns:
        working['profile_name'] = 'Unknown'
    working['Amount Bucket'] = working['Amount'].apply(_amount_bucket)
    working['Group Bucket'] = working.apply(_group_bucket, axis=1)
    grouped = working.sort_values('Transaction Date').groupby(['Payee', 'Group Bucket'])

    detail_cols = [
        col for col in [
            'id',
            'Description',
            'Transaction Date',
            'Amount',
            'Category',
            'Payee',
            'Amount Bucket',
            'Group Bucket',
            'profile_name',
        ]
        if col in working.columns
    ]
    detail_df = working[detail_cols].copy()

    summary_rows = [_summarize_group(name, group) for name, group in grouped]
    usage = pd.DataFrame([row for row in summary_rows if row['Occurrences'] >= 2])
    if usage.empty:
        return (usage, detail_df) if return_details else usage

    usage['Is Consistent'] = usage.apply(_is_consistent, axis=1)
    usage['Date Range'] = usage['First'].dt.date.astype(str) + ' → ' + usage['Last'].dt.date.astype(str)
    usage['Amount Range'] = usage['Amount Range Min'].astype(str) + ' → ' + usage['Amount Range Max'].astype(str)
    usage['Recurring Key'] = usage.apply(
        lambda row: f"{row['Payee']} | ${abs(row['Average Amount']):,.2f}",
        axis=1,
    )
    usage['Is Recurring Candidate'] = usage.apply(_is_recurring_candidate, axis=1)
    usage = usage[usage['Is Recurring Candidate']]
    if usage.empty:
        return (usage, detail_df) if return_details else usage
    usage = usage.drop(columns=['Is Recurring Candidate'])
    usage = usage.drop(columns=['First', 'Last'])

    display_cols = [
        'Recurring Key', 'Payee', 'Profile', 'Category', 'Recurring Type',
        'Occurrences', 'Active Months', 'Avg Frequency (days)', 'Median Frequency (days)',
        'Frequency Std (days)', 'Frequency Label', 'Average Amount', 'Amount Bucket',
        'Amount Range Min', 'Amount Range Max', 'Amount Range', 'Monthly Estimate',
        'Date Range', 'Is Consistent', 'Group Key'
    ]
    usage = usage[display_cols]
    usage = _apply_confidence_scores(usage)
    usage = usage.sort_values(['Confidence Score', 'Occurrences', 'Monthly Estimate'], ascending=[False, False, False])
    result = usage.reset_index(drop=True)

    if return_details:
        return result, detail_df
    return result


def monthly_plan_candidates(recurring_df: pd.DataFrame) -> pd.DataFrame:
    """Return recurring entries that behave like monthly obligations."""
    if recurring_df is None or recurring_df.empty:
        return recurring_df

    df = recurring_df.copy()
    if 'Frequency Label' not in df.columns or 'Recurring Type' not in df.columns:
        return df

    monthly_mask = df['Frequency Label'].isin(MONTHLY_FREQUENCIES)
    priority_mask = df['Recurring Type'].isin(PRIORITY_RECURRING_TYPES)
    subset = df[monthly_mask | priority_mask]

    return subset if not subset.empty else df


def calculate_confidence_scores(recurring_df: pd.DataFrame) -> pd.DataFrame:
    """Public helper for tests to add confidence scores to a recurring DataFrame."""
    return _apply_confidence_scores(recurring_df.copy()) if not recurring_df.empty else recurring_df


def _summarize_group(name: Any, group: pd.DataFrame) -> Dict[str, Any]:
    payee, bucket_key = name
    dates = group['Transaction Date']
    first = dates.min()
    last = dates.max()
    stats = _frequency_stats(dates)
    avg_amount = group['Amount'].mean()
    min_amount = group['Amount'].min()
    max_amount = group['Amount'].max()
    category = _mode(group['Category'], 'Uncategorized')
    profile = _mode(group['profile_name'], 'Unknown')
    recurring_type = _classify_recurring_type(category, payee, avg_amount)
    monthly_estimate = _monthly_estimate(avg_amount, stats)
    if isinstance(bucket_key, (int, float)):
        bucket_value = bucket_key
    else:
        bucket_value = round(float(group['Amount Bucket'].median()), 2)
        if bucket_key == ESSENTIAL_GROUP_KEY:
            bucket_value = 'ESSENTIAL'

    return {
        'Payee': payee,
        'Amount Bucket': bucket_value,
        'Group Key': bucket_key,
        'Occurrences': len(group),
        'Active Months': group['Transaction Date'].dt.to_period('M').nunique(),
        'First': first,
        'Last': last,
        'Category': category,
        'Profile': profile,
        'Recurring Type': recurring_type,
        'Avg Frequency (days)': stats.avg_days,
        'Median Frequency (days)': stats.median_days,
        'Frequency Std (days)': stats.std_days,
        'Frequency Label': stats_label(stats),
        'Average Amount': round(float(avg_amount), 2),
        'Amount Range Min': round(float(min_amount), 2),
        'Amount Range Max': round(float(max_amount), 2),
        'Monthly Estimate': monthly_estimate,
    }


def _amount_bucket(amount: float) -> float:
    if pd.isna(amount):
        return 0.0
    absolute = abs(amount)
    tolerance = max(1.0, min(75.0, round(absolute * 0.02, 2)))
    bucket = round(amount / tolerance) * tolerance
    return round(bucket, 2)


def _group_bucket(row: pd.Series) -> Any:
    if _is_essential_row(row):
        return ESSENTIAL_GROUP_KEY
    return row['Amount Bucket']


def _frequency_stats(dates: pd.Series) -> FrequencyStats:
    ordered = dates.sort_values().dropna()
    if ordered.empty:
        return FrequencyStats(0.0, 0.0, 0.0, 0)
    diffs = ordered.diff().dt.days.dropna()
    if diffs.empty:
        return FrequencyStats(0.0, 0.0, 0.0, 0)
    avg = float(diffs.mean()) if not diffs.empty else 0.0
    median = float(diffs.median()) if not diffs.empty else avg
    std = float(diffs.std(ddof=0)) if len(diffs) > 1 else 0.0
    if pd.isna(std):
        std = 0.0
    return FrequencyStats(round(avg, 1), round(median, 1), round(std, 1), len(diffs))


def stats_label(stats: FrequencyStats) -> str:
    return _label_frequency(stats.median_days or stats.avg_days)


def _label_frequency(gap: float) -> str:
    if gap <= 0:
        return 'Irregular'
    for label, window in FREQUENCY_WINDOWS.items():
        if window['min'] <= gap <= window['max']:
            return label
    return 'Irregular'


def _classify_recurring_type(category: str, payee: str, avg_amount: float) -> str:
    category_upper = (category or '').strip().title()
    payee_upper = (payee or '').upper()
    absolute = abs(avg_amount)

    if category_upper in STUDENT_LOAN_CATEGORIES or _looks_like_student_loan(payee_upper):
        return 'Student Loan'
    if category_upper in HOUSING_CATEGORIES or _looks_like_rent(payee_upper):
        return 'Rent/Mortgage'
    if category_upper in UTILITY_CATEGORIES or any(keyword in payee_upper for keyword in UTILITY_KEYWORDS):
        return 'Utility'
    if (category_upper in SUBSCRIPTION_CATEGORIES or any(keyword in payee_upper for keyword in SUBSCRIPTION_KEYWORDS)):
        return 'Subscription'
    if category_upper in ESSENTIAL_RECURRING_CATEGORIES or absolute >= 500:
        return 'Essential'
    return 'Recurring Expense'


def _monthly_estimate(avg_amount: float, stats: FrequencyStats) -> float:
    amount = abs(avg_amount)
    label = stats_label(stats)
    window = FREQUENCY_WINDOWS.get(label)
    if window:
        return round(amount * window['monthly_multiplier'], 2)
    if stats.avg_days > 0:
        return round(amount * (30.0 / stats.avg_days), 2)
    return round(amount, 2)


def _mode(series: pd.Series, default: str) -> str:
    clean = series.dropna()
    if clean.empty:
        return default
    return str(clean.mode().iloc[0])


def _is_consistent(row: pd.Series) -> bool:
    occurrences = row['Occurrences']
    label = row['Frequency Label']
    std = row['Frequency Std (days)']
    active_months = row['Active Months']
    recurring_type = row['Recurring Type']

    if occurrences >= 4 and label in FREQUENCY_WINDOWS:
        allowed = FREQUENCY_WINDOWS[label]['std'] + 2
        return std <= allowed or active_months >= 4
    if occurrences >= 3 and label in FREQUENCY_WINDOWS:
        allowed = FREQUENCY_WINDOWS[label]['std']
        return std <= allowed or active_months >= 3
    if occurrences >= 2:
        if recurring_type == 'Rent/Mortgage' and label == 'Monthly':
            return True
        if recurring_type == 'Student Loan' and label in {'Monthly', 'Semi-Monthly', 'Biweekly'}:
            return True
        if recurring_type == 'Subscription' and label in {'Monthly', 'Quarterly', 'Annual'} and active_months >= 2:
            return True
        if recurring_type == 'Utility' and label in {'Monthly', 'Biweekly', 'Semi-Monthly'} and active_months >= 2:
            return True
    return False


def _is_essential_row(row: pd.Series) -> bool:
    category = str(row.get('Category', '')).title()
    payee = str(row.get('Payee', '')).upper()
    if category in ESSENTIAL_RECURRING_CATEGORIES:
        return True
    if _looks_like_rent(payee):
        return True
    if _looks_like_student_loan(payee):
        return True
    return False


def _looks_like_rent(payee_upper: str) -> bool:
    return any(keyword in payee_upper for keyword in RENT_KEYWORDS)


def _looks_like_student_loan(payee_upper: str) -> bool:
    return any(keyword in payee_upper for keyword in STUDENT_LOAN_KEYWORDS)


def _is_recurring_candidate(row: pd.Series) -> bool:
    if row['Is Consistent']:
        return True
    occ = row['Occurrences']
    active_months = row['Active Months']
    label = row['Frequency Label']
    rtype = row['Recurring Type']

    if rtype == 'Rent/Mortgage':
        return occ >= 2 and active_months >= 2
    if rtype == 'Student Loan':
        return occ >= 2 and label in {'Monthly', 'Semi-Monthly', 'Biweekly', 'Quarterly'}
    if rtype == 'Utility':
        return (occ >= 3 or active_months >= 3) and label != 'Irregular'
    if rtype == 'Subscription':
        return occ >= 3 and label != 'Irregular'
    if rtype == 'Essential':
        return occ >= 3 and label != 'Irregular'
    return False


def _apply_confidence_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    scored = df.copy()
    freq = scored['Avg Frequency (days)'].fillna(30.0)
    freq_score = 1 - (abs(freq - 30) / 30).clip(0, 1)

    occ_score = (scored['Occurrences'].clip(0, 12) / 6).clip(0, 1)
    months_score = (scored['Active Months'].clip(0, 12) / 6).clip(0, 1)
    std_score = 1 - (scored['Frequency Std (days)'].fillna(0).abs() / 20).clip(0, 1)

    type_weights = {
        'Rent/Mortgage': 1.0,
        'Student Loan': 0.95,
        'Utility': 0.9,
        'Subscription': 0.85,
        'Essential': 0.8,
    }
    type_score = scored['Recurring Type'].map(type_weights).fillna(0.6)

    amount_range = (scored['Amount Range Max'].abs() - scored['Amount Range Min'].abs()).abs()
    avg_amount = scored['Average Amount'].abs().replace(0, np.nan)
    variance_ratio = (amount_range / avg_amount).fillna(0)
    variance_score = 1 - variance_ratio.clip(0, 1)

    consistent_bonus = scored['Is Consistent'].astype(float) * 0.15

    score = (
        occ_score * 0.2
        + months_score * 0.15
        + freq_score * 0.2
        + std_score * 0.15
        + type_score * 0.2
        + variance_score * 0.1
        + consistent_bonus
    )
    scored['Confidence Score'] = (score.clip(0, 1) * 100).round(1)
    scored['Confidence Label'] = pd.cut(
        scored['Confidence Score'],
        bins=[-1, 50, 70, 85, 101],
        labels=['Low', 'Medium', 'High', 'Very High']
    ).astype(str)
    return scored
