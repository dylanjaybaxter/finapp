import pandas as pd

from finance_dashboard.plan_storage import load_plan, save_plan
from finance_dashboard.recurring import (
    calculate_confidence_scores,
    detect_recurring,
    monthly_plan_candidates,
    MONTHLY_FREQUENCIES,
)


def _build_df(rows):
    return pd.DataFrame(rows)


def test_detects_monthly_subscription():
    df = _build_df([
        {'Description': 'Netflix.com', 'Transaction Date': '2024-01-12', 'Amount': -15.49, 'Category': 'Subscription'},
        {'Description': 'Netflix.com', 'Transaction Date': '2024-02-11', 'Amount': -15.49, 'Category': 'Subscription'},
        {'Description': 'Netflix.com', 'Transaction Date': '2024-03-12', 'Amount': -15.99, 'Category': 'Subscription'},
        {'Description': 'Netflix.com', 'Transaction Date': '2024-04-10', 'Amount': -15.99, 'Category': 'Subscription'},
    ])

    recurring = detect_recurring(df)
    netflix = recurring[recurring['Payee'] == 'NETFLIX.COM']

    assert not netflix.empty, "expected netflix charges to be detected"
    row = netflix.iloc[0]
    assert row['Frequency Label'] == 'Monthly'
    assert row['Recurring Type'] == 'Subscription'
    assert row['Is Consistent']
    assert 10 <= row['Monthly Estimate'] <= 20
    assert 'Confidence Score' in recurring.columns


def test_detect_recurring_can_return_transaction_details():
    df = _build_df([
        {'Description': 'Spotify USA', 'Transaction Date': '2024-01-01', 'Amount': -10.00, 'Category': 'Subscription'},
        {'Description': 'Spotify USA', 'Transaction Date': '2024-02-01', 'Amount': -10.00, 'Category': 'Subscription'},
    ])

    summary, details = detect_recurring(df, return_details=True)

    assert not summary.empty
    assert not details.empty
    assert set(details['Payee']) == {'SPOTIFY USA'}
    assert 'Confidence Score' in summary.columns


def test_rent_with_two_payments_is_considered_recurring():
    df = _build_df([
        {'Description': 'Main Street Apartments', 'Transaction Date': '2024-01-01', 'Amount': -1850.00, 'Category': 'Housing'},
        {'Description': 'Main Street Apartments', 'Transaction Date': '2024-02-01', 'Amount': -1850.00, 'Category': 'Housing'},
    ])

    recurring = detect_recurring(df)
    rent = recurring[recurring['Payee'].str.contains('MAIN STREET', na=False)]

    assert not rent.empty, "Rent should be detected even with limited history"
    row = rent.iloc[0]
    assert row['Recurring Type'] == 'Rent/Mortgage'
    assert row['Is Consistent']
    assert row['Frequency Label'] == 'Monthly'


def test_enclave_rent_with_variable_amounts_is_grouped():
    df = _build_df([
        {'Description': 'WEB  The Enclave Apar WEB PMTS', 'Transaction Date': '2025-04-10', 'Amount': -1493.50, 'Category': 'Housing'},
        {'Description': 'WEB  The Enclave Apar WEB PMTS', 'Transaction Date': '2025-05-09', 'Amount': -2745.00, 'Category': 'Housing'},
        {'Description': 'WEB  The Enclave Apar WEB PMTS', 'Transaction Date': '2025-06-10', 'Amount': -2844.23, 'Category': 'Housing'},
        {'Description': 'WEB  The Enclave Apar WEB PMTS', 'Transaction Date': '2025-07-08', 'Amount': -2911.02, 'Category': 'Housing'},
    ])

    recurring = detect_recurring(df)
    enclave = recurring[recurring['Payee'].str.contains('ENCLAVE', na=False)]

    assert not enclave.empty, "Variable rent payments should still group together"
    row = enclave.iloc[0]
    assert row['Recurring Type'] == 'Rent/Mortgage'
    assert row['Is Consistent']
    assert row['Frequency Label'] == 'Monthly'


def test_enclave_variations_collapse_to_single_payee():
    df = _build_df([
        {'Description': 'RPS*The Enclave Apa CD', 'Transaction Date': '2025-01-03', 'Amount': -2000.00, 'Category': 'Housing'},
        {'Description': 'WEB  The Enclave Apar WEB PMTS', 'Transaction Date': '2025-02-03', 'Amount': -2050.00, 'Category': 'Housing'},
        {'Description': 'The Enclave Apartments', 'Transaction Date': '2025-03-03', 'Amount': -2075.00, 'Category': 'Housing'},
        {'Description': 'RPS*The Enclave Apa CD', 'Transaction Date': '2025-04-03', 'Amount': -2100.00, 'Category': 'Housing'},
    ])

    recurring = detect_recurring(df)
    enclave = recurring[recurring['Payee'].str.contains('ENCLAVE', na=False)]

    assert len(enclave) == 1, "All Enclave variations should collapse into one recurring group"
    assert enclave.iloc[0]['Occurrences'] == 4


def test_student_loan_is_detected_with_two_payments():
    df = _build_df([
        {'Description': 'MOHELA STUDENT LOAN SERV', 'Transaction Date': '2024-01-15', 'Amount': -320.25, 'Category': 'Student Loan'},
        {'Description': 'Mohela Student Loan Serv', 'Transaction Date': '2024-02-15', 'Amount': -320.25, 'Category': 'Student Loan'},
    ])

    recurring = detect_recurring(df)
    loans = recurring[recurring['Recurring Type'] == 'Student Loan']

    assert not loans.empty, "Student loans should be surfaced even with limited history"
    row = loans.iloc[0]
    assert row['Frequency Label'] == 'Monthly'
    assert row['Is Consistent']


def test_utilities_have_reasonable_frequency():
    df = _build_df([
        {'Description': 'Verizon Wireless', 'Transaction Date': '2024-01-05', 'Amount': -95.00, 'Category': 'Utilities'},
        {'Description': 'Verizon Wireless', 'Transaction Date': '2024-02-06', 'Amount': -95.00, 'Category': 'Utilities'},
        {'Description': 'Verizon Wireless', 'Transaction Date': '2024-03-07', 'Amount': -97.50, 'Category': 'Utilities'},
    ])

    recurring = detect_recurring(df)
    verizon = recurring[recurring['Payee'] == 'VERIZON WIRELESS']

    assert not verizon.empty
    assert verizon.iloc[0]['Recurring Type'] == 'Utility'
    assert verizon.iloc[0]['Frequency Label'] == 'Monthly'


def test_irregular_charges_not_marked_consistent():
    df = _build_df([
        {'Description': 'Special Project Services', 'Transaction Date': '2024-01-03', 'Amount': -220.00, 'Category': 'Business'},
        {'Description': 'Special Project Services', 'Transaction Date': '2024-01-21', 'Amount': -260.00, 'Category': 'Business'},
        {'Description': 'Special Project Services', 'Transaction Date': '2024-03-15', 'Amount': -205.00, 'Category': 'Business'},
    ])

    recurring = detect_recurring(df)
    assert recurring.empty, "Irregular project work should be filtered out from recurring results"


def test_confidence_scores_rank_recurring_entries():
    df = _build_df([
        {'Description': 'Rent Payment', 'Transaction Date': '2024-01-01', 'Amount': -1500.00, 'Category': 'Housing'},
        {'Description': 'Rent Payment', 'Transaction Date': '2024-02-01', 'Amount': -1500.00, 'Category': 'Housing'},
        {'Description': 'Electric Company', 'Transaction Date': '2024-01-15', 'Amount': -120.00, 'Category': 'Utilities'},
        {'Description': 'Electric Company', 'Transaction Date': '2024-02-15', 'Amount': -118.00, 'Category': 'Utilities'},
        {'Description': 'Electric Company', 'Transaction Date': '2024-03-15', 'Amount': -119.00, 'Category': 'Utilities'},
        {'Description': 'Electric Company', 'Transaction Date': '2024-04-15', 'Amount': -121.00, 'Category': 'Utilities'},
    ])
    recurring = detect_recurring(df)
    scored = calculate_confidence_scores(recurring)
    assert 'Confidence Score' in scored.columns
    rent_score = scored.loc[scored['Payee'] == 'RENT', 'Confidence Score'].iloc[0]
    electric_score = scored.loc[scored['Payee'] == 'ELECTRIC COMPANY', 'Confidence Score'].iloc[0]
    assert electric_score >= rent_score
    assert rent_score >= 80


def test_plan_storage_roundtrip(tmp_path):
    target = tmp_path / 'plan.json'
    save_plan(['A', 'B'], {'A': {'avg': 10.0, 'min': 5.0, 'max': 15.0}}, target)
    data = load_plan(target)
    assert data['selection'] == ['A', 'B']
    assert data['overrides']['A']['avg'] == 10.0


def test_monthly_plan_candidates_focus_on_monthly_obligations():
    rent_rows = [
        {'Description': 'WEB  The Enclave Apar WEB PMTS', 'Transaction Date': '2025-04-10', 'Amount': -2000.00, 'Category': 'Housing'},
        {'Description': 'WEB  The Enclave Apar WEB PMTS', 'Transaction Date': '2025-05-09', 'Amount': -2000.00, 'Category': 'Housing'},
        {'Description': 'WEB  The Enclave Apar WEB PMTS', 'Transaction Date': '2025-06-10', 'Amount': -2000.00, 'Category': 'Housing'},
    ]
    gym_rows = [
        {'Description': 'Local Gym Membership', 'Transaction Date': '2025-04-04', 'Amount': -55.00, 'Category': 'Subscription'},
        {'Description': 'Local Gym Membership', 'Transaction Date': '2025-05-04', 'Amount': -55.00, 'Category': 'Subscription'},
        {'Description': 'Local Gym Membership', 'Transaction Date': '2025-06-04', 'Amount': -55.00, 'Category': 'Subscription'},
    ]
    irregular_rows = [
        {'Description': 'One-Off Consulting', 'Transaction Date': '2025-04-15', 'Amount': -400.00, 'Category': 'Business'},
        {'Description': 'One-Off Consulting', 'Transaction Date': '2025-07-15', 'Amount': -400.00, 'Category': 'Business'},
    ]
    df = _build_df(rent_rows + gym_rows + irregular_rows)

    recurring = detect_recurring(df)
    plan = monthly_plan_candidates(recurring)

    assert 'THE ENCLAVE | $2,000.00' in plan['Recurring Key'].values
    assert 'LOCAL GYM MEMBERSHIP | $55.00' in plan['Recurring Key'].values
    assert plan.apply(
        lambda row: row['Frequency Label'] in MONTHLY_FREQUENCIES or row['Recurring Type'] in {'Rent/Mortgage', 'Utility', 'Subscription', 'Student Loan', 'Essential'},
        axis=1
    ).all()


def test_monthly_plan_includes_priority_types_even_if_not_monthly():
    df = _build_df([
        {'Description': 'WEB  The Enclave Apar WEB PMTS', 'Transaction Date': '2025-04-10', 'Amount': -2000.00, 'Category': 'Housing'},
        {'Description': 'WEB  The Enclave Apar WEB PMTS', 'Transaction Date': '2025-05-01', 'Amount': -2000.00, 'Category': 'Housing'},
        {'Description': 'WEB  The Enclave Apar WEB PMTS', 'Transaction Date': '2025-06-20', 'Amount': -2000.00, 'Category': 'Housing'},
        {'Description': 'One-Off Consulting', 'Transaction Date': '2025-04-15', 'Amount': -400.00, 'Category': 'Business'},
        {'Description': 'One-Off Consulting', 'Transaction Date': '2025-05-19', 'Amount': -500.00, 'Category': 'Business'},
    ])

    recurring = detect_recurring(df)
    plan = monthly_plan_candidates(recurring)

    assert any(plan['Recurring Type'] == 'Rent/Mortgage')
    assert 'THE ENCLAVE | $2,000.00' in plan['Recurring Key'].values
