import pandas as pd

from finance_dashboard.transfer_utils import annotate_transfers


def _sample_df():
    return pd.DataFrame([
        {
            'id': 1,
            'Amount': -500.00,
            'Category': 'Transfers',
            'Type': 'Payment',
            'Transaction Date': '2024-01-10',
            'profile_name': 'Checking',
        },
        {
            'id': 2,
            'Amount': 500.00,
            'Category': 'Transfers',
            'Type': 'Deposit',
            'Transaction Date': '2024-01-11',
            'profile_name': 'Savings',
        },
        {
            'id': 3,
            'Amount': -42.00,
            'Category': 'Transfers',
            'Type': 'Payment',
            'Transaction Date': '2024-01-12',
            'profile_name': 'Checking',
        },
    ])


def test_annotate_transfers_matches_pairs():
    result = annotate_transfers(_sample_df())
    transfers = result['transfers_df']

    assert result['detection_mode'] == 'pairing'
    assert len(transfers) == 3
    assert (transfers.loc[transfers['id'] == 1, 'Transfer Classification'].item()) == 'Likely Internal'
    assert (transfers.loc[transfers['id'] == 2, 'Transfer Classification'].item()) == 'Likely Internal'
    # unmatched entry left for review
    assert (transfers.loc[transfers['id'] == 3, 'Transfer Classification'].item()) == 'Needs Review'
