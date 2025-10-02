"""Streamlit app for the Finance Dashboard.

This module defines the user interface and orchestrates the data
processing and visualisation functions.  It uses the Streamlit API
to build a modern, interactive web application that can be run
locally or deployed to Streamlit Cloud.  The dashboard is designed
to be modular: new analyses can be integrated by adding sidebar
controls and corresponding visualisation calls.  Extensive comments
throughout the file indicate where to hook in additional features.

To run the dashboard from the command line::

    streamlit run finance_dashboard/dashboard.py

Alternatively, execute the ``scripts/run_dashboard.sh`` shell script
to automatically set up a virtual environment and launch the app.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import pandas as pd
import streamlit as st

# Conditional imports to support execution both as part of a package
# (e.g. via ``python -m finance_dashboard.dashboard`` or
# ``streamlit run finance_dashboard/dashboard.py``) and directly as a
# standalone script (e.g. ``python finance_dashboard/dashboard.py``).
if __package__:
    # When executed as part of a package, use relative imports
    from . import data_processing as dp
    from . import visualization as viz
else:
    # When executed directly, adjust sys.path to include the parent
    # directory so that absolute imports resolve correctly.  This
    # branch allows ``python finance_dashboard/dashboard.py`` to work
    # without raising ``ImportError: attempted relative import``.
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    from finance_dashboard import data_processing as dp  # type: ignore
    from finance_dashboard import visualization as viz  # type: ignore


def load_data(file) -> pd.DataFrame:
    """Load uploaded file using the helper from data_processing.

    Streamlit file objects expose a ``name`` attribute which is
    inspected by :func:`dp.read_file` to determine whether the
    contents are CSV or Excel.  If reading fails, an error message
    is displayed and an empty DataFrame is returned.
    """
    try:
        return dp.read_file(file)
    except Exception as exc:  # pragma: no cover - UI display only
        st.error(f"Failed to read file: {exc}")
        return pd.DataFrame()


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Finance Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Finance Dashboard")
    st.markdown(
        "Upload a CSV or Excel file containing your financial data. Use the sidebar to configure the analyses."
    )

    # Sidebar: file upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file", type=["csv", "xls", "xlsx"], accept_multiple_files=False
    )

    # If no file is provided, show instructions and exit early
    if uploaded_file is None:
        st.info("Please upload a file to begin.")
        st.stop()

    # Load data
    data = load_data(uploaded_file)
    if data.empty:
        st.warning("The uploaded file contains no data.")
        st.stop()

    # Detect column types
    detected_date = dp.detect_date_column(data)
    numeric_cols = dp.detect_numeric_columns(data)
    category_cols = dp.detect_category_columns(data)

    # Sidebar: Column selections
    st.sidebar.header("Configuration")
    date_col = st.sidebar.selectbox(
        "Date column (optional)", options=["None"] + list(data.columns), index=(list(data.columns).index(detected_date) + 1) if detected_date in data.columns else 0
    )
    # Convert "None" selection to None
    date_col = None if date_col == "None" else date_col

    # Sidebar: choose numeric columns for analyses
    default_numeric_selection: List[str] = numeric_cols[:1] if numeric_cols else []
    numeric_selection = st.sidebar.multiselect(
        "Numeric columns to analyse", options=numeric_cols, default=default_numeric_selection
    )

    # Sidebar: choose a categorical column for breakdowns
    category_col: Optional[str] = None
    if category_cols:
        category_col = st.sidebar.selectbox(
            "Categorical column for breakdowns", options=["None"] + category_cols, index=0
        )
        category_col = None if category_col == "None" else category_col

    # Sidebar: choose aggregation frequency
    freq_map = {"Daily": "D", "Monthly": "M", "Yearly": "A"}
    freq_label = st.sidebar.selectbox("Aggregation frequency", options=list(freq_map.keys()), index=1)
    freq = freq_map[freq_label]

    # Sidebar: toggles for optional analyses
    show_describe = st.sidebar.checkbox("Show descriptive statistics", value=True)
    show_cumsum = st.sidebar.checkbox("Show cumulative sum", value=False)
    show_category = st.sidebar.checkbox("Show category breakdown", value=bool(category_col))
    show_correlation = st.sidebar.checkbox("Show correlation heatmap", value=False)
    show_histogram = st.sidebar.checkbox("Show histogram", value=False)
    show_ratios = st.sidebar.checkbox("Show financial ratios", value=False)
    show_horizontal = st.sidebar.checkbox("Show horizontal analysis", value=False)
    show_vertical = st.sidebar.checkbox("Show vertical analysis", value=False)
    show_dupont = st.sidebar.checkbox("Show DuPont analysis", value=False)

    # Main area: preview table
    st.subheader("Data preview")
    st.dataframe(data.head(100))

    # Descriptive statistics
    if show_describe and numeric_selection:
        st.subheader("Descriptive statistics")
        desc = dp.describe_numeric(data[numeric_selection])
        st.dataframe(desc)

    # Time‑series and cumulative sum
    if date_col and numeric_selection:
        for num_col in numeric_selection:
            # Aggregate by period
            aggregated = dp.aggregate_by_period(data, date_col, num_col, freq)
            st.subheader(f"{num_col} over time ({freq_label.lower()})")
            st.plotly_chart(viz.create_line_chart(aggregated))
            if show_cumsum:
                st.subheader(f"Cumulative {num_col}")
                st.plotly_chart(viz.create_cumulative_sum_chart(aggregated))
    elif show_cumsum and not date_col:
        st.info("Select a date column to enable time‑series and cumulative analyses.")

    # Category breakdown
    if show_category and category_col and numeric_selection:
        st.subheader("Category breakdown")
        # Let user choose numeric column for breakdown
        col_for_category = st.selectbox(
            "Numeric column for breakdown", options=numeric_selection, index=0
        )
        cat_series = dp.aggregate_by_category(data, category_col, col_for_category)
        chart_type = st.radio("Chart type", options=["Bar", "Pie"], horizontal=True)
        if chart_type == "Bar":
            st.plotly_chart(viz.create_category_bar_chart(cat_series))
        else:
            st.plotly_chart(viz.create_category_pie_chart(cat_series))

    # Correlation heatmap
    if show_correlation and numeric_selection:
        st.subheader("Correlation heatmap")
        corr_df = data[numeric_selection].select_dtypes(include="number")
        st.plotly_chart(viz.create_correlation_heatmap(corr_df))

    # Histogram
    if show_histogram and numeric_selection:
        st.subheader("Distribution histogram")
        hist_col = st.selectbox(
            "Select numeric column for histogram", options=numeric_selection, index=0
        )
        st.plotly_chart(viz.create_histogram(data, hist_col))

    # Financial ratios
    if show_ratios:
        st.subheader("Financial ratios")
        ratios = dp.compute_financial_ratios(data)
        if ratios:
            st.plotly_chart(viz.create_ratios_bar_chart(ratios))
            # Display ratio values in a table as well
            st.table(pd.DataFrame(list(ratios.items()), columns=["Ratio", "Value"]))
        else:
            st.info("No ratios could be computed. Ensure your data contains standard financial statement line items.")

    # Horizontal analysis
    if show_horizontal:
        if date_col and len(numeric_selection) >= 1:
            st.subheader("Horizontal (period‑over‑period) analysis")
            ha_df = dp.horizontal_analysis(data, date_col, numeric_selection, freq)
            st.plotly_chart(viz.create_horizontal_analysis_chart(ha_df))
            # Show underlying DataFrame for reference
            st.dataframe(ha_df)
        else:
            st.info("Select a date column and at least one numeric column for horizontal analysis.")

    # Vertical analysis
    if show_vertical:
        if len(numeric_selection) >= 2:
            st.subheader("Vertical (common size) analysis")
            va_df = dp.vertical_analysis(data, numeric_selection)
            st.plotly_chart(viz.create_vertical_analysis_chart(va_df))
            st.dataframe(va_df)
        else:
            st.info("Select at least two numeric columns for vertical analysis.")

    # DuPont analysis
    if show_dupont:
        st.subheader("DuPont analysis")
        dupont = dp.dupont_analysis(data)
        if dupont:
            st.plotly_chart(viz.create_dupont_bar_chart(dupont))
            st.table(pd.DataFrame(list(dupont.items()), columns=["Component", "Value"]))
        else:
            st.info("Unable to compute DuPont components. Ensure the dataset contains net income, revenue, assets and equity.")


if __name__ == "__main__":  # pragma: no cover
    # When executed as a script via `python -m finance_dashboard.dashboard`
    # or `streamlit run finance_dashboard/dashboard.py`, call main().
    main()
