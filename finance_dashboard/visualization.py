"""Plotly visualisation helpers for the Finance Dashboard.

This module defines a suite of functions that accept data objects
returned by the corresponding functions in :mod:`data_processing` and
produce interactive Plotly figures.  Each function is focused on a
specific chart type to keep the code easy to navigate.  Future
analyses can be integrated by adding a new function here that
transforms a DataFrame, Series or dictionary into a Plotly figure.

All functions return a `plotly.graph_objects.Figure` instance or a
dictionary representation that Streamlit can render via
``st.plotly_chart``.  See the individual docstrings for details on
parameters and behaviour.

Imports are kept local to minimise the initial import time when
serving the Streamlit app.  If a function requires a heavy import
(e.g. seaborn or matplotlib), consider importing it inside the
function body.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_line_chart(aggregated: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Generate a line chart for a time‑series aggregation.

    Parameters
    ----------
    aggregated : pandas.DataFrame
        A DataFrame indexed by a date/time period with a single
        numeric column.
    title : str, optional
        Chart title.  If ``None``, a default title is derived from
        the column name.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive line chart.
    """
    if aggregated.empty:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig
    numeric_col = aggregated.columns[0]
    df = aggregated.reset_index().rename(columns={aggregated.index.name: "Period"})
    fig = px.line(df, x="Period", y=numeric_col)
    fig.update_layout(
        title=title or f"{numeric_col} over time",
        xaxis_title="Period",
        yaxis_title=numeric_col,
    )
    return fig


def create_cumulative_sum_chart(aggregated: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Create a cumulative sum line chart from an aggregated time‑series.

    Parameters
    ----------
    aggregated : pandas.DataFrame
        DataFrame indexed by period with a single numeric column.
    title : str, optional
        Title for the chart.

    Returns
    -------
    plotly.graph_objects.Figure
        Cumulative sum line chart.
    """
    if aggregated.empty:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig
    numeric_col = aggregated.columns[0]
    cumulative = aggregated.cumsum()
    df = cumulative.reset_index().rename(columns={cumulative.index.name: "Period"})
    fig = px.line(df, x="Period", y=numeric_col)
    fig.update_layout(
        title=title or f"Cumulative {numeric_col}",
        xaxis_title="Period",
        yaxis_title=f"Cumulative {numeric_col}",
    )
    return fig


def create_category_bar_chart(series: pd.Series, title: str | None = None) -> go.Figure:
    """Generate a bar chart showing the breakdown of a numeric column by category.

    Parameters
    ----------
    series : pandas.Series
        Series indexed by category with summed values.
    title : str, optional
        Title for the chart.

    Returns
    -------
    plotly.graph_objects.Figure
        Bar chart of categories vs values.
    """
    if series.empty:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig
    df = series.reset_index()
    df.columns = ["Category", "Value"]
    fig = px.bar(df, x="Category", y="Value")
    fig.update_layout(
        title=title or f"Breakdown by {df['Category'].name}",
        xaxis_title="Category",
        yaxis_title="Value",
    )
    return fig


def create_category_pie_chart(series: pd.Series, title: str | None = None) -> go.Figure:
    """Generate a pie chart showing the breakdown of a numeric column by category.

    Parameters
    ----------
    series : pandas.Series
        Series indexed by category with summed values.
    title : str, optional
        Title for the chart.

    Returns
    -------
    plotly.graph_objects.Figure
        Pie chart.
    """
    if series.empty:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig
    df = series.reset_index()
    df.columns = ["Category", "Value"]
    fig = px.pie(df, names="Category", values="Value")
    fig.update_layout(title=title or "Category breakdown")
    return fig


def create_correlation_heatmap(df: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Create a correlation heatmap for numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing only numeric columns.
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
        Heatmap of the correlation matrix.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig
    corr = df.corr()
    fig = px.imshow(
        corr,
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig.update_layout(title=title or "Correlation heatmap")
    return fig


def create_histogram(df: pd.DataFrame, numeric_col: str, title: str | None = None) -> go.Figure:
    """Generate a histogram for a numeric column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing the numeric column.
    numeric_col : str
        Name of the numeric column to plot.
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
        Histogram chart.
    """
    if numeric_col not in df.columns or df[numeric_col].dropna().empty:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig
    fig = px.histogram(df, x=numeric_col, nbins=30)
    fig.update_layout(
        title=title or f"Distribution of {numeric_col}",
        xaxis_title=numeric_col,
        yaxis_title="Count",
    )
    return fig


def create_ratios_bar_chart(ratios: Dict[str, float], title: str | None = None) -> go.Figure:
    """Render financial ratios as a bar chart.

    Parameters
    ----------
    ratios : dict
        Dictionary of ratio names to their computed values.
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
        Bar chart of ratios.
    """
    if not ratios:
        fig = go.Figure()
        fig.update_layout(title="No ratios to display")
        return fig
    df = pd.DataFrame(list(ratios.items()), columns=["Ratio", "Value"])
    fig = px.bar(df, x="Ratio", y="Value")
    fig.update_layout(
        title=title or "Financial ratios",
        xaxis_title="Ratio",
        yaxis_title="Value",
    )
    return fig


def create_horizontal_analysis_chart(ha_df: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Visualise horizontal (period‑over‑period) analysis.

    The input DataFrame should be indexed by period and have one or
    more numeric columns representing percentage change.  Each series
    is plotted as a separate line.  If the DataFrame contains NaN for
    the first period (as is typical for ``pct_change``), those values
    are ignored.

    Parameters
    ----------
    ha_df : pandas.DataFrame
        DataFrame of percentage changes with periods on the index.
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
        Multi‑line chart of percentage changes.
    """
    if ha_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig
    df = ha_df.reset_index().rename(columns={ha_df.index.name or "index": "Period"})
    df = df.dropna(how="all", subset=ha_df.columns)
    # Melt to long format for easier plotting
    long_df = df.melt(id_vars="Period", value_vars=ha_df.columns, var_name="Metric", value_name="Change")
    fig = px.line(long_df, x="Period", y="Change", color="Metric")
    fig.update_layout(
        title=title or "Horizontal analysis",
        xaxis_title="Period",
        yaxis_title="Percentage change",
    )
    return fig


def create_vertical_analysis_chart(va_df: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Visualise vertical (common size) analysis as a bar chart.

    Given a DataFrame of proportions (rows sum to 1), the function
    computes the mean proportion for each column and displays a bar
    chart.  This provides a high‑level view of the relative weight of
    each line item.

    Parameters
    ----------
    va_df : pandas.DataFrame
        DataFrame where each value is a proportion of the row total.
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
        Bar chart showing average proportions.
    """
    if va_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig
    means = va_df.mean(axis=0)
    df = pd.DataFrame({"Metric": means.index, "Proportion": means.values})
    fig = px.bar(df, x="Metric", y="Proportion")
    fig.update_layout(
        title=title or "Vertical analysis (average proportions)",
        xaxis_title="Metric",
        yaxis_title="Average proportion",
    )
    return fig


def create_dupont_bar_chart(dupont: Dict[str, float], title: str | None = None) -> go.Figure:
    """Render the components of the DuPont analysis as a bar chart.

    Parameters
    ----------
    dupont : dict
        Dictionary with keys 'Net Profit Margin', 'Asset Turnover',
        'Financial Leverage' and 'Return on Equity'.
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
        Bar chart of DuPont components.
    """
    if not dupont:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig
    df = pd.DataFrame(list(dupont.items()), columns=["Component", "Value"])
    fig = px.bar(df, x="Component", y="Value")
    fig.update_layout(
        title=title or "DuPont decomposition",
        xaxis_title="Component",
        yaxis_title="Value",
    )
    return fig
