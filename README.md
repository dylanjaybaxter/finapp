# Finance Dashboard

## Overview

The **Finance Dashboard** is an extensible web application for
uploading, analysing and visualising tabular financial data.  It
builds upon the core analytics from the previous version of the
project and wraps them in a modern, interactive user interface built
with [Streamlit](https://streamlit.io) and [Plotly](https://plotly.com).

This project is organised as a Python package (`finance_dashboard`) to
encourage modularity and ease of maintenance.  Core functionality is
divided into separate modules: one for data ingestion and analysis,
another for generating Plotly figures, and a `dashboard.py` script that
defines the Streamlit app.  Tests are provided using the `pytest`
framework to ensure the analytical functions behave as expected.

## Features

* **File upload widget** – Users can drag‑and‑drop or browse for CSV
  and Excel files directly in the browser.  Uploaded files are kept
  in memory; no data is persisted on disk.
* **Data type detection** – The app automatically identifies date,
  numeric and categorical columns.  Users can override these choices
  via sidebar controls if the heuristics mis‑identify a column.
* **Interactive charts** – All visualisations are generated using
  Plotly and rendered inside Streamlit.  Charts update
  automatically when users select different columns or filters.  The
  following chart types are available:

  - **Preview table** with sorting and pagination.
  - **Descriptive statistics** showing count, mean, standard deviation
    and percentiles for numeric columns.
  - **Time‑series line charts** for any numeric column grouped by
    day, month or year.
  - **Cumulative sums** to track running totals over time.
  - **Category breakdowns** as bar or pie charts for categorical
    variables.
  - **Correlation heatmaps** for multiple numeric columns.
  - **Distribution histograms** for individual numeric columns.
  - **Financial ratios** visualised as bar charts (liquidity,
    profitability, leverage and efficiency ratios).
  - **Horizontal and vertical analyses** presented as bar or pie
    charts, following the definitions used in professional financial
    analysis【327922635543314†L462-L584】.
  - **DuPont decomposition** of Return on Equity (ROE) into net
    margin, asset turnover and financial leverage.

* **Sidebar configuration** – Users can choose which numeric and
  categorical columns to analyse, select the frequency for
  time‑series aggregation (daily, monthly, yearly) and toggle
  optional analyses on or off.  This makes it easy to explore large
  datasets without modifying the code.
* **Extensible architecture** – The codebase is heavily commented and
  annotated with type hints.  New analyses can be added by defining
  pure functions in `data_processing.py`, creating Plotly figures in
  `visualization.py`, and wiring them into the Streamlit layout in
  `dashboard.py`.  Comments in each module point to the relevant
  extension points.
* **Tests included** – Unit tests located under `tests/` verify the
  correctness of core analytical functions such as date detection,
  category detection and ratio computation.  Running `pytest` will
  execute all tests.

## Getting started

1. **Clone or unzip** the repository.
2. **Create a virtual environment** and install dependencies.  The
   provided `scripts/run_dashboard.sh` automates this process:

   ```bash
   cd finance_dashboard_pro
   bash scripts/run_dashboard.sh
   ```

   The script creates a `.venv` directory, installs the packages listed
   in `requirements.txt` and launches the Streamlit server.  By default
   the server runs on `localhost:8501`.
3. **Open your browser** and navigate to `http://localhost:8501` to
   interact with the dashboard.  Upload a CSV or Excel file and
   explore the available analyses.  Use the sidebar to change
   parameters such as aggregation frequency or which columns to plot.

## Repository layout

```
finance_dashboard_pro/
├── finance_dashboard/          # Python package with modular code
│   ├── __init__.py             # Package initialisation
│   ├── data_processing.py      # Pure functions for reading and analysing data
│   ├── visualization.py        # Functions returning Plotly figures
│   └── dashboard.py            # Streamlit app definition
├── scripts/
│   └── run_dashboard.sh        # Environment setup and app launcher
├── tests/
│   └── test_data_processing.py # Unit tests for core analytics
├── sample_data/
│   └── sample_finance.csv      # Example dataset for manual testing
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT license
└── README.md                   # Project documentation (this file)
```

## Extending the dashboard

### Adding a new analysis

1. **Write a data function** in `finance_dashboard/data_processing.py`.  It
   should accept a `pandas.DataFrame` and return either a DataFrame,
   Series or scalar value.  Keep it pure: avoid side effects and do not
   mutate global state.
2. **Create a visualisation function** in
   `finance_dashboard/visualization.py`.  It should accept the output
   of your data function and return a Plotly figure (either a `dict`
   representation or a `plotly.graph_objects.Figure` instance).  See
   existing functions for examples.
3. **Update the Streamlit app** in `finance_dashboard/dashboard.py`.
   Decide where the analysis should appear (e.g., main area, sidebar,
   or an expandable section) and invoke your data and visualisation
   functions accordingly.  Use Streamlit components such as
   `st.selectbox`, `st.multiselect` or `st.expander` to allow users to
   customise the analysis.
4. **Write tests** for your function under `tests/` if applicable.

The modular design and liberal use of type hints should make the code
base approachable for future contributions.  Consult the extensive
docstrings and inline comments for further guidance.

## License

This project is licensed under the MIT License.  See `LICENSE` for
details.