"""Top‑level package for the Finance Dashboard.

This file makes the directory a Python package and exposes
convenient names.  The primary modules are:

* ``data_processing`` – functions for reading and analysing financial data
* ``visualization`` – functions that generate Plotly figures
* ``dashboard`` – a Streamlit app that ties everything together

To run the dashboard from the command line you can execute:

```bash
streamlit run finance_dashboard/dashboard.py
```

Alternatively, use the provided shell script in ``scripts/`` which
handles virtual environment creation and dependency installation.
"""

# Export submodules to simplify import paths.  These imports are
# intentionally placed at the bottom of the file to avoid circular
# dependencies during package initialisation.

from . import data_processing  # noqa: F401  # re-exported for convenience
from . import visualization  # noqa: F401  # re-exported for convenience
# Import dashboard lazily.  Streamlit may not be installed in all
# environments (e.g. during unit testing).  If the import fails,
# assign ``None`` and document the missing dependency in the error
# message.  See README for installation instructions.
try:
    from . import dashboard  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    dashboard = None  # type: ignore


__all__ = ["data_processing", "visualization", "dashboard"]