#!/usr/bin/env python3
"""Direct launcher for the Enhanced Personal Finance Dashboard.

This script launches Streamlit with the finance_dashboard directory as the app root,
enabling automatic page discovery from the pages/ subdirectory.
"""

import sys
import subprocess
import os
from pathlib import Path

# Get the project root and finance_dashboard directory
project_root = Path(__file__).parent.resolve()
finance_dashboard_dir = project_root / "finance_dashboard"

# Launch Streamlit with finance_dashboard as the working directory
# This enables automatic page discovery from finance_dashboard/pages/
if __name__ == "__main__":
    # Change to finance_dashboard directory so Streamlit can discover pages/
    os.chdir(finance_dashboard_dir)
    # Add project root to path for imports
    sys.path.insert(0, str(project_root))
    # Run Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "Home.py"
    ])



