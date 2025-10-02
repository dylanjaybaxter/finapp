#!/usr/bin/env bash

# A helper script to launch the Finance Dashboard in a reproducible
# environment.  This script creates a Python virtual environment in
# ``.venv`` if one does not already exist, installs the required
# dependencies from ``requirements.txt``, and starts the Streamlit
# server.  You can pass additional arguments after the script to
# forward them to ``streamlit run``, e.g. specifying a custom port.

set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

cd "$PROJECT_ROOT"

if [[ ! -d .venv ]]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
fi

# Activate the environment
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null

echo "Starting Streamlit server..."
# Forward any additional arguments to streamlit run
streamlit run finance_dashboard/dashboard.py "$@"