#!/usr/bin/env bash

# Enhanced Personal Finance Dashboard Launcher (Module Approach)
# This script uses the module approach to avoid import issues

set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

cd "$PROJECT_ROOT"

echo "🚀 Starting Enhanced Personal Finance Dashboard..."
echo "=================================================="

# Create virtual environment if it doesn't exist
if [[ ! -d .venv ]]; then
    echo "📦 Creating virtual environment in .venv..."
    python3 -m venv .venv
fi

# Activate the environment
source .venv/bin/activate

echo "📥 Installing dependencies..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null

echo "✅ Dependencies installed successfully!"

# Create data directory for user data
mkdir -p data
mkdir -p data/budgets
mkdir -p data/goals
mkdir -p data/reports

echo "📊 Starting Enhanced Personal Finance Dashboard..."
echo "🌐 Dashboard will be available at: http://localhost:8501"
echo "📱 Mobile-optimized interface included"
echo "💰 Features: Budgeting, Goal Tracking, Spending Analysis, Financial Insights"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Launch the enhanced dashboard using module approach
python -m streamlit run finance_dashboard/enhanced_dashboard.py "$@"



