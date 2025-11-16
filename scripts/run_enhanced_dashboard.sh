#!/usr/bin/env bash

# Enhanced Personal Finance Dashboard Launcher
# NOTE: This script is deprecated. Use run_enhanced_dashboard_module.sh or 'make run' instead.
# This script redirects to the module approach for consistency.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "⚠️  Note: This script is deprecated. Consider using:"
echo "   - bash scripts/run_enhanced_dashboard_module.sh (recommended)"
echo "   - make run"
echo ""

# Redirect to the module approach
exec "$SCRIPT_DIR/run_enhanced_dashboard_module.sh" "$@"
