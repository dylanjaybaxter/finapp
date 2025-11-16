#!/usr/bin/env python3
"""Direct launcher for the Enhanced Personal Finance Dashboard.

This script can be run directly to launch the enhanced dashboard
without import issues.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# Import and run the enhanced dashboard
from finance_dashboard.enhanced_dashboard import main

if __name__ == "__main__":
    main()



