#!/usr/bin/env python3
"""Direct launcher for the Enhanced Personal Finance Dashboard.

This script can be run directly to launch the enhanced dashboard
without import issues.
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Change to the finance_dashboard directory
os.chdir(os.path.join(current_dir, 'finance_dashboard'))

# Import and run the enhanced dashboard
from enhanced_dashboard import main

if __name__ == "__main__":
    main()



