"""Main entry point for Streamlit multi-page app.

This file enables Streamlit's automatic page discovery.
Pages in the pages/ directory will automatically appear in the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and run the enhanced dashboard main function
try:
    from finance_dashboard.enhanced_dashboard import main
except ImportError:
    from enhanced_dashboard import main

if __name__ == "__main__":
    main()

