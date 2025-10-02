# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-09-24
### Added
- Enhanced personal finance dashboard (`finance_dashboard/enhanced_dashboard.py`).
- Personal finance UI components (`finance_dashboard/personal_finance_ui.py`).
- Personal finance analytics module (`finance_dashboard/personal_finance_analytics.py`).
- New run scripts: `scripts/run_enhanced_dashboard.sh`, `scripts/run_enhanced_dashboard_module.sh`, and `run_enhanced_dashboard.py`.
- Documentation: `README_ENHANCED.md`, `ENHANCEMENT_PLAN.md`, `IMPLEMENTATION_SUMMARY.md`, `IMPORT_FIX_SUMMARY.md`.
- Test harness `test_enhanced_dashboard.py`.
- Smoothing slider, category Clear All/Add All for spending trends.

### Changed
- `requirements.txt`: added scikit-learn, seaborn, matplotlib, scipy.
- Updated enhanced launcher script to use direct launcher.

### Fixed
- ImportError due to relative imports by adding absolute import fallbacks.
- Streamlit Period serialization error by casting Period to string.
- Sidebar filters now drive all overview sections.

## [0.1.0] - 2025-09-24
- Initial version with base `dashboard.py`, `data_processing.py`, `visualization.py`, tests, and run script.



