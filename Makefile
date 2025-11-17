.PHONY: help install run run-enhanced run-dashboard test clean lint format type-check setup-dirs

# Default target
help:
	@echo "Available targets:"
	@echo "  make install        - Create venv and install dependencies"
	@echo "  make run            - Run enhanced dashboard (recommended)"
	@echo "  make run-enhanced   - Run enhanced dashboard"
	@echo "  make run-dashboard  - Run original dashboard"
	@echo "  make test           - Run test suite"
	@echo "  make clean          - Remove cache files and __pycache__"
	@echo "  make lint           - Run linters (flake8, pylint)"
	@echo "  make format         - Format code with black and isort"
	@echo "  make type-check     - Run type checking with mypy"
	@echo "  make setup-dirs     - Create required data directories"

# Virtual environment setup
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

install:
	@echo "📦 Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "📥 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Installation complete!"

# Run targets
run: run-enhanced

run-enhanced:
	@if [ ! -d "$(VENV)" ]; then \
		echo "⚠️  Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "🚀 Starting Enhanced Personal Finance Dashboard..."
	$(PYTHON) -m streamlit run finance_dashboard/enhanced_dashboard.py

run-dashboard:
	@if [ ! -d "$(VENV)" ]; then \
		echo "⚠️  Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "🚀 Starting Finance Dashboard..."
	$(PYTHON) -m streamlit run finance_dashboard/dashboard.py

# Testing
test:
	@if [ ! -d "$(VENV)" ]; then \
		echo "⚠️  Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	$(PYTHON) -m pytest tests/ -v

# Code quality
lint:
	@if [ ! -d "$(VENV)" ]; then \
		echo "⚠️  Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "🔍 Running linters..."
	$(PYTHON) -m flake8 finance_dashboard/ tests/ --max-line-length=100 --ignore=E501,W503 || true
	@echo "✅ Linting complete"

format:
	@if [ ! -d "$(VENV)" ]; then \
		echo "⚠️  Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "🎨 Formatting code..."
	$(PYTHON) -m black finance_dashboard/ tests/ scripts/*.py
	$(PYTHON) -m isort finance_dashboard/ tests/ scripts/*.py
	@echo "✅ Formatting complete"

type-check:
	@if [ ! -d "$(VENV)" ]; then \
		echo "⚠️  Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "🔍 Running type checker..."
	$(PYTHON) -m mypy finance_dashboard/ --ignore-missing-imports || true
	@echo "✅ Type checking complete"

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Setup directories
setup-dirs:
	@echo "📁 Creating data directories..."
	mkdir -p data/raw data/budgets data/goals data/reports
	@echo "✅ Directories created"

