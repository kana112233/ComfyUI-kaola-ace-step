.PHONY: help install install-acestep test clean format lint

help:
	@echo "ACE-Step ComfyUI Nodes - Available Commands:"
	@echo ""
	@echo "  make install         - Install Python dependencies"
	@echo "  make install-acestep - Install ACE-Step and download models"
	@echo "  make clean           - Clean temporary files"
	@echo "  make format          - Format Python code with black"
	@echo "  make lint            - Run flake8 linter"
	@echo "  make test            - Run tests (if available)"
	@echo ""

install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

install-acestep:
	@echo "Installing ACE-Step..."
	@if [ ! -d "acestep_repo" ]; then \
		git clone https://github.com/ace-step/ACE-Step-1.5.git acestep_repo; \
		cd acestep_repo && uv sync && uv run acestep-download; \
	else \
		echo "ACE-Step already installed"; \
	fi

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true

format:
	@echo "Formatting Python code..."
	@command -v black >/dev/null 2>&1 || { echo "black not installed. Run: pip install black"; exit 1; }
	black nodes.py __init__.py

lint:
	@echo "Running linter..."
	@command -v flake8 >/dev/null 2>&1 || { echo "flake8 not installed. Run: pip install flake8"; exit 1; }
	flake8 nodes.py __init__.py --max-line-length=100

test:
	@echo "Running tests..."
	@echo "No tests configured yet"
