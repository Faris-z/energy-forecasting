# ─────────────────────────────────────────────────────────────────────────────
#  Energy Forecasting — Makefile
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: all data train test lint format clean help

PYTHON       := python
PIP          := pip
CONFIG       := config/config.yaml
SRC_DIRS     := src tests train.py prepare_data.py

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Environment ───────────────────────────────────────────────────────────────
install:       ## Install all Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ── Data ──────────────────────────────────────────────────────────────────────
data:          ## Download and preprocess the UCI dataset
	$(PYTHON) prepare_data.py

# ── Training ──────────────────────────────────────────────────────────────────
train:         ## Train all models and generate reports
	$(PYTHON) train.py

# ── Testing ───────────────────────────────────────────────────────────────────
test:          ## Run the full pytest test suite with coverage
	$(PYTHON) -m pytest tests/ -v \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:reports/coverage \
		-p no:warnings

# ── Code quality ──────────────────────────────────────────────────────────────
lint:          ## Run flake8 linter on all source files
	$(PYTHON) -m flake8 $(SRC_DIRS) \
		--max-line-length=99 \
		--extend-ignore=E203,W503

format:        ## Auto-format code with black and isort
	$(PYTHON) -m black $(SRC_DIRS) --line-length 99
	$(PYTHON) -m isort $(SRC_DIRS) --profile black

format-check:  ## Verify formatting without modifying files (CI use)
	$(PYTHON) -m black $(SRC_DIRS) --line-length 99 --check
	$(PYTHON) -m isort $(SRC_DIRS) --profile black --check-only

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:         ## Remove all generated artefacts (data, models, reports, logs)
	rm -rf data/processed/
	rm -rf models/*.json models/*.pkl models/lstm/
	rm -rf reports/figures/
	rm -rf reports/coverage/
	rm -rf logs/
	rm -rf mlruns/
	rm -rf .pytest_cache/ .mypy_cache/ __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

clean-data:    ## Remove only raw downloaded data (re-download on next make data)
	rm -rf data/raw/

# ── Full pipeline ─────────────────────────────────────────────────────────────
all: data train ## Run the complete pipeline: download data then train models
