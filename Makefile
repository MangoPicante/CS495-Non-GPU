PYTHON     := py -3.13
VENV_DIR   := .venv
VENV_BIN   := $(VENV_DIR)/Scripts
ACTIVATE   := . $(VENV_BIN)/activate

.DEFAULT_GOAL := help

.PHONY: help venv install install-dev update test lint format type-check check clean nuke run benchmark plots

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  venv         Create .venv with Python 3.13"
	@echo "  install      Install production dependencies via Poetry into .venv"
	@echo "  install-dev  Install all dependencies (including dev) via Poetry into .venv"
	@echo "  update       Update dependencies to latest allowed versions"
	@echo "  test         Run pytest"
	@echo "  lint         Run ruff linter"
	@echo "  format       Auto-format with ruff"
	@echo "  type-check   Run mypy"
	@echo "  check        lint + type-check + test"
	@echo "  clean        Remove __pycache__, .pytest_cache, *.pyc"
	@echo "  nuke         Remove .venv and clean"
	@echo "  run          Run the package entry point"
	@echo "  benchmark    Run inference benchmarks (set BINARY= and MODEL=)"
	@echo "  plots        Generate comparison plots from results/"

# ── Environment ────────────────────────────────────────────────────────────────

$(VENV_DIR):
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_BIN)/python -m pip install --upgrade pip

venv: $(VENV_DIR)

# Tell Poetry to use the .venv we created (in-project venv)
install: $(VENV_DIR)
	poetry config virtualenvs.in-project true
	poetry env use $(VENV_DIR)/Scripts/python.exe
	poetry install --only main

install-dev: $(VENV_DIR)
	poetry config virtualenvs.in-project true
	poetry env use $(VENV_DIR)/Scripts/python.exe
	poetry install

update: $(VENV_DIR)
	poetry update

# ── Quality ────────────────────────────────────────────────────────────────────

test:
	poetry run pytest -v

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

type-check:
	poetry run mypy .

check: lint type-check test

# ── Housekeeping ───────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache  -exec rm -rf {} +
	find . -type d -name .ruff_cache  -exec rm -rf {} +
	find . -name "*.pyc" -delete

nuke: clean
	rm -rf $(VENV_DIR)

# ── Run ────────────────────────────────────────────────────────────────────────

run:
	poetry run python -m cs495_non_gpu

# ── Benchmarks ─────────────────────────────────────────────────────────────────

BINARY ?= build/bin/llama-cli
MODEL  ?= models/BitNet-b1.58-2B4T/ggml-model-i2_s.gguf
THREADS ?= 4
RUNS ?= 5

benchmark:
	poetry run python scripts/metrics_tracker.py \
		--binary $(BINARY) --model $(MODEL) \
		--threads $(THREADS) --runs $(RUNS)

plots:
	poetry run python scripts/compare_runs.py
