PYTHON     := py -3.13
VENV_DIR   := .venv
VENV_BIN   := $(VENV_DIR)/Scripts
ACTIVATE   := . $(VENV_BIN)/activate

.DEFAULT_GOAL := help

.PHONY: help venv install install-dev update test lint format type-check check clean nuke run benchmark plots

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Project tooling:"
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
	@echo "  benchmark    Run inference benchmark (set BITNET_DIR=, MODEL=, THREADS=)"
	@echo "  plots        Generate comparison plots from results/"
	@echo ""
	@echo "BitNet environment setup:"
	@echo "  bitnet-setup       Full pipeline: clone + pin commit + submodules + deps + build + model"
	@echo "  bitnet-clone       git clone microsoft/BitNet into \$$(BITNET_DIR)"
	@echo "  bitnet-checkout    Pin repo to commit \$$(BITNET_COMMIT)"
	@echo "  bitnet-submodules  git submodule update --init --recursive"
	@echo "  bitnet-deps        pip install requirements + gguf-py"
	@echo "  bitnet-patch       Apply ClangCL compatibility patches (const + chrono + converter)"
	@echo "  bitnet-build       Generate TL2 kernels (2B-4T), cmake configure (ClangCL), cmake build"
	@echo "  bitnet-model       Download safetensors, convert to f32 GGUF, quantize to i2_s"
	@echo "  bitnet-verify      Quick inference smoke-test (requires completed setup)"
	@echo "  bitnet-clean       Delete \$$(BITNET_DIR)"
	@echo ""
	@echo "  Override BITNET_DIR to target a different path, e.g.:"
	@echo "    make bitnet-setup BITNET_DIR=../BitNet2"

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

MODEL   ?= $(BITNET_DIR)/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
THREADS ?= 4

benchmark:
	python scripts/metrics_tracker.py \
		--bitnet-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS)

plots:
	python scripts/compare_runs.py

# ── BitNet environment setup ───────────────────────────────────────────────────
#
# Reproducibly clones and builds bitnet.cpp at the exact commit used in this
# project's benchmarks. Run `make bitnet-setup` to perform the full pipeline,
# or run individual targets to repeat specific steps.
#
# Requirements: git, Python 3.11+, CMake 3.22+,
#               Clang 18+ or Visual Studio 2022 with C++ workload
#
# Override destination:  make bitnet-setup BITNET_DIR=../MyBitNet

BITNET_DIR    ?= ../BitNet2
BITNET_COMMIT := 01eb415772c342d9f20dc42772f1583ae1e5b102
BITNET_MODEL  := models/BitNet-b1.58-2B-4T
BITNET_QUANT  := i2_s

.PHONY: bitnet-setup bitnet-clone bitnet-checkout bitnet-submodules bitnet-deps bitnet-patch bitnet-build bitnet-verify bitnet-clean

bitnet-setup: bitnet-clone bitnet-checkout bitnet-submodules bitnet-deps bitnet-patch bitnet-build bitnet-model
	@echo "BitNet setup complete in $(BITNET_DIR)"

bitnet-clone:
	git clone https://github.com/microsoft/BitNet.git $(BITNET_DIR)

bitnet-checkout:
	cd $(BITNET_DIR) && git checkout $(BITNET_COMMIT)

bitnet-submodules:
	cd $(BITNET_DIR) && git submodule update --init --recursive

bitnet-deps:
	cd $(BITNET_DIR) && pip install -r requirements.txt
	cd $(BITNET_DIR) && pip install 3rdparty/llama.cpp/gguf-py

# Patches required for commit $(BITNET_COMMIT):
#   patches/bitnet-clangcl-const.patch       — adds const to y_col pointer (ggml-bitnet-mad.cpp); ClangCL on LLVM 18+
#   patches/llama-chrono.patch               — adds missing <chrono> include (common.cpp and log.cpp); ClangCL on LLVM 18+
#   patches/llama-chrono-examples.patch      — adds missing <chrono> include (imatrix.cpp and perplexity.cpp); ClangCL on LLVM 18+
#   patches/bitnet-converter-arch-name.patch — BitNetForCausalLM alias + vocab fallback + weight_scale unpack for 2B-4T
bitnet-patch:
	cd $(BITNET_DIR) && git apply $(CURDIR)/patches/bitnet-clangcl-const.patch
	cd $(BITNET_DIR)/3rdparty/llama.cpp && git apply $(CURDIR)/patches/llama-chrono.patch
	cd $(BITNET_DIR)/3rdparty/llama.cpp && git apply $(CURDIR)/patches/llama-chrono-examples.patch
	cd $(BITNET_DIR) && git apply $(CURDIR)/patches/bitnet-converter-arch-name.patch

bitnet-build:
	cd $(BITNET_DIR) && python utils/codegen_tl2.py \
		--model bitnet_b1_58-3B --BM 160,320,320 --BK 96,96,96 --bm 32,32,32
	cd $(BITNET_DIR) && cmake -B build -DBITNET_X86_TL2=ON \
		-T ClangCL
	cd $(BITNET_DIR) && cmake --build build --config Release

bitnet-model:
	cd $(BITNET_DIR) && hf download microsoft/BitNet-b1.58-2B-4T \
		--local-dir $(BITNET_MODEL)
	cd $(BITNET_DIR) && python utils/convert-hf-to-gguf-bitnet.py \
		$(BITNET_MODEL) --outtype f32
	cd $(BITNET_DIR) && build/bin/Release/llama-quantize \
		$(BITNET_MODEL)/ggml-model-f32.gguf \
		$(BITNET_MODEL)/ggml-model-$(BITNET_QUANT).gguf I2_S 1

bitnet-verify:
	cd $(BITNET_DIR) && python run_inference.py \
		-m $(BITNET_MODEL)/ggml-model-$(BITNET_QUANT).gguf \
		-p "What is 2+2?" \
		-n 32 \
		-t $(THREADS)

bitnet-clean:
	rm -rf $(BITNET_DIR)
