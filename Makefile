PYTHON     := py -3.11
POETRY     := poetry
THREADS    ?= 4
LIMIT      ?= 500

# vswhere.exe ships with every VS 2017+ install at this fixed location.
# It is used by check-deps to verify the ClangCL VS components are present.
VSWHERE := C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe

BITNET_DIR    ?= ../Models/BitNet
BITNET_COMMIT := 01eb415772c342d9f20dc42772f1583ae1e5b102  # HEAD as of May 2026; pinned for reproducibility
BITNET_MODEL  := models/BitNet-b1.58-2B-4T
BITNET_QUANT  := i2_s
MODEL         ?= $(BITNET_DIR)/$(BITNET_MODEL)/ggml-model-$(BITNET_QUANT).gguf

.DEFAULT_GOAL := help

.PHONY: help \
        venv install install-dev \
        check-deps \
        bitnet-setup bitnet-clone bitnet-submodules bitnet-deps \
        bitnet-patch bitnet-build bitnet-model bitnet-verify bitnet-clean \
        benchmark plots smoke-test \
        eval-winogrande eval-hellaswag \
        clean nuke

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Python environment (Poetry):"
	@echo "  venv                Create/update the Poetry virtual environment"
	@echo "  install             Install runtime dependencies (pyproject.toml)"
	@echo "  install-dev         Install runtime + dev dependencies"
	@echo ""
	@echo "Prerequisites:"
	@echo "  check-deps          Verify cmake, Python 3.11, git, and VS ClangCL are present"
	@echo ""
	@echo "BitNet environment setup (run bitnet-setup for the full pipeline):"
	@echo "  bitnet-setup        check-deps + clone + submodules + deps + patch + build"
	@echo "  bitnet-clone        git clone microsoft/BitNet into \$$(BITNET_DIR)"
	@echo "  bitnet-submodules   git submodule update --init --recursive"
	@echo "  bitnet-deps         Install Python 3.11 deps + gguf-py"
	@echo "  bitnet-patch        Apply ClangCL / chrono compatibility patches"
	@echo "  bitnet-build        Generate TL2 kernels, cmake configure + build"
	@echo "  bitnet-model        Download pre-built i2_s GGUF from microsoft/BitNet-b1.58-2B-4T-gguf"
	@echo "  bitnet-verify       Quick inference smoke-test"
	@echo "  bitnet-clean        Remove \$$(BITNET_DIR)"
	@echo ""
	@echo "Benchmarks & analysis:"
	@echo "  benchmark           Run inference benchmark (latency, throughput, memory, energy)"
	@echo "  eval-winogrande     Run WinoGrande eval with continuation scoring (LIMIT=$(LIMIT))"
	@echo "  eval-hellaswag      Run HellaSwag eval with continuation scoring (LIMIT=$(LIMIT))"
	@echo "  plots               Generate plots + comparison_table.csv from benchmark and accuracy results"
	@echo "  smoke-test          Verify scripts produce expected outputs"
	@echo ""
	@echo "Housekeeping:"
	@echo "  clean               Remove results/plots/ and cached .pyc files"
	@echo "  nuke                clean + remove the Poetry virtualenv"
	@echo ""
	@echo "Overrides:"
	@echo "  BITNET_DIR=../Other/Path   (default: $(BITNET_DIR))"
	@echo "  THREADS=8                  (default: $(THREADS))"
	@echo "  LIMIT=100                  (default: $(LIMIT), applies to eval targets)"

# ── Python environment (Poetry) ───────────────────────────────────────────────

venv:
	$(POETRY) env use $(shell $(PYTHON) -c "import sys; print(sys.executable)")

install: venv
	$(POETRY) install --only main

install-dev: venv
	$(POETRY) install

# ── Prerequisites check ───────────────────────────────────────────────────────
#
# Fails with a human-readable message if any required tool is missing, rather
# than letting the build die with a cryptic compiler or cmake error.
#
# ClangCL check uses vswhere.exe (ships with every VS 2017+ install at the path
# defined by VSWHERE above) to confirm both VS components are registered:
#   Microsoft.VisualStudio.Component.VC.Llvm.Clang       (clang-cl.exe binary)
#   Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset (MSBuild toolset)
# Without both, cmake -T ClangCL fails with MSB8020.
#
# To install: VS Installer → Modify → Individual components → search "Clang":
#   [x] C++ Clang Compiler for Windows
#   [x] MSBuild support for LLVM (clang-cl) toolset

check-deps:
	@echo "Checking prerequisites..."
	@cmake --version 2>/dev/null | grep -q cmake \
	    || { echo "MISSING  cmake >= 3.22  ->  https://cmake.org/download/"; exit 1; }
	@$(PYTHON) --version 2>/dev/null | grep -q Python \
	    || { echo "MISSING  Python 3.11   ->  https://www.python.org/downloads/"; exit 1; }
	@git --version 2>/dev/null | grep -q git \
	    || { echo "MISSING  git           ->  https://git-scm.com/"; exit 1; }
	@"$(VSWHERE)" -latest \
	    -requires Microsoft.VisualStudio.Component.VC.Llvm.Clang \
	    -requires Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset \
	    -find "VC/Tools/Llvm/x64/bin/clang-cl.exe" 2>/dev/null | grep -q clang-cl \
	    || { printf "MISSING  ClangCL not found in Visual Studio.\n         VS Installer > Modify > Individual components:\n           [x] C++ Clang Compiler for Windows\n           [x] MSBuild support for LLVM (clang-cl) toolset\n"; exit 1; }
	@echo "All prerequisites OK."

# ── BitNet environment setup ───────────────────────────────────────────────────
#
# Tested on: Windows 11, Visual Studio 18 (2026), ClangCL 20.1.8, cmake 4.3.2
# Tested commit: $(BITNET_COMMIT)  (HEAD of microsoft/BitNet as of May 2026)
#
# Requirements:
#   - git, cmake >= 3.22
#   - Visual Studio 2022+ with:
#       Desktop development with C++
#       C++ Clang Compiler for Windows  (installs ClangCL)
#       MS-Build Support for LLVM-Toolset
#   - Python 3.11  (py -3.11)
#     Python 3.13 fails: numpy~=1.26.4 has no wheel for 3.13+.
#   - hf  (huggingface_hub CLI, for bitnet-model)

bitnet-setup: check-deps bitnet-clone bitnet-submodules bitnet-deps bitnet-patch bitnet-build
	@echo "BitNet build complete in $(BITNET_DIR)"
	@echo "Run 'make bitnet-model' to download the 2B4T weights."

bitnet-clone:
	git clone https://github.com/microsoft/BitNet.git $(BITNET_DIR)
	cd $(BITNET_DIR) && git checkout $(BITNET_COMMIT)

bitnet-submodules:
	cd $(BITNET_DIR) && git submodule update --init --recursive

# py -3.11 is required throughout.  numpy~=1.26.4 (pinned in requirements.txt)
# has no pre-built wheel for Python 3.13+; building from source fails under
# ClangCL due to C99 complex-type compatibility errors.
# --only-binary=:all: prevents pip from falling back to source compilation.
bitnet-deps:
	cd $(BITNET_DIR) && $(PYTHON) -m pip install -r requirements.txt --only-binary=:all:
	cd $(BITNET_DIR) && $(PYTHON) -m pip install 3rdparty/llama.cpp/gguf-py

# Three patches are required at commit $(BITNET_COMMIT) when building with
# ClangCL 18+ on Windows.  Each is a minimal git-format patch stored in
# patches/ so it can be replayed cleanly with 'git apply'.
#
# patches/bitnet-clangcl-const.patch  — applied to the BitNet repo
#   src/ggml-bitnet-mad.cpp:811
#   'int8_t * y_col = y + col * by' initialises a non-const pointer from a
#   const pointer.  MSVC (cl.exe) emits C4090 and continues; ClangCL 20+
#   treats this as a hard error.  Adding 'const' fixes it.
#   Compiler-flag workarounds tried and ruled out:
#     -Wno-error            — only downgrades promoted warnings, not hard errors
#     /clang:-fpermissive   — no effect on C++ const-correctness errors
#
# patches/llama-chrono.patch  — applied to 3rdparty/llama.cpp
#   common/common.cpp, common/log.cpp
#   Both files use std::chrono but omit '#include <chrono>'.  Under MSVC the
#   header is pulled in transitively; ClangCL is stricter.
#
# patches/llama-chrono-examples.patch  — applied to 3rdparty/llama.cpp
#   examples/imatrix/imatrix.cpp, examples/perplexity/perplexity.cpp
#   Same missing '#include <chrono>' in the example binaries.
bitnet-patch:
	cd $(BITNET_DIR) && git apply $(CURDIR)/patches/bitnet-clangcl-const.patch
	cd $(BITNET_DIR)/3rdparty/llama.cpp && git apply $(CURDIR)/patches/llama-chrono.patch
	cd $(BITNET_DIR)/3rdparty/llama.cpp && git apply $(CURDIR)/patches/llama-chrono-examples.patch

# codegen_tl2.py generates include/bitnet-lut-kernels.h and
# include/kernel_config.ini for the 2B-4T model shape.  (src/ggml-bitnet-mad.cpp
# and src/ggml-bitnet-lut.cpp are already in the repo; only the header is
# generated.)
# cmake -T ClangCL selects the Clang-CL toolchain via the VS generator.
# /EHsc enables C++ exception handling, which ClangCL disables by default but
# llama.cpp requires (it uses throw/catch throughout llama.cpp).
# MSYS_NO_PATHCONV=1 prevents git bash from converting the /EHsc flag to a
# Windows path (C:/Program Files/Git/EHsc) — not needed under PowerShell.
bitnet-build:
	cd $(BITNET_DIR) && $(PYTHON) utils/codegen_tl2.py \
		--model bitnet_b1_58-3B --BM 160,320,320 --BK 96,96,96 --bm 32,32,32
	cd $(BITNET_DIR) && MSYS_NO_PATHCONV=1 cmake -B build -DBITNET_X86_TL2=ON \
		-T ClangCL \
		"-DCMAKE_CXX_FLAGS=/EHsc"
	cd $(BITNET_DIR) && cmake --build build --config Release

# ── Model download ─────────────────────────────────────────────────────────────
#
# The microsoft/BitNet-b1.58-2B-4T-gguf repo ships a ready-to-use
# ggml-model-i2_s.gguf, so no conversion or re-quantization is needed.
# (The safetensors repo microsoft/BitNet-b1.58-2B-4T stores weights in a
# pre-quantized uint8+scale format that the standard converter cannot handle
# without the weight_scale unpacking logic; the GGUF repo sidesteps this.)

bitnet-model:
	hf download microsoft/BitNet-b1.58-2B-4T-gguf \
		--local-dir $(BITNET_DIR)/$(BITNET_MODEL)

# ── Verify / Benchmark ─────────────────────────────────────────────────────────

bitnet-verify:
	cd $(BITNET_DIR) && $(PYTHON) run_inference.py \
		-m $(BITNET_MODEL)/ggml-model-$(BITNET_QUANT).gguf \
		-p "What is 2+2?" \
		-n 32 \
		-t $(THREADS)

benchmark:
	$(POETRY) run python scripts/metrics_tracker.py \
		--bitnet-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS)

eval-winogrande:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task winogrande \
		--bitnet-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--start-server

eval-hellaswag:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task hellaswag \
		--bitnet-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--start-server

plots:
	$(POETRY) run python scripts/compare_runs.py

smoke-test:
	$(POETRY) run python scripts/smoke_test.py

# ── Housekeeping ───────────────────────────────────────────────────────────────

bitnet-clean:
	rm -rf $(BITNET_DIR)

clean:
	rm -rf results/plots
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

nuke: clean
	$(POETRY) env remove --all
