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

QWEN_DIR           ?= ../Models/Qwen
QWEN_REPO          := Qwen/Qwen2.5-1.5B-Instruct-GGUF
QWEN_QUANT         := q8_0
QWEN_FILE          := qwen2.5-1.5b-instruct-$(QWEN_QUANT).gguf
QWEN_MODEL         ?= $(QWEN_DIR)/$(QWEN_FILE)
QWEN_LLAMACPP_DIR  := $(QWEN_DIR)/llama.cpp
QWEN_LLAMACPP_REPO := https://github.com/ggml-org/llama.cpp.git
QWEN_LLAMACPP_COMMIT := 1e5ad35d560b90a8ac447d149c8f8447ae1fcaa0  # HEAD as of May 2026; pinned for reproducibility
QWEN_CLI           ?= $(QWEN_LLAMACPP_DIR)/build/bin/Release/llama-cli.exe
QWEN_SERVER        ?= $(QWEN_LLAMACPP_DIR)/build/bin/Release/llama-server.exe

# Qwen Q4_K_M variant — same model, same llama.cpp build, more aggressive
# quantization (~1 GB instead of 1.65 GB).  Tests "would Q4 have been enough?"
# alongside the existing Q8_0 measurement.
QWEN_Q4_QUANT      := q4_k_m
QWEN_Q4_FILE       := qwen2.5-1.5b-instruct-$(QWEN_Q4_QUANT).gguf
QWEN_Q4_MODEL      ?= $(QWEN_DIR)/$(QWEN_Q4_FILE)

.DEFAULT_GOAL := help

.PHONY: help \
        venv install install-dev \
        check-deps \
        bitnet-setup bitnet-clone bitnet-submodules bitnet-deps \
        bitnet-patch bitnet-build bitnet-model bitnet-verify bitnet-clean \
        qwen-setup qwen-clone qwen-build qwen-model qwen-verify qwen-clean \
        qwen-q4-model qwen-q4-verify \
        benchmark-bitnet benchmark-qwen benchmark-qwen-q4 benchmark \
        benchmark-qwen-on-bitnet-fork \
        plots smoke-test smoke-test-bitnet smoke-test-qwen smoke-test-qwen-q4 \
        eval-arc-easy-bitnet eval-arc-easy-qwen eval-arc-easy-qwen-q4 eval-arc-easy \
        eval-arc-challenge-bitnet eval-arc-challenge-qwen eval-arc-challenge-qwen-q4 eval-arc-challenge \
        eval-mmlu-bitnet eval-mmlu-qwen eval-mmlu-qwen-q4 eval-mmlu \
        eval-winogrande-bitnet eval-winogrande-qwen eval-winogrande-qwen-q4 eval-winogrande \
        eval-hellaswag-bitnet eval-hellaswag-qwen eval-hellaswag-qwen-q4 eval-hellaswag \
        eval-accuracy-bitnet eval-accuracy-qwen eval-accuracy-qwen-q4 eval-accuracy \
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
	@echo "Qwen2.5 1.5B baseline (fully independent of BitNet):"
	@echo "  qwen-setup          clone + build + model (full pipeline)"
	@echo "  qwen-clone          git clone ggml-org/llama.cpp into \$$(QWEN_LLAMACPP_DIR)"
	@echo "  qwen-build          cmake configure + build (MSVC, no ClangCL)"
	@echo "  qwen-model          Download Qwen2.5-1.5B-Instruct Q8_0 GGUF into \$$(QWEN_DIR)"
	@echo "  qwen-verify         Quick inference smoke-test with Qwen's llama-cli"
	@echo "  qwen-q4-model       Download Qwen2.5-1.5B-Instruct Q4_K_M GGUF (reuses qwen-build)"
	@echo "  qwen-q4-verify      Quick inference smoke-test with Qwen Q4_K_M"
	@echo "  qwen-clean          Remove \$$(QWEN_DIR)"
	@echo ""
	@echo "Benchmarks & analysis:"
	@echo "  benchmark-bitnet         BitNet inference benchmark (throughput, memory, energy)"
	@echo "  benchmark-qwen           Qwen Q8_0 inference benchmark"
	@echo "  benchmark-qwen-q4        Qwen Q4_K_M inference benchmark"
	@echo "  benchmark                All three models"
	@echo "  eval-arc-easy-bitnet         BitNet ARC-Easy eval (LIMIT=$(LIMIT))"
	@echo "  eval-arc-easy-qwen           Qwen Q8_0 ARC-Easy eval"
	@echo "  eval-arc-easy-qwen-q4        Qwen Q4_K_M ARC-Easy eval"
	@echo "  eval-arc-easy                All three models ARC-Easy"
	@echo "  eval-arc-challenge-bitnet    BitNet ARC-Challenge eval"
	@echo "  eval-arc-challenge-qwen      Qwen Q8_0 ARC-Challenge eval"
	@echo "  eval-arc-challenge-qwen-q4   Qwen Q4_K_M ARC-Challenge eval"
	@echo "  eval-arc-challenge           All three models ARC-Challenge"
	@echo "  eval-mmlu-bitnet             BitNet MMLU 5-shot eval"
	@echo "  eval-mmlu-qwen               Qwen Q8_0 MMLU 5-shot eval"
	@echo "  eval-mmlu-qwen-q4            Qwen Q4_K_M MMLU 5-shot eval"
	@echo "  eval-mmlu                    All three models MMLU"
	@echo "  eval-winogrande-bitnet       BitNet WinoGrande eval"
	@echo "  eval-winogrande-qwen         Qwen Q8_0 WinoGrande eval"
	@echo "  eval-winogrande-qwen-q4      Qwen Q4_K_M WinoGrande eval"
	@echo "  eval-winogrande              All three models WinoGrande"
	@echo "  eval-hellaswag-bitnet        BitNet HellaSwag eval"
	@echo "  eval-hellaswag-qwen          Qwen Q8_0 HellaSwag eval"
	@echo "  eval-hellaswag-qwen-q4       Qwen Q4_K_M HellaSwag eval"
	@echo "  eval-hellaswag               All three models HellaSwag"
	@echo "  eval-accuracy-bitnet         All tasks, BitNet only"
	@echo "  eval-accuracy-qwen           All tasks, Qwen Q8_0 only"
	@echo "  eval-accuracy-qwen-q4        All tasks, Qwen Q4_K_M only"
	@echo "  eval-accuracy                All tasks, all three models"
	@echo "  plots                Generate plots + comparison_table.csv from benchmark and accuracy results"
	@echo "  smoke-test           Verify scripts produce expected outputs (all three models)"
	@echo "  smoke-test-bitnet    Verify BitNet inference only"
	@echo "  smoke-test-qwen      Verify Qwen Q8_0 inference only"
	@echo "  smoke-test-qwen-q4   Verify Qwen Q4_K_M inference only"
	@echo ""
	@echo "Housekeeping:"
	@echo "  clean               Remove results/plots/ and cached .pyc files"
	@echo "  nuke                clean + remove the Poetry virtualenv"
	@echo ""
	@echo "Overrides:"
	@echo "  BITNET_DIR=../Other/Path   (default: $(BITNET_DIR))"
	@echo "  QWEN_DIR=../Other/Path     (default: $(QWEN_DIR))"
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

QWEN_BENCH_OUT    ?= results/qwen_step_metrics.csv
QWEN_Q4_BENCH_OUT ?= results/qwen_q4_step_metrics.csv

benchmark-bitnet:
	$(POETRY) run python scripts/metrics_tracker.py \
		--llama-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS)

benchmark-qwen:
	$(POETRY) run python scripts/metrics_tracker.py \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_MODEL) \
		--out $(QWEN_BENCH_OUT) \
		--threads $(THREADS)

benchmark-qwen-q4:
	$(POETRY) run python scripts/metrics_tracker.py \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q4_MODEL) \
		--out $(QWEN_Q4_BENCH_OUT) \
		--threads $(THREADS)

benchmark: benchmark-bitnet benchmark-qwen benchmark-qwen-q4

# Sensitivity check (FINAL_REPORT §6.8): run Qwen Q8_0 against BitNet's
# llama.cpp fork instead of upstream.  Isolates how much of Qwen's measured
# throughput is attributable to upstream's ~1 year of optimization vs the
# quantization itself.  The TL2 ternary kernel only activates for i2_s
# weights, so Qwen runs through the fork's standard Q8_0 kernels (older
# llama.cpp), not the BitNet-specific path.
QWEN_ON_BITNET_FORK_OUT ?= results/qwen_on_bitnet_fork_step_metrics.csv

benchmark-qwen-on-bitnet-fork:
	$(POETRY) run python scripts/metrics_tracker.py \
		--llama-dir $(BITNET_DIR) \
		--model $(QWEN_MODEL) \
		--out $(QWEN_ON_BITNET_FORK_OUT) \
		--threads $(THREADS)

BITNET_ACC_OUT  ?= results/accuracy_results_bitnet.json
QWEN_ACC_OUT    ?= results/accuracy_results_qwen.json
QWEN_Q4_ACC_OUT ?= results/accuracy_results_qwen_q4.json

eval-arc-easy-bitnet:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task arc_easy \
		--llama-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--start-server

eval-arc-easy-qwen:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task arc_easy \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_ACC_OUT) \
		--start-server

eval-arc-easy-qwen-q4:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task arc_easy \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q4_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_Q4_ACC_OUT) \
		--start-server

eval-arc-easy: eval-arc-easy-bitnet eval-arc-easy-qwen eval-arc-easy-qwen-q4

eval-arc-challenge-bitnet:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task arc_challenge \
		--llama-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--start-server

eval-arc-challenge-qwen:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task arc_challenge \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_ACC_OUT) \
		--start-server

eval-arc-challenge-qwen-q4:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task arc_challenge \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q4_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_Q4_ACC_OUT) \
		--start-server

eval-arc-challenge: eval-arc-challenge-bitnet eval-arc-challenge-qwen eval-arc-challenge-qwen-q4

eval-mmlu-bitnet:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task mmlu \
		--num-fewshot 5 \
		--llama-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--start-server

eval-mmlu-qwen:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task mmlu \
		--num-fewshot 5 \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_ACC_OUT) \
		--start-server

eval-mmlu-qwen-q4:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task mmlu \
		--num-fewshot 5 \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q4_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_Q4_ACC_OUT) \
		--start-server

eval-mmlu: eval-mmlu-bitnet eval-mmlu-qwen eval-mmlu-qwen-q4

eval-winogrande-bitnet:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task winogrande \
		--llama-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--start-server

eval-winogrande-qwen:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task winogrande \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_ACC_OUT) \
		--start-server

eval-winogrande-qwen-q4:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task winogrande \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q4_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_Q4_ACC_OUT) \
		--start-server

eval-winogrande: eval-winogrande-bitnet eval-winogrande-qwen eval-winogrande-qwen-q4

eval-hellaswag-bitnet:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task hellaswag \
		--llama-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--start-server

eval-hellaswag-qwen:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task hellaswag \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_ACC_OUT) \
		--start-server

eval-hellaswag-qwen-q4:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task hellaswag \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q4_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_Q4_ACC_OUT) \
		--start-server

eval-hellaswag: eval-hellaswag-bitnet eval-hellaswag-qwen eval-hellaswag-qwen-q4

eval-accuracy-bitnet: eval-arc-easy-bitnet eval-arc-challenge-bitnet eval-mmlu-bitnet eval-winogrande-bitnet eval-hellaswag-bitnet

eval-accuracy-qwen: eval-arc-easy-qwen eval-arc-challenge-qwen eval-mmlu-qwen eval-winogrande-qwen eval-hellaswag-qwen

eval-accuracy-qwen-q4: eval-arc-easy-qwen-q4 eval-arc-challenge-qwen-q4 eval-mmlu-qwen-q4 eval-winogrande-qwen-q4 eval-hellaswag-qwen-q4

eval-accuracy: eval-accuracy-bitnet eval-accuracy-qwen eval-accuracy-qwen-q4

plots:
	$(POETRY) run python scripts/compare_runs.py

smoke-test:
	$(POETRY) run python scripts/smoke_test.py

smoke-test-bitnet:
	$(POETRY) run python scripts/smoke_test.py bitnet

smoke-test-qwen:
	$(POETRY) run python scripts/smoke_test.py qwen

smoke-test-qwen-q4:
	$(POETRY) run python scripts/smoke_test.py qwen-q4

# ── Qwen2.5 1.5B baseline ─────────────────────────────────────────────────────
#
# Fully independent of BitNet: clones upstream ggml-org/llama.cpp and builds
# it with standard MSVC (no ClangCL, no BitNet patches required).
#
# Q8_0 (1.89 GB) is the quantization closest to FP16 accuracy, chosen so
# that accuracy comparisons against published FP16 baselines are as fair as
# possible without requiring a full F16 download (~3 GB).
#
# Tested on: Windows 11, Visual Studio 2022, MSVC 19.x, cmake 4.3.2
# Tested commit: $(QWEN_LLAMACPP_COMMIT)
#
# Requirements:
#   - git, cmake >= 3.22
#   - Visual Studio 2022+ with Desktop development with C++ workload
#   - hf  (huggingface_hub CLI, for qwen-model)

qwen-setup: qwen-clone qwen-build qwen-model
	@echo "Qwen build complete. Run 'make qwen-verify' to test inference."

qwen-clone:
	git clone $(QWEN_LLAMACPP_REPO) $(QWEN_LLAMACPP_DIR)
	cd $(QWEN_LLAMACPP_DIR) && git checkout $(QWEN_LLAMACPP_COMMIT)

# -DLLAMA_CURL=OFF  avoids an optional OpenSSL dependency not needed for inference.
# -DGGML_NATIVE=ON  enables AVX2/FMA/F16C on the build machine for better throughput.
qwen-build:
	cd $(QWEN_LLAMACPP_DIR) && cmake -B build \
		-DLLAMA_CURL=OFF \
		-DGGML_NATIVE=ON
	cd $(QWEN_LLAMACPP_DIR) && cmake --build build --config Release

qwen-model:
	hf download $(QWEN_REPO) $(QWEN_FILE) \
		--local-dir $(QWEN_DIR)

# Notes on the recipe form:
#   - cmd.exe mis-parses forward-slash paths that start with "..": it
#     treats "../Models/..." as command ".." with switches "/Models/...".
#     Quoting the executable path tells cmd to treat it as one literal token
#     (sh also accepts quoted paths), and the single-line form avoids
#     line-continuation parsing differences between shells.
#   - Upstream llama.cpp defaults to interactive conversation mode and
#     ignores stdin EOF, so the CLI never exits.  --single-turn makes it
#     process one prompt and exit while still applying the chat template
#     (unlike --no-cnv, which strips the template and returns nothing).
qwen-verify:
	"$(QWEN_CLI)" -m $(QWEN_MODEL) -p "What is 2+2?" -n 32 -t $(THREADS) --single-turn

# Qwen Q4_K_M variant — reuses the qwen-build (same llama.cpp binary).
# Run qwen-setup first if the build doesn't exist yet.
qwen-q4-model:
	hf download $(QWEN_REPO) $(QWEN_Q4_FILE) \
		--local-dir $(QWEN_DIR)

qwen-q4-verify:
	"$(QWEN_CLI)" -m $(QWEN_Q4_MODEL) -p "What is 2+2?" -n 32 -t $(THREADS) --single-turn

qwen-clean:
	rm -rf $(QWEN_DIR)

# ── Housekeeping ───────────────────────────────────────────────────────────────

bitnet-clean:
	rm -rf $(BITNET_DIR)

clean:
	rm -rf results/plots
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

nuke: clean
	$(POETRY) env remove --all
