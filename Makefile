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
QWEN_LLAMACPP_DIR  := $(QWEN_DIR)/llama.cpp
QWEN_LLAMACPP_REPO := https://github.com/ggml-org/llama.cpp.git
QWEN_LLAMACPP_COMMIT := 1e5ad35d560b90a8ac447d149c8f8447ae1fcaa0  # HEAD as of May 2026; pinned for reproducibility
QWEN_CLI           ?= $(QWEN_LLAMACPP_DIR)/build/bin/Release/llama-cli.exe
QWEN_SERVER        ?= $(QWEN_LLAMACPP_DIR)/build/bin/Release/llama-server.exe

# Qwen Q8_0 variant — 8-bit GGUF, near-FP16 accuracy, ~1.65 GiB.
QWEN_Q8_QUANT      := q8_0
QWEN_Q8_FILE       := qwen2.5-1.5b-instruct-$(QWEN_Q8_QUANT).gguf
QWEN_Q8_MODEL      ?= $(QWEN_DIR)/$(QWEN_Q8_FILE)

# Qwen Q4_K_M variant — same model + same llama.cpp build, more aggressive
# quantization (~1 GB instead of 1.65 GB).  Tests "would Q4 have been enough?"
# alongside the Q8_0 measurement.
QWEN_Q4_QUANT      := q4_k_m
QWEN_Q4_FILE       := qwen2.5-1.5b-instruct-$(QWEN_Q4_QUANT).gguf
QWEN_Q4_MODEL      ?= $(QWEN_DIR)/$(QWEN_Q4_FILE)

# ── System-card / technical-report PDFs ──────────────────────────────────────
# These URLs were verified 2026-05-20.  Anthropic rotates the asset hashes
# in their /m/<hash>/ paths from time to time — if a download 404s, check
# https://www.anthropic.com/system-cards for the current link.
CLOUD_DIR             ?= ../Models/Cloud
CURL                  := curl -L --fail --silent --show-error
BITNET_PAPER_URL      := https://arxiv.org/pdf/2504.12285
QWEN_PAPER_URL        := https://arxiv.org/pdf/2412.15115
GPT4O_CARD_URL        := https://cdn.openai.com/gpt-4o-system-card.pdf
HAIKU_45_CARD_URL     := https://assets.anthropic.com/m/99128ddd009bdcb/Claude-Haiku-4-5-System-Card.pdf
SONNET_45_CARD_URL    := https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf
OPUS_47_CARD_URL      := https://www.anthropic.com/claude-opus-4-7-system-card

.DEFAULT_GOAL := help

.PHONY: help \
        venv install install-dev \
        check-deps \
        bitnet-setup bitnet-clone bitnet-submodules bitnet-deps \
        bitnet-patch bitnet-build bitnet-model bitnet-verify bitnet-clean \
        qwen-q8-setup qwen-clone qwen-build qwen-q8-model qwen-q8-verify qwen-clean \
        qwen-q4-model qwen-q4-verify \
        benchmark-bitnet benchmark-qwen-q8 benchmark-qwen-q4 benchmark \
        benchmark-qwen-q8-on-bitnet-fork \
        benchmark-threads-bitnet benchmark-threads-qwen-q8 benchmark-threads-qwen-q4 benchmark-threads \
        plots marginal-energy \
        smoke-test smoke-test-bitnet smoke-test-qwen-q8 smoke-test-qwen-q4 \
        system-cards system-cards-bitnet system-cards-qwen system-cards-cloud \
        eval-arc-easy-bitnet eval-arc-easy-qwen-q8 eval-arc-easy-qwen-q4 eval-arc-easy \
        eval-arc-challenge-bitnet eval-arc-challenge-qwen-q8 eval-arc-challenge-qwen-q4 eval-arc-challenge \
        eval-mmlu-bitnet eval-mmlu-qwen-q8 eval-mmlu-qwen-q4 eval-mmlu \
        eval-winogrande-bitnet eval-winogrande-qwen-q8 eval-winogrande-qwen-q4 eval-winogrande \
        eval-hellaswag-bitnet eval-hellaswag-qwen-q8 eval-hellaswag-qwen-q4 eval-hellaswag \
        eval-accuracy-bitnet eval-accuracy-qwen-q8 eval-accuracy-qwen-q4 eval-accuracy \
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
	@echo "  qwen-q8-setup                 clone + build + Q8_0 model (full pipeline)"
	@echo "  qwen-clone                    git clone ggml-org/llama.cpp into \$$(QWEN_LLAMACPP_DIR)"
	@echo "  qwen-build                    cmake configure + build (MSVC, no ClangCL)"
	@echo "  qwen-q8-model                 Download Qwen2.5-1.5B-Instruct Q8_0 GGUF into \$$(QWEN_DIR)"
	@echo "  qwen-q8-verify                Quick inference smoke-test with Qwen Q8_0"
	@echo "  qwen-q4-model                 Download Qwen2.5-1.5B-Instruct Q4_K_M GGUF (reuses qwen-build)"
	@echo "  qwen-q4-verify                Quick inference smoke-test with Qwen Q4_K_M"
	@echo "  qwen-clean                    Remove \$$(QWEN_DIR)"
	@echo ""
	@echo "Benchmarks & analysis:"
	@echo "  benchmark-bitnet              BitNet inference benchmark (throughput, memory, energy)"
	@echo "  benchmark-qwen-q8             Qwen Q8_0 inference benchmark"
	@echo "  benchmark-qwen-q4             Qwen Q4_K_M inference benchmark"
	@echo "  benchmark                     All three models"
	@echo "  eval-arc-easy-bitnet          BitNet ARC-Easy eval (LIMIT=$(LIMIT))"
	@echo "  eval-arc-easy-qwen-q8         Qwen Q8_0 ARC-Easy eval"
	@echo "  eval-arc-easy-qwen-q4         Qwen Q4_K_M ARC-Easy eval"
	@echo "  eval-arc-easy                 All three models ARC-Easy"
	@echo "  eval-arc-challenge-bitnet     BitNet ARC-Challenge eval"
	@echo "  eval-arc-challenge-qwen-q8    Qwen Q8_0 ARC-Challenge eval"
	@echo "  eval-arc-challenge-qwen-q4    Qwen Q4_K_M ARC-Challenge eval"
	@echo "  eval-arc-challenge            All three models ARC-Challenge"
	@echo "  eval-mmlu-bitnet              BitNet MMLU 5-shot eval"
	@echo "  eval-mmlu-qwen-q8             Qwen Q8_0 MMLU 5-shot eval"
	@echo "  eval-mmlu-qwen-q4             Qwen Q4_K_M MMLU 5-shot eval"
	@echo "  eval-mmlu                     All three models MMLU"
	@echo "  eval-winogrande-bitnet        BitNet WinoGrande eval"
	@echo "  eval-winogrande-qwen-q8       Qwen Q8_0 WinoGrande eval"
	@echo "  eval-winogrande-qwen-q4       Qwen Q4_K_M WinoGrande eval"
	@echo "  eval-winogrande               All three models WinoGrande"
	@echo "  eval-hellaswag-bitnet         BitNet HellaSwag eval"
	@echo "  eval-hellaswag-qwen-q8        Qwen Q8_0 HellaSwag eval"
	@echo "  eval-hellaswag-qwen-q4        Qwen Q4_K_M HellaSwag eval"
	@echo "  eval-hellaswag                All three models HellaSwag"
	@echo "  eval-accuracy-bitnet          All tasks, BitNet only"
	@echo "  eval-accuracy-qwen-q8         All tasks, Qwen Q8_0 only"
	@echo "  eval-accuracy-qwen-q4         All tasks, Qwen Q4_K_M only"
	@echo "  eval-accuracy                 All tasks, all three models"
	@echo "  plots                         Generate plots + comparison_table.csv from benchmark and accuracy results"
	@echo "  marginal-energy               Measure CPU idle baseline (90s) and print marginal J/tok per bench row (REPORT.md §4.3)"
	@echo "  smoke-test                    Verify scripts produce expected outputs (all three models)"
	@echo "  smoke-test-bitnet             Verify BitNet inference only"
	@echo "  smoke-test-qwen-q8            Verify Qwen Q8_0 inference only"
	@echo "  smoke-test-qwen-q4            Verify Qwen Q4_K_M inference only"
	@echo ""
	@echo "System cards / technical reports (one-off PDF downloads):"
	@echo "  system-cards                  Download every paper / system card used in the comparison"
	@echo "  system-cards-bitnet           BitNet b1.58 2B4T technical report (arXiv:2504.12285) -> \$$(BITNET_DIR)"
	@echo "  system-cards-qwen             Qwen2.5 technical report (arXiv:2412.15115) -> \$$(QWEN_DIR)"
	@echo "  system-cards-cloud            GPT-4o + Claude 4.5/4.7 system cards -> \$$(CLOUD_DIR)"
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

QWEN_Q8_BENCH_OUT    ?= results/qwen_q8_step_metrics.csv
QWEN_Q4_BENCH_OUT ?= results/qwen_q4_step_metrics.csv

benchmark-bitnet:
	$(POETRY) run python scripts/metrics_tracker.py \
		--llama-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS)

benchmark-qwen-q8:
	$(POETRY) run python scripts/metrics_tracker.py \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q8_MODEL) \
		--out $(QWEN_Q8_BENCH_OUT) \
		--threads $(THREADS)

benchmark-qwen-q4:
	$(POETRY) run python scripts/metrics_tracker.py \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q4_MODEL) \
		--out $(QWEN_Q4_BENCH_OUT) \
		--threads $(THREADS)

benchmark: benchmark-bitnet benchmark-qwen-q8 benchmark-qwen-q4

# Sensitivity check (REPORT §6.8): run Qwen Q8_0 against BitNet's
# llama.cpp fork instead of upstream.  Isolates how much of Qwen's measured
# throughput is attributable to upstream's ~1 year of optimization vs the
# quantization itself.  The TL2 ternary kernel only activates for i2_s
# weights, so Qwen runs through the fork's standard Q8_0 kernels (older
# llama.cpp), not the BitNet-specific path.
QWEN_Q8_ON_BITNET_FORK_OUT ?= results/qwen_q8_on_bitnet_fork_step_metrics.csv

benchmark-qwen-q8-on-bitnet-fork:
	$(POETRY) run python scripts/metrics_tracker.py \
		--llama-dir $(BITNET_DIR) \
		--model $(QWEN_Q8_MODEL) \
		--out $(QWEN_Q8_ON_BITNET_FORK_OUT) \
		--threads $(THREADS)

# Thread-count sensitivity sweep (Phase 5).  Re-runs each model's bench
# at THREADS=1, 2, 4, 6 (the i5-9400F has 6 cores, no SMT).  Writes to
# dedicated *_thread_sweep.csv files so the main comparison CSVs stay
# clean at the THREADS=4 reference condition.  Used by compare_runs.py
# to produce thread_scaling.png and by REPORT §5.4.
BITNET_THREAD_SWEEP_OUT  ?= results/bitnet_thread_sweep.csv
QWEN_Q8_THREAD_SWEEP_OUT    ?= results/qwen_q8_thread_sweep.csv
QWEN_Q4_THREAD_SWEEP_OUT ?= results/qwen_q4_thread_sweep.csv

benchmark-threads-bitnet:
	# threads=1 deliberately skipped: BitNet's TL2 kernel (BM=160) hits
	# STATUS_STACK_OVERFLOW (0xC00000FD) at single-thread regardless of
	# -ub setting.  threads=2 requires --ubatch 64 (the default 128 also
	# crashes); for sweep consistency we use --ubatch 64 across all
	# three thread counts.  Numbers will be slightly lower than the main
	# reference (which used --ubatch 128 at threads=4) but the scaling
	# behavior is what this sweep measures.  Documented in REPORT §5.4.
	$(POETRY) run python scripts/metrics_tracker.py --llama-dir $(BITNET_DIR) --model $(MODEL) --threads 2 --out $(BITNET_THREAD_SWEEP_OUT) --no-energy --ubatch 64
	$(POETRY) run python scripts/metrics_tracker.py --llama-dir $(BITNET_DIR) --model $(MODEL) --threads 4 --out $(BITNET_THREAD_SWEEP_OUT) --no-energy --ubatch 64
	$(POETRY) run python scripts/metrics_tracker.py --llama-dir $(BITNET_DIR) --model $(MODEL) --threads 6 --out $(BITNET_THREAD_SWEEP_OUT) --no-energy --ubatch 64

benchmark-threads-qwen-q8:
	$(POETRY) run python scripts/metrics_tracker.py --llama-dir $(QWEN_LLAMACPP_DIR) --model $(QWEN_Q8_MODEL) --threads 1 --out $(QWEN_Q8_THREAD_SWEEP_OUT) --no-energy
	$(POETRY) run python scripts/metrics_tracker.py --llama-dir $(QWEN_LLAMACPP_DIR) --model $(QWEN_Q8_MODEL) --threads 2 --out $(QWEN_Q8_THREAD_SWEEP_OUT) --no-energy
	$(POETRY) run python scripts/metrics_tracker.py --llama-dir $(QWEN_LLAMACPP_DIR) --model $(QWEN_Q8_MODEL) --threads 4 --out $(QWEN_Q8_THREAD_SWEEP_OUT) --no-energy
	$(POETRY) run python scripts/metrics_tracker.py --llama-dir $(QWEN_LLAMACPP_DIR) --model $(QWEN_Q8_MODEL) --threads 6 --out $(QWEN_Q8_THREAD_SWEEP_OUT) --no-energy

benchmark-threads-qwen-q4:
	$(POETRY) run python scripts/metrics_tracker.py --llama-dir $(QWEN_LLAMACPP_DIR) --model $(QWEN_Q4_MODEL) --threads 1 --out $(QWEN_Q4_THREAD_SWEEP_OUT) --no-energy
	$(POETRY) run python scripts/metrics_tracker.py --llama-dir $(QWEN_LLAMACPP_DIR) --model $(QWEN_Q4_MODEL) --threads 2 --out $(QWEN_Q4_THREAD_SWEEP_OUT) --no-energy
	$(POETRY) run python scripts/metrics_tracker.py --llama-dir $(QWEN_LLAMACPP_DIR) --model $(QWEN_Q4_MODEL) --threads 4 --out $(QWEN_Q4_THREAD_SWEEP_OUT) --no-energy
	$(POETRY) run python scripts/metrics_tracker.py --llama-dir $(QWEN_LLAMACPP_DIR) --model $(QWEN_Q4_MODEL) --threads 6 --out $(QWEN_Q4_THREAD_SWEEP_OUT) --no-energy

benchmark-threads: benchmark-threads-bitnet benchmark-threads-qwen-q8 benchmark-threads-qwen-q4

BITNET_ACC_OUT  ?= results/accuracy_results_bitnet.json
QWEN_Q8_ACC_OUT    ?= results/accuracy_results_qwen_q8.json
QWEN_Q4_ACC_OUT ?= results/accuracy_results_qwen_q4.json

eval-arc-easy-bitnet:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task arc_easy \
		--llama-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--start-server

eval-arc-easy-qwen-q8:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task arc_easy \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q8_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_Q8_ACC_OUT) \
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

eval-arc-easy: eval-arc-easy-bitnet eval-arc-easy-qwen-q8 eval-arc-easy-qwen-q4

eval-arc-challenge-bitnet:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task arc_challenge \
		--llama-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--start-server

eval-arc-challenge-qwen-q8:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task arc_challenge \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q8_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_Q8_ACC_OUT) \
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

eval-arc-challenge: eval-arc-challenge-bitnet eval-arc-challenge-qwen-q8 eval-arc-challenge-qwen-q4

eval-mmlu-bitnet:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task mmlu \
		--num-fewshot 5 \
		--llama-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--start-server

eval-mmlu-qwen-q8:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task mmlu \
		--num-fewshot 5 \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q8_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_Q8_ACC_OUT) \
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

eval-mmlu: eval-mmlu-bitnet eval-mmlu-qwen-q8 eval-mmlu-qwen-q4

eval-winogrande-bitnet:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task winogrande \
		--llama-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--start-server

eval-winogrande-qwen-q8:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task winogrande \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q8_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_Q8_ACC_OUT) \
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

eval-winogrande: eval-winogrande-bitnet eval-winogrande-qwen-q8 eval-winogrande-qwen-q4

eval-hellaswag-bitnet:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task hellaswag \
		--llama-dir $(BITNET_DIR) \
		--model $(MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--start-server

eval-hellaswag-qwen-q8:
	$(POETRY) run python scripts/eval_accuracy.py \
		--task hellaswag \
		--llama-dir $(QWEN_LLAMACPP_DIR) \
		--model $(QWEN_Q8_MODEL) \
		--threads $(THREADS) \
		--limit $(LIMIT) \
		--out $(QWEN_Q8_ACC_OUT) \
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

eval-hellaswag: eval-hellaswag-bitnet eval-hellaswag-qwen-q8 eval-hellaswag-qwen-q4

eval-accuracy-bitnet: eval-arc-easy-bitnet eval-arc-challenge-bitnet eval-mmlu-bitnet eval-winogrande-bitnet eval-hellaswag-bitnet

eval-accuracy-qwen-q8: eval-arc-easy-qwen-q8 eval-arc-challenge-qwen-q8 eval-mmlu-qwen-q8 eval-winogrande-qwen-q8 eval-hellaswag-qwen-q8

eval-accuracy-qwen-q4: eval-arc-easy-qwen-q4 eval-arc-challenge-qwen-q4 eval-mmlu-qwen-q4 eval-winogrande-qwen-q4 eval-hellaswag-qwen-q4

eval-accuracy: eval-accuracy-bitnet eval-accuracy-qwen-q8 eval-accuracy-qwen-q4

plots:
	$(POETRY) run python scripts/compare_runs.py

# Measure a CPU idle baseline (90s) and subtract it from each bench row's
# energy_kwh to estimate inference-marginal J/tok.  Closes most of the
# gap between our total-system numbers and the BitNet paper's
# inference-only J/tok (REPORT.md §4.3, §6.4).
marginal-energy:
	$(POETRY) run python scripts/measure_marginal_energy.py

smoke-test:
	$(POETRY) run python scripts/smoke_test.py

smoke-test-bitnet:
	$(POETRY) run python scripts/smoke_test.py bitnet

smoke-test-qwen-q8:
	$(POETRY) run python scripts/smoke_test.py qwen-q8

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
#   - hf  (huggingface_hub CLI, for qwen-q8-model / qwen-q4-model)

qwen-q8-setup: qwen-clone qwen-build qwen-q8-model
	@echo "Qwen build complete. Run 'make qwen-q8-verify' to test inference."

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

qwen-q8-model:
	hf download $(QWEN_REPO) $(QWEN_Q8_FILE) \
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
qwen-q8-verify:
	"$(QWEN_CLI)" -m $(QWEN_Q8_MODEL) -p "What is 2+2?" -n 32 -t $(THREADS) --single-turn

# Qwen Q4_K_M variant — reuses the qwen-build (same llama.cpp binary).
# Run qwen-q8-setup first if the build doesn't exist yet.
qwen-q4-model:
	hf download $(QWEN_REPO) $(QWEN_Q4_FILE) \
		--local-dir $(QWEN_DIR)

qwen-q4-verify:
	"$(QWEN_CLI)" -m $(QWEN_Q4_MODEL) -p "What is 2+2?" -n 32 -t $(THREADS) --single-turn

qwen-clean:
	rm -rf $(QWEN_DIR)

# ── System-card / technical-report downloads ──────────────────────────────────
#
# One-off PDF downloads of each model's paper or safety/system card.  The
# files land next to the model weights:
#
#   $(BITNET_DIR)/BitNet-b1.58-2B4T-Technical-Report.pdf
#   $(QWEN_DIR)/Qwen2.5-Technical-Report.pdf
#   $(CLOUD_DIR)/{GPT-4o,Claude-Haiku-4-5,Claude-Sonnet-4-5,Claude-Opus-4-7}-System-Card.pdf
#
# Each target is idempotent — already-present files are not re-downloaded, so
# 'make system-cards' can be safely re-run after a single missing file is
# deleted.  Pass -B to force re-download.
#
# GPT-4o mini is intentionally not its own download: OpenAI does not publish a
# standalone system card for it.  Its safety eval numbers appear inside
# GPT-4o-System-Card.pdf as a comparison column.  See $(CLOUD_DIR)/README.md.

system-cards: system-cards-bitnet system-cards-qwen system-cards-cloud
	@echo "All system cards present."

system-cards-bitnet:
	@mkdir -p $(BITNET_DIR)
	@[ -f $(BITNET_DIR)/BitNet-b1.58-2B4T-Technical-Report.pdf ] \
	    || $(CURL) -o $(BITNET_DIR)/BitNet-b1.58-2B4T-Technical-Report.pdf $(BITNET_PAPER_URL)

system-cards-qwen:
	@mkdir -p $(QWEN_DIR)
	@[ -f $(QWEN_DIR)/Qwen2.5-Technical-Report.pdf ] \
	    || $(CURL) -o $(QWEN_DIR)/Qwen2.5-Technical-Report.pdf $(QWEN_PAPER_URL)

system-cards-cloud:
	@mkdir -p $(CLOUD_DIR)
	@[ -f $(CLOUD_DIR)/GPT-4o-System-Card.pdf ] \
	    || $(CURL) -o $(CLOUD_DIR)/GPT-4o-System-Card.pdf $(GPT4O_CARD_URL)
	@[ -f $(CLOUD_DIR)/Claude-Haiku-4-5-System-Card.pdf ] \
	    || $(CURL) -o $(CLOUD_DIR)/Claude-Haiku-4-5-System-Card.pdf $(HAIKU_45_CARD_URL)
	@[ -f $(CLOUD_DIR)/Claude-Sonnet-4-5-System-Card.pdf ] \
	    || $(CURL) -o $(CLOUD_DIR)/Claude-Sonnet-4-5-System-Card.pdf $(SONNET_45_CARD_URL)
	@[ -f $(CLOUD_DIR)/Claude-Opus-4-7-System-Card.pdf ] \
	    || $(CURL) -o $(CLOUD_DIR)/Claude-Opus-4-7-System-Card.pdf $(OPUS_47_CARD_URL)

# ── Housekeeping ───────────────────────────────────────────────────────────────

bitnet-clean:
	rm -rf $(BITNET_DIR)

clean:
	rm -rf results/plots
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

nuke: clean
	$(POETRY) env remove --all
