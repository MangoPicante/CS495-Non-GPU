# Cross-architecture benchmark container for the Phase 6 sweep.
#
# One image builds for x86 (c5.xlarge, c6a.xlarge) and ARM (c7g.xlarge)
# via `docker buildx --platform=...`.  Reproduces the full CPU inference
# stack from source at the pinned commits used on the local Windows build,
# then leaves `make benchmark` ready to run.
#
# Build (locally or in CI):
#   docker buildx build --platform=linux/amd64 -t cs495-non-gpu:x86 .
#   docker buildx build --platform=linux/arm64 -t cs495-non-gpu:arm .
#
# Run (mounts results back to the host):
#   docker run --rm -v "$(pwd)/results:/capstone/results" cs495-non-gpu:x86 \
#       make benchmark
#
# Notes on cross-arch behaviour:
#   * On x86 BitNet builds with the TL2 kernel (same as our local Windows
#     reference); on ARM BitNet builds with the TL1 kernel.  Cross-arch
#     numbers compare different kernel implementations, not just ISAs —
#     called out in REPORT.md §6.1 once the data lands.
#   * All three patches in patches/ apply.  Originally
#     `bitnet-clangcl-const.patch` was assumed to be ClangCL-only, but
#     Linux clang-18 also rejects the non-const initializer at
#     src/ggml-bitnet-mad.cpp:811 — verified empirically during the
#     Docker-build bring-up.  Apply path differs: the const patch
#     targets BitNet's own src/ directory; the chrono patches target
#     the 3rdparty/llama.cpp/ submodule.

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BITNET_COMMIT=01eb415772c342d9f20dc42772f1583ae1e5b102
ARG QWEN_LLAMACPP_COMMIT=1e5ad35d560b90a8ac447d149c8f8447ae1fcaa0

# ── 1. System dependencies ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git build-essential cmake \
        software-properties-common gnupg lsb-release \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11 (Ubuntu 22.04 ships 3.10; deadsnakes provides 3.11 wheels).
# Pinned to 3.11 to match the project's pyproject.toml and avoid the
# numpy 1.26-no-wheels-for-3.13+ issue documented in PLAN.md.
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python

# Clang 18 from apt.llvm.org — exactly what microsoft/BitNet's README
# recommends for non-Windows builds.  Works on both x86 and ARM hosts.
RUN bash -c "$(wget -qO - https://apt.llvm.org/llvm.sh)" -- 18 all && \
    ln -sf /usr/bin/clang-18    /usr/local/bin/clang && \
    ln -sf /usr/bin/clang++-18  /usr/local/bin/clang++

# Poetry — same dependency-manager the project uses on Windows.
RUN curl -sSL https://install.python-poetry.org | python3.11 - && \
    ln -sf /root/.local/bin/poetry /usr/local/bin/poetry

# Hugging Face CLI for model downloads (matches `pip install
# huggingface_hub[cli]` documented in README.md prerequisites).
RUN pip install --no-cache-dir "huggingface_hub[cli]"

# ── 2. Build microsoft/BitNet at pinned commit ───────────────────────────────
ENV BITNET_DIR=/Models/BitNet
RUN git clone --recursive https://github.com/microsoft/BitNet.git ${BITNET_DIR} && \
    cd ${BITNET_DIR} && \
    git checkout ${BITNET_COMMIT} && \
    git submodule update --init --recursive

# Apply all three patches from patches/ (see header for the
# Linux-vs-Windows context).  The const patch targets BitNet's own
# src/ggml-bitnet-mad.cpp; the chrono patches target the bundled
# 3rdparty/llama.cpp submodule.
COPY patches/llama-chrono.patch patches/llama-chrono-examples.patch \
     patches/bitnet-clangcl-const.patch /tmp/patches/
RUN cd ${BITNET_DIR}/3rdparty/llama.cpp && \
    git apply /tmp/patches/llama-chrono.patch && \
    git apply /tmp/patches/llama-chrono-examples.patch
RUN cd ${BITNET_DIR} && \
    git apply /tmp/patches/bitnet-clangcl-const.patch

# Split into three RUN layers so each caches independently — the build
# step is the one most likely to need iteration, and we don't want to
# re-download torch wheels or the 1.7 GiB GGUF every cycle.
RUN cd ${BITNET_DIR} && \
    pip install --no-cache-dir -r requirements.txt --only-binary=:all: && \
    pip install --no-cache-dir 3rdparty/llama.cpp/gguf-py

RUN cd ${BITNET_DIR} && \
    huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
        --local-dir models/BitNet-b1.58-2B-4T

# BitNet's setup_env.py generates the TL kernel headers (TL2 on x86,
# TL1 on ARM) and runs cmake/clang under the hood.  Cross-platform by
# design per the README.  Dump logs/compile.log on failure so debugging
# Linux-build issues doesn't require an interactive shell into a
# partially-built image.
RUN cd ${BITNET_DIR} && \
    (CC=clang CXX=clang++ python3 setup_env.py \
        -md models/BitNet-b1.58-2B-4T -q i2_s \
     || (echo "===== compile.log =====" && cat logs/compile.log && false))

# ── 3. Build ggml-org/llama.cpp at pinned commit ─────────────────────────────
ENV QWEN_DIR=/Models/Qwen
RUN git clone https://github.com/ggml-org/llama.cpp.git ${QWEN_DIR}/llama.cpp && \
    cd ${QWEN_DIR}/llama.cpp && \
    git checkout ${QWEN_LLAMACPP_COMMIT} && \
    CC=clang CXX=clang++ cmake -B build \
        -DLLAMA_CURL=OFF \
        -DGGML_NATIVE=ON && \
    cmake --build build --config Release -j

# All three Qwen GGUFs (Q8_0, Q4_K_M, Q2_K) from the first-party
# HuggingFace repo — bit-identical to the local Windows benchmark.
RUN cd ${QWEN_DIR} && \
    huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
        qwen2.5-1.5b-instruct-q8_0.gguf \
        qwen2.5-1.5b-instruct-q4_k_m.gguf \
        qwen2.5-1.5b-instruct-q2_k.gguf \
        --local-dir .

# Llama-3.2-1B-Instruct Q4_K_M — second model family (community quant
# from bartowski; Meta does not ship GGUFs).  Reuses the upstream
# llama.cpp build from §3 — no separate build needed.
ENV LLAMA_DIR=/Models/Llama
RUN mkdir -p ${LLAMA_DIR} && \
    cd ${LLAMA_DIR} && \
    huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
        Llama-3.2-1B-Instruct-Q4_K_M.gguf \
        --local-dir .

# ── 4. Install project Python deps via Poetry ────────────────────────────────
WORKDIR /capstone

# Copy just the dep manifests first so this layer caches independently
# of code changes.
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true && \
    poetry install --only main --no-root

# Now copy the rest of the project.  Anything in .dockerignore (results/,
# .git/, etc.) is excluded.
COPY Makefile ./
COPY scripts/ ./scripts/
COPY patches/ ./patches/

# ── 5. Runtime defaults ──────────────────────────────────────────────────────
# `make benchmark` calls metrics_tracker.py which falls back to
# `${BITNET_DIR}/build/bin/llama-bench` (no Release/, no .exe) on
# non-Windows hosts, so the existing Makefile targets work unmodified.
VOLUME /capstone/results

# Default action: dump the Makefile help so a bare `docker run` prints
# the runnable targets instead of failing silently.
CMD ["make", "help"]
