#!/usr/bin/env bash
# audio.cpp bring-up on Jetson Orin (Nano 8GB / NX 16GB, JetPack 7.2, CUDA 13.2, SM 8.7).
# No root required. Run directly on-device: bash bringup.sh
set -euo pipefail

WORKDIR="${HOME}/audio-cpp-jetson"
REPO_URL="https://github.com/0xShug0/audio.cpp"
CUDA_ROOT="/usr/local/cuda"
LOG="${WORKDIR}/bringup.log"

mkdir -p "${WORKDIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "=== $(date -u +%FT%TZ) audio.cpp bringup starting on $(hostname) ==="

# --- Preflight: CUDA toolkit present and >= 12.0 ---
if [ ! -x "${CUDA_ROOT}/bin/nvcc" ]; then
    echo "FATAL: nvcc not found at ${CUDA_ROOT}/bin/nvcc" >&2
    exit 1
fi
CUDA_VER="$("${CUDA_ROOT}/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
CUDA_MAJOR="${CUDA_VER%%.*}"
echo "CUDA toolkit version: ${CUDA_VER}"
if [ "${CUDA_MAJOR}" -lt 12 ]; then
    echo "FATAL: audio.cpp requires CUDA 12.0+, found ${CUDA_VER} (JetPack 5.x/CUDA 11.x is too old)" >&2
    exit 1
fi

# --- Clone or update ---
if [ -d "${WORKDIR}/audio.cpp/.git" ]; then
    echo "Repo exists, pulling latest main..."
    git -C "${WORKDIR}/audio.cpp" fetch origin main
    git -C "${WORKDIR}/audio.cpp" checkout main
    git -C "${WORKDIR}/audio.cpp" pull --ff-only origin main
else
    git clone "${REPO_URL}" "${WORKDIR}/audio.cpp"
fi
cd "${WORKDIR}/audio.cpp"

echo "audio.cpp commit: $(git rev-parse --short HEAD)"
echo "ggml commit (vendored): $(git log -1 --format=%h -- external/ggml 2>/dev/null || echo unknown)"

# Note: the known aarch64/GCC compile blocker (missing <cstddef> in noise.h,
# incomplete json::Value type in a std::unordered_map) was fixed and merged
# upstream in commit 92fd23a ("Fix JSON portability and add native CI"), well
# before this script's target commit — no patch to apply here. If a *new*
# aarch64 compile error shows up, it is genuinely new; check issue trackers
# before re-deriving a fix from scratch.

# --- Configure: explicit SM 8.7 real-arch build, NOT optional ---
# Confirmed 2026-08-19 by an actual local build: audio.cpp's own top-level
# CMakeLists.txt calls enable_language(CUDA) itself (before ggml's
# subdirectory is even processed), which makes CMake compute its own default
# CMAKE_CUDA_ARCHITECTURES right there. That means ggml's own carefully
# commented fallback-architecture logic (native-detect, or a list including
# 80-virtual PTX) NEVER RUNS in a real audio.cpp build — it only fires when
# CMAKE_CUDA_ARCHITECTURES is still undefined by the time ggml's CMake checks
# for it, which it no longer is. Observed on this laptop: with the flag
# omitted, CMake defaulted to bare "75" (Turing) even though the actual GPU
# (RTX 5060, sm_120) was correctly autodetected as CMAKE_CUDA_ARCHITECTURES_NATIVE
# — the autodetected value was computed but never used. Any Ampere-or-newer-gated
# optimized kernel path in ggml-cuda (__CUDA_ARCH__ >= 800 code) would silently
# compile OUT under that default, not just run unoptimized via PTX JIT. So the
# explicit 87-real below is load-bearing, not a safety margin.
# -DGGML_CUDA_NO_VMM=ON: on Orin Nano 8GB specifically, several model families crash
# with "CUDA error" at cuMemAddressReserve(...CUDA_POOL_VMM_MAX_SIZE...) in ggml's CUDA
# pool allocator (ggml-cuda.cu) -- it tries to reserve a 32GB virtual address range
# (1ull<<35) for its VMM pool, which fails on the Nano's smaller unified-memory address
# space even though it never needs to be physically backed. Confirmed on real hardware
# 2026-08-20: dots_tts and vibevoice both crash with VMM on, both run clean with it off.
# NOT a safe no-op elsewhere, corrected 2026-08-20: A/B tested on Orin NX 16GB (bs_roformer,
# 3 runs each way) -- 18.66s mean with VMM on vs 19.01s with it off, a real ~1.9% slowdown
# (means differ by 2-4x each side's stdev, not noise). So: only pass this flag when
# targeting the Nano specifically -- gate on total system RAM as a reliable proxy, since
# the Nano and NX report distinctly different totals and there's no other clean signal
# available before the CUDA device itself is queried.
EXTRA_CUDA_FLAGS=()
TOTAL_RAM_KB="$(awk '/MemTotal/{print $2}' /proc/meminfo)"
if [ "${TOTAL_RAM_KB}" -lt 10000000 ]; then
    echo "Detected ~8GB RAM (Nano-class board) -- adding -DGGML_CUDA_NO_VMM=ON"
    EXTRA_CUDA_FLAGS+=("-DGGML_CUDA_NO_VMM=ON")
else
    echo "Detected >10GB RAM (NX-class board) -- leaving CUDA VMM enabled (no measured benefit disabling it here, and it costs ~2% throughput)"
fi
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DENGINE_ENABLE_CUDA=ON \
    -DCUDAToolkit_ROOT="${CUDA_ROOT}" \
    -DCMAKE_CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="87-real" \
    "${EXTRA_CUDA_FLAGS[@]}"

cmake --build build -j"$(nproc)" --target audiocpp_cli 2>&1 | tee "${WORKDIR}/build_$(date -u +%Y%m%dT%H%M%SZ).log"

BIN="$(find build -iname 'audiocpp_cli*' -type f -executable | head -1)"
if [ -z "${BIN}" ]; then
    echo "FATAL: build finished but audiocpp_cli binary not found" >&2
    exit 1
fi
echo "=== BUILD OK: ${BIN} ==="
echo "$(date -u +%FT%TZ) build succeeded, commit $(git rev-parse --short HEAD)" >> "${WORKDIR}/bringup_status.txt"

# --- Runtime env for the smoke test / any subsequent invocation ---
# Upstream ggml/llama.cpp issue #19219 root-caused a CUDA command-buffer
# deadlock on Jetson to CUDA_SCALE_LAUNCH_QUEUES=4x interacting badly with
# the Jetson CUDA stack (NVIDIA-confirmed driver/JetPack-level bug, fix
# promised in "the next JetPack release" as of that report). It's MoE-path
# specific and audio.cpp's models are dense, so it likely doesn't apply here
# — but CUDA_DEVICE_MAX_CONNECTIONS=1 is the documented workaround and costs
# nothing to set as a precaution before JetPack 7.2 is confirmed to include
# the fix. Export it for whoever runs audiocpp_cli next in this shell.
export CUDA_DEVICE_MAX_CONNECTIONS=1
echo "Set CUDA_DEVICE_MAX_CONNECTIONS=1 (precaution against upstream ggml/llama.cpp#19219-class command-buffer deadlock)"
