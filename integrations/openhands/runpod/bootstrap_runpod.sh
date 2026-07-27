#!/usr/bin/env bash
# =============================================================================
# bootstrap_runpod.sh — bare runpod pod  ->  ready to run the fair-parity A/B.
#
# This is the step BEFORE 00_setup_a100.sh. That script assumes the pie source
# is already on the box (its section [2] exits if not) and assumes a toolchain,
# NCCL, and Python >=3.12 are present. On a fresh runpod pod none of that is
# true, and the downstream scripts (10_/30_/run_litellm_baseline_fair.sh)
# default PIE_VENV/HF_HOME to $HOME paths while 00_setup_a100.sh puts them on
# $WORK — a mismatch that silently re-downloads 60 GB of weights or reports
# "no vllm". So this script:
#
#   [0] preflight   — GPU arch/mem, disk on the persistent volume, nvcc
#   [1] toolchain   — build-essential / cmake>=3.23 / ninja / rust + wasm target
#   [2] python 3.12 — the harness requires >=3.12; provisioned via uv if absent
#   [3] source      — clone/update the branch carrying the §4 + bug-C fixes
#   [4] NCCL        — REQUIRED by driver/cuda/CMakeLists.txt even for 1 GPU
#   [5] env file    — ONE $WORK/pie-bench-env.sh that every later script sources
#   [6] venvs       — vLLM venv + harness venv, both on the persistent volume
#   [7] delegate    — hand off to 00_setup_a100.sh for the builds + model
#
# Idempotent and resumable: every step skips work that is already done, so
# re-running after a dropped SSH session or an OOM picks up where it stopped.
#
# Usage (run it under tmux — the CUDA driver build takes 30-60 min):
#     tmux new -s pie
#     bash bootstrap_runpod.sh
#
# Useful overrides:
#     WORK=/workspace          persistent volume (must survive pod restart)
#     REPO_REF=<branch>        git ref to build
#     VLLM_VERSION=0.11.0      pin vLLM instead of taking latest
#     SKIP_MODEL=1             skip the ~60 GB weight download
#     SKIP_APT=1               skip apt (non-root pods / preinstalled toolchain)
# =============================================================================
set -euo pipefail

WORK=${WORK:-/workspace}
REPO_URL=${REPO_URL:-https://github.com/YangLiuWillow/pie.git}
REPO_REF=${REPO_REF:-openhands-integration-updated}
# STORAGE SPLIT — the single most important thing in this script.
#
# On runpod $WORK is typically a MooseFS network volume (check: `df -h $WORK`
# shows mfs#...runpod.net). It has effectively unlimited space and survives pod
# restarts, but every file operation is a network round-trip. Small-file-heavy
# workloads do not merely run slow on it — they stall. Observed: `uv pip
# install` of 67 small wheels parked 43 parallel downloads at ~15 KiB each and
# never progressed, while curl to the same CDN ran at 125 MB/s. uv streams
# downloads into UV_CACHE_DIR before linking them into the venv, so a cache on
# MooseFS stalls the install no matter where the venv itself lives.
#
#   $FAST (local container disk) : caches, venvs, cargo target/ — small files,
#                                  hot, cheap to rebuild. Lost on pod stop.
#   $WORK (persistent volume)    : the repo and the ~60 GB model weights —
#                                  large sequential files, expensive to refetch.
#
# The container disk is finite (60 GB is the runpod default; 100 GB is the
# recommendation in TEST_PLAN.md §10), so this trades rebuild-on-restart for a
# setup that actually completes.
FAST=${FAST:-/root}
PIE_SRC=${PIE_SRC:-$WORK/pie}
PIE_VENV=${PIE_VENV:-$FAST/venvs/pie-vllm}
HF_HOME=${HF_HOME:-$WORK/hf-cache}          # weights: big, sequential, persist
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
CARGO_TARGET=${CARGO_TARGET:-$FAST/pie-target}
CPM_SOURCE_CACHE=${CPM_SOURCE_CACHE:-$FAST/.cpm-cache}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-$FAST/.pip-cache}
UV_CACHE_DIR=${UV_CACHE_DIR:-$FAST/.uv-cache}
# Resolve the harness dependency graph as of when openhands-sdk 1.21.1 shipped
# (2026-05-08). It declares 14 open-ended floors (fastmcp>=3.0.0,
# litellm>=1.83.7, pydantic>=2.12.5, ...); resolving those at today's newest
# gives a set the SDK was never tested against — litellm 1.93.0 fails to import
# its own MessagesInterceptor, fastmcp 3.x moved Client. Pin time, not packages.
HARNESS_EXCLUDE_NEWER=${HARNESS_EXCLUDE_NEWER:-2026-05-15}
# uv defaults to a 30s HTTP timeout, which the multi-hundred-MB wheels in the
# vLLM/torch/CUDA dependency set routinely blow through on a runpod link.
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-900}
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-900}
# uv fetches wheels in parallel by default. On a saturated link that starves
# every stream until each trips its own timeout — observed on runpod as a 90 MB
# wheel failing at a 300s timeout. Capping concurrency fixes this where raising
# the timeout alone does not.
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}
VLLM_VERSION=${VLLM_VERSION:-}
SKIP_MODEL=${SKIP_MODEL:-0}
SKIP_APT=${SKIP_APT:-0}
ENV_FILE=$WORK/pie-bench-env.sh
# The real venv is on $FAST; run_pie_backend.sh hardcodes VENV=$SCRIPT_DIR/.venv
# and is not overridable, so $HARNESS_VENV_LINK is symlinked at it.
HARNESS_VENV=${HARNESS_VENV:-$FAST/venvs/harness}
HARNESS_VENV_LINK=$PIE_SRC/integrations/openhands/.venv

step() { printf '\n\033[1;36m=== [%s] %s\033[0m\n' "$1" "$2"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$1"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }

mkdir -p "$WORK" "$WORK/logs" "$FAST/venvs" "$PIP_CACHE_DIR" "$CPM_SOURCE_CACHE" \
         "$UV_CACHE_DIR" "$CARGO_TARGET" "$HF_HOME"
LOG=$WORK/logs/bootstrap_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "bootstrap log: $LOG"

if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ]; then
    warn "not inside tmux/screen. The CUDA driver build takes 30-60 min and a"
    warn "dropped SSH connection will kill it. Ctrl-C now and run: tmux new -s pie"
    sleep 8
fi

# -----------------------------------------------------------------------------
step 0 "preflight"
# -----------------------------------------------------------------------------
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — is this a GPU pod?"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
ARCH=${CC/./}
[ -n "$ARCH" ] || die "could not read compute capability"
if [ "$ARCH" != "80" ]; then
    warn "compute cap is $CC (sm_$ARCH), not sm_80. The test plan and"
    warn "pie_cuda_native_config_30b_moe_a100.toml assume an A100 80GB SXM."
    warn "Building for sm_$ARCH anyway; re-check the KV/swap sizes in the toml."
fi

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$GPU_MEM" -ge 70000 ] || warn "GPU has ${GPU_MEM} MiB (<80 GB) — the A100 config assumes 80 GB; lower gpu_memory_utilization / kv sizes."

if command -v mountpoint >/dev/null && ! mountpoint -q "$WORK"; then
    warn "$WORK is not a separate mount — it may be ephemeral container disk."
    warn "On runpod the persistent volume must be mounted at $WORK (see TEST_PLAN.md §10)."
fi
WORK_FS=$(df -h "$WORK" 2>/dev/null | tail -1 | awk '{print $1}')
echo "  $WORK filesystem: ${WORK_FS:-unknown}"
case "$WORK_FS" in
    mfs*|*:/*|*nfs*)
        echo "  -> network filesystem detected: caches/venvs/build go on \$FAST ($FAST)."
        echo "     Only the repo and the model weights stay on $WORK." ;;
esac
AVAIL=$(df -BG --output=avail "$WORK" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "  free on $WORK: ${AVAIL:-?} GB"
FAST_AVAIL=$(df -BG --output=avail "$FAST" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "  free on $FAST (local, holds venvs+build): ${FAST_AVAIL:-?} GB"
[ "${FAST_AVAIL:-0}" -ge 55 ] || warn "$FAST has ${FAST_AVAIL:-?} GB. venvs (~18) + cargo target (~30) + CPM (~5) need ~55 GB; raise the pod's container disk or expect a build failure."
[ "${AVAIL:-0}" -ge 100 ] || warn "TEST_PLAN.md §10 wants 100 GB on $WORK for the repo + weights (150-200 GB if also scoring SWE-bench here)."
ROOT_AVAIL=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
echo "  free on / (container disk): ${ROOT_AVAIL:-?} GB"

command -v nvcc >/dev/null && nvcc --version | grep -oE 'release [0-9.]+' \
    || warn "nvcc not on PATH — the CUDA driver build in step [7] will fail. Install a CUDA toolkit matching the driver."

# -----------------------------------------------------------------------------
step 1 "toolchain (build tools, cmake>=3.23, rust + wasm32-wasip2)"
# -----------------------------------------------------------------------------
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
if [ "$SKIP_APT" != "1" ]; then
    export DEBIAN_FRONTEND=noninteractive
    $SUDO apt-get update -qq || warn "apt-get update failed — continuing"
    $SUDO apt-get install -y -qq \
        build-essential git curl ca-certificates pkg-config libssl-dev \
        ninja-build python3-venv unzip || warn "apt-get install failed — continuing"
fi

# The driver CMakeLists require >= 3.23; Ubuntu 22.04 ships 3.22. Prefer a pip
# cmake on $PATH over fighting apt repos.
cmake_ok=0
if command -v cmake >/dev/null; then
    CM=$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    [ "$(printf '%s\n3.23\n' "$CM" | sort -V | head -1)" = "3.23" ] && cmake_ok=1
    echo "  cmake $CM (need >= 3.23)"
fi
if [ "$cmake_ok" -ne 1 ]; then
    echo "  installing cmake via pip (system cmake missing or < 3.23)"
    PIP_CACHE_DIR=$PIP_CACHE_DIR python3 -m pip install --quiet --upgrade "cmake>=3.27" ninja \
        || die "could not install cmake — install one >= 3.23 manually"
    hash -r
    cmake --version | head -1
fi

if ! command -v cargo >/dev/null; then
    echo "  installing rust via rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
fi
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
command -v cargo >/dev/null || die "cargo still not on PATH after rustup install"
# rust-toolchain.toml pins `stable`; add the wasm target for that toolchain.
rustup target add wasm32-wasip2 || warn "could not add wasm32-wasip2 target"
cargo --version

# -----------------------------------------------------------------------------
step 2 "python >= 3.12 (pie-openhands requires-python >=3.12)"
# -----------------------------------------------------------------------------
# 3.12 SPECIFICALLY, not "newest >= 3.12". On 3.13 the vLLM dependency set
# resolves and installs fine, then torch fails at IMPORT time: TorchScript's
# overload parser (torch/_sources.py parse_def -> ast.parse) raises
# IndentationError on torch/nn/modules/rnn.py. Resolution succeeding is not
# evidence the stack works. 3.12 is what vLLM and torch are built against.
# Override with PYTHON_VERSION= / PY= if you know better.
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
PY=${PY:-}
if [ -z "$PY" ] && command -v "python$PYTHON_VERSION" >/dev/null; then
    PY=$(command -v "python$PYTHON_VERSION")
fi
if [ -z "$PY" ]; then
    echo "  python$PYTHON_VERSION not on PATH ($(python3 --version 2>&1) is the default); provisioning it with uv"
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null || die "uv install failed — install python$PYTHON_VERSION manually"
    uv python install "$PYTHON_VERSION"
    PY=$(uv python find "$PYTHON_VERSION")
fi
[ -x "$PY" ] || die "no usable python$PYTHON_VERSION interpreter (got: '$PY')"
"$PY" -c "import sys; sys.exit(0 if sys.version_info[:2] == tuple(int(x) for x in '$PYTHON_VERSION'.split('.')) else 1)" \
    || warn "interpreter is $("$PY" --version), not $PYTHON_VERSION — torch may fail at import on 3.13"
echo "  python: $PY ($("$PY" --version))"

# -----------------------------------------------------------------------------
step 3 "pie source at $PIE_SRC (ref: $REPO_REF)"
# -----------------------------------------------------------------------------
# LOGISTICS note 1 in 00_setup_a100.sh: this ref carries the §4 CUDA fix
# (cuda_memory_planner.cpp output_rows R0->N) and bug C (drain_queues
# starvation). Without the §4 fix any coder-session prefill > 512 tokens faults
# the driver, so do NOT build an older ref.
if [ -d "$PIE_SRC/.git" ]; then
    echo "  existing checkout — fetching $REPO_REF"
    git -C "$PIE_SRC" fetch --depth 1 origin "$REPO_REF"
    git -C "$PIE_SRC" checkout -B "$REPO_REF" FETCH_HEAD
else
    git clone --depth 1 -b "$REPO_REF" "$REPO_URL" "$PIE_SRC"
fi
git -C "$PIE_SRC" --no-pager log -1 --oneline
RUNPOD_DIR=$PIE_SRC/integrations/openhands/runpod
[ -f "$RUNPOD_DIR/00_setup_a100.sh" ] || die "$RUNPOD_DIR/00_setup_a100.sh missing — wrong ref?"

# -----------------------------------------------------------------------------
step 4 "NCCL (find_path/find_library are REQUIRED in driver/cuda/CMakeLists.txt)"
# -----------------------------------------------------------------------------
# Single-GPU runs never call into NCCL (the call sites are gated on tp_size > 1)
# but the CUDA driver always links it, so configure fails without a header+lib.
# server/build.rs honours PIE_NCCL_HOME and handles the wheel layout that ships
# only the versioned libnccl.so.2 with no unversioned symlink.
PIE_NCCL_HOME=${PIE_NCCL_HOME:-}
if [ -z "$PIE_NCCL_HOME" ]; then
    if [ -f /usr/include/nccl.h ] && ls /usr/lib/x86_64-linux-gnu/libnccl.so* >/dev/null 2>&1; then
        echo "  using system NCCL (/usr/include/nccl.h)"
    else
        # torch on the runpod image usually already carries the nccl wheel.
        WHEEL_NCCL=$("$PY" - <<'PY' 2>/dev/null || true
import importlib.util, pathlib
spec = importlib.util.find_spec("nvidia.nccl")
if spec and spec.submodule_search_locations:
    p = pathlib.Path(list(spec.submodule_search_locations)[0])
    if (p / "include" / "nccl.h").is_file():
        print(p)
PY
)
        if [ -z "$WHEEL_NCCL" ]; then
            echo "  no system or wheel NCCL — installing nvidia-nccl-cu12"
            PIP_CACHE_DIR=$PIP_CACHE_DIR "$PY" -m pip install --quiet nvidia-nccl-cu12
            WHEEL_NCCL=$("$PY" -c 'import importlib.util,pathlib;print(list(importlib.util.find_spec("nvidia.nccl").submodule_search_locations)[0])')
        fi
        [ -f "$WHEEL_NCCL/include/nccl.h" ] || die "no nccl.h under $WHEEL_NCCL — install libnccl-dev or set PIE_NCCL_HOME"
        PIE_NCCL_HOME=$WHEEL_NCCL
        echo "  using wheel NCCL: $PIE_NCCL_HOME"
    fi
fi

# -----------------------------------------------------------------------------
step 5 "env file -> $ENV_FILE"
# -----------------------------------------------------------------------------
# Downstream scripts default PIE_VENV/HF_HOME to $HOME paths while
# 00_setup_a100.sh writes them under $WORK. Sourcing this file before every
# later command is what keeps the two in agreement.
cat > "$ENV_FILE" <<EOF
# generated by bootstrap_runpod.sh — source before every run:
#   source $ENV_FILE
export WORK=$WORK
export FAST=$FAST
export PIE_SRC=$PIE_SRC
export RUNPOD_DIR=$PIE_SRC/integrations/openhands/runpod
export HARNESS_DIR=$PIE_SRC/integrations/openhands
export PIE_VENV=$PIE_VENV
export HARNESS_VENV=$HARNESS_VENV
export HF_HOME=$HF_HOME
export MODEL=$MODEL
export PIE_BIN=$PIE_SRC/target/release/pie
export CPM_SOURCE_CACHE=$CPM_SOURCE_CACHE
export PIP_CACHE_DIR=$PIP_CACHE_DIR
export UV_CACHE_DIR=$UV_CACHE_DIR
export UV_HTTP_TIMEOUT=$UV_HTTP_TIMEOUT
export PIP_DEFAULT_TIMEOUT=$PIP_DEFAULT_TIMEOUT
export UV_CONCURRENT_DOWNLOADS=$UV_CONCURRENT_DOWNLOADS
export CMAKE_CUDA_ARCHITECTURES=$ARCH
export PIE_PORTABLE_CUDA_ARCH=$ARCH
$([ -n "$PIE_NCCL_HOME" ] && echo "export PIE_NCCL_HOME=$PIE_NCCL_HOME")
export PATH="\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH"
[ -f "\$HOME/.cargo/env" ] && . "\$HOME/.cargo/env"
EOF
cat "$ENV_FILE"
# shellcheck disable=SC1090
. "$ENV_FILE"

# -----------------------------------------------------------------------------
step 6 "venvs (vLLM + harness, both on the persistent volume)"
# -----------------------------------------------------------------------------
mkvenv() {  # mkvenv <path>
    [ -x "$1/bin/python" ] && return 0
    # --seed so bin/pip exists (uv venvs omit it by default and some downstream
    # scripts reach for $VENV/bin/pip directly).
    if command -v uv >/dev/null; then uv venv --seed --python "$PY" "$1"; else "$PY" -m venv "$1"; fi
}
pipinstall() {  # pipinstall <venv> <args...>
    local v=$1; shift
    local attempt
    # Retry: these are multi-GB downloads over a link that drops them. Both
    # tools resume from cache ($UV_CACHE_DIR / $PIP_CACHE_DIR on $WORK), so a
    # retry re-fetches only what actually failed.
    for attempt in 1 2 3; do
        if command -v uv >/dev/null; then
            # Halve concurrency each retry — a link that starved 4 parallel
            # streams may still carry 2, then 1.
            UV_CONCURRENT_DOWNLOADS=$(( UV_CONCURRENT_DOWNLOADS > 1 ? UV_CONCURRENT_DOWNLOADS / attempt : 1 )) \
                VIRTUAL_ENV=$v uv pip install "$@" && return 0
        else
            "$v/bin/pip" install -q "$@" && return 0
        fi
        warn "install attempt $attempt/3 failed ($*) — retrying in 10s"
        sleep 10
    done
    # Last resort: pip is serial and a different HTTP stack, so it sometimes
    # completes a download set that uv cannot.
    if [ -x "$v/bin/pip" ]; then
        warn "falling back to pip (serial) for: $*"
        "$v/bin/pip" install "$@" && return 0
    fi
    die "install failed after 3 uv attempts + pip fallback: $*"
}

# Gate on the PACKAGE importing, not on the venv directory existing. A venv is
# created before its (multi-GB) installs finish, so a network failure mid-install
# leaves a venv that exists but is empty — and a directory-existence guard would
# then skip the retry on re-run and fail at the import check instead.
has_pkg() { [ -x "$1/bin/python" ] && "$1/bin/python" -c "import $2" >/dev/null 2>&1; }

# A venv built on the wrong interpreter can never be repaired by reinstalling
# into it — packages are version-keyed by path. Discard it so it is rebuilt.
# (This is the recovery path for a venv created on 3.13 before the pin above.)
drop_stale_venv() {
    [ -x "$1/bin/python" ] || return 0
    local got; got=$("$1/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo unknown)
    [ "$got" = "$PYTHON_VERSION" ] && return 0
    warn "$1 was built on python $got, need $PYTHON_VERSION — removing and rebuilding"
    rm -rf "$1"
}
drop_stale_venv "$PIE_VENV"
drop_stale_venv "$HARNESS_VENV"

if ! has_pkg "$PIE_VENV" vllm; then
    echo "  provisioning vLLM venv at $PIE_VENV"
    mkvenv "$PIE_VENV"
    pipinstall "$PIE_VENV" --upgrade pip
    # Pin VLLM_VERSION for a reproducible baseline; unpinned takes latest.
    if [ -n "$VLLM_VERSION" ]; then pipinstall "$PIE_VENV" "vllm==$VLLM_VERSION"
    else warn "installing unpinned vllm — set VLLM_VERSION=x.y.z to make the baseline reproducible"; pipinstall "$PIE_VENV" vllm; fi
    pipinstall "$PIE_VENV" huggingface_hub
fi
"$PIE_VENV/bin/python" -c 'import vllm; print("  vLLM", vllm.__version__)'

if ! has_pkg "$HARNESS_VENV" pie_openhands; then
    echo "  provisioning harness venv at $HARNESS_VENV (resolved as of $HARNESS_EXCLUDE_NEWER)"
    mkvenv "$HARNESS_VENV"
    pipinstall "$HARNESS_VENV" --upgrade pip
    # litellm is a direct import in pie_openhands/llm.py but is not declared in
    # its pyproject, so name it explicitly — bounded by --exclude-newer rather
    # than pinned, which is what keeps it consistent with openhands-sdk.
    pipinstall "$HARNESS_VENV" --exclude-newer "$HARNESS_EXCLUDE_NEWER" \
        -e "$PIE_SRC/integrations/openhands" litellm
fi
"$HARNESS_VENV/bin/python" -c 'import pie_openhands, openhands.sdk; print("  harness ok")'

# run_pie_backend.sh looks for integrations/openhands/.venv unconditionally.
if [ ! -e "$HARNESS_VENV_LINK" ] || [ "$(readlink -f "$HARNESS_VENV_LINK")" != "$(readlink -f "$HARNESS_VENV")" ]; then
    rm -rf "$HARNESS_VENV_LINK"
    ln -sfn "$HARNESS_VENV" "$HARNESS_VENV_LINK"
    echo "  linked $HARNESS_VENV_LINK -> $HARNESS_VENV"
fi

# Same trick for the Rust build tree: CARGO_TARGET_DIR is global and would also
# redirect the wasm inferlet build, breaking the path run_pie_backend.sh expects.
# A symlink keeps target/ where every script looks while the bytes land on $FAST.
if [ ! -L "$PIE_SRC/target" ]; then
    [ -d "$PIE_SRC/target" ] && rm -rf "$PIE_SRC/target"
    ln -sfn "$CARGO_TARGET" "$PIE_SRC/target"
    echo "  linked $PIE_SRC/target -> $CARGO_TARGET"
fi

# -----------------------------------------------------------------------------
step 7 "hand off to 00_setup_a100.sh (pie sm_$ARCH build, wasm inferlet, model)"
# -----------------------------------------------------------------------------
# Its venv sections are no-ops now (both exist); it does the two cargo builds
# and the model download. Everything it reads is exported above.
export SKIP_MODEL
cd "$RUNPOD_DIR"
bash 00_setup_a100.sh

[ -x "$PIE_SRC/target/release/pie" ] || die "pie binary not built — check the log above"
WASM=$PIE_SRC/inferlets/openhands-coder-session/target/wasm32-wasip2/release/openhands_coder_session.wasm
[ -f "$WASM" ] || die "coder-session wasm not built: $WASM"
echo "  pie:  $PIE_SRC/target/release/pie"
echo "  wasm: $WASM"
"$PIE_SRC/target/release/pie" driver list || warn "'pie driver list' failed — check the CUDA build"

# `driver list` proves cuda_native was COMPILED IN; it says nothing about which
# GPU arch the kernels were compiled FOR. That is the exact failure the TEST_PLAN
# prereq warns about ("do not copy the sm_120 Blackwell binary"), and it would
# otherwise surface as a runtime fault well into an arm. Check the fatbin.
if command -v cuobjdump >/dev/null; then
    ELF_ARCHS=$(cuobjdump --list-elf "$PIE_SRC/target/release/pie" 2>/dev/null \
                | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')
    if [ -n "$ELF_ARCHS" ]; then
        echo "  embedded GPU code: $ELF_ARCHS(this GPU: sm_$ARCH)"
        case " $ELF_ARCHS" in
            *" sm_$ARCH "*) ;;
            *) warn "binary carries [$ELF_ARCHS] but this GPU is sm_$ARCH — it will fault at runtime. Rebuild with CMAKE_CUDA_ARCHITECTURES=$ARCH." ;;
        esac
    else
        echo "  (cuobjdump found no embedded ELF — kernels may be JIT/PTX-only)"
    fi
else
    echo "  (cuobjdump not on PATH — skipping arch verification)"
fi

# -----------------------------------------------------------------------------
step 8 "done"
# -----------------------------------------------------------------------------
cat <<EOF

Bootstrap complete. Log: $LOG

Every later shell must start with:
    source $ENV_FILE
    cd $RUNPOD_DIR

Then follow TEST_PLAN.md §6:
    bash 11_autotune_moe.sh                                  # once, for the 'fair' tier
    ARM=pie bash 30_ab_run.sh                                # Pie arm (KV_VERIFY=1)
    VLLM_TIER=crippled    ARM=litellm bash 30_ab_run.sh
    VLLM_TIER=graphs-only ARM=litellm bash 30_ab_run.sh
    VLLM_TIER=fair        ARM=litellm bash 30_ab_run.sh
    python summarize_ab.py pie=../predictions/ab_a100_pie_*.jsonl \\
      litellm-crippled=../predictions/ab_a100_litellm_crippled_*.jsonl \\
      litellm-graphs-only=../predictions/ab_a100_litellm_graphs-only_*.jsonl \\
      litellm-fair=../predictions/ab_a100_litellm_fair_*.jsonl

Record per arm (TEST_PLAN.md §6): the assert_vllm_fair.sh banner, the
summarize_ab.py row, and for the Pie arm the KV-reuse % + kv-verify error
count (must be 0).
EOF
