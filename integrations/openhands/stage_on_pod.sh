#!/usr/bin/env bash
# stage_on_pod.sh — run ON a RunPod staging pod (network volume mounted) to
# finish preparing the volume for OpenHands↔Pie cuda_native jobs.
#
# Split of responsibilities (see docs/RUNPOD_STAGING.md):
#   - The repo working tree is rsync'd onto the volume FROM the HPC login node
#     BEFORE this runs (brings the exact tree incl. the prebuilt inferlet .wasm
#     and any uncommitted harness edits — a git clone would miss both).
#   - This script does the two things best done on the pod's fast cloud network
#     with a pod-native toolchain: (1) download the model into the volume HF
#     cache, (2) rebuild the OpenHands harness venv (the HPC venv's interpreter
#     symlink is not portable). Both steps are idempotent — safe to re-run.
#
# Env (all optional; defaults match runpod_submit.sh):
#   MOUNT=/workspace          volume mountpoint
#   WORKDIR=$MOUNT/pie        rsync'd repo root
#   HF_HOME=$MOUNT/hf         model cache on the volume
#   MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
set -euo pipefail

MOUNT="${MOUNT:-/workspace}"
WORKDIR="${WORKDIR:-$MOUNT/pie}"
HF_HOME="${HF_HOME:-$MOUNT/hf}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
OH="$WORKDIR/integrations/openhands"
WASM="$WORKDIR/inferlets/openhands-coder-session/target/wasm32-wasip2/release/openhands_coder_session.wasm"

echo "== stage_on_pod =="
echo "  MOUNT=$MOUNT  WORKDIR=$WORKDIR  HF_HOME=$HF_HOME  MODEL=$MODEL"

# ── 0. Preconditions: the rsync must have run, and the prebuilt wasm shipped ──
[ -f "$OH/pyproject.toml" ] || { echo "ERROR: $OH/pyproject.toml missing — rsync the repo onto the volume first"; exit 1; }
[ -f "$WORKDIR/client/python/pyproject.toml" ] || { echo "ERROR: client/python missing — the venv needs '-e ../../client/python'"; exit 1; }
if [ ! -f "$WASM" ]; then
  echo "ERROR: prebuilt inferlet wasm missing at:"; echo "  $WASM"
  echo "The runtime image has no Rust toolchain, so this MUST be shipped by rsync."
  echo "Build it on the login node: (cd inferlets/openhands-coder-session && cargo build --release --target wasm32-wasip2)"
  exit 1
fi
echo ">> preconditions OK (repo + client + wasm present)"

# ── 1. Download the model into the volume HF cache (idempotent/resumable) ─────
export HF_HOME
mkdir -p "$HF_HOME"
echo ">> ensuring huggingface_hub CLI"
python3 -m pip install --quiet --upgrade "huggingface_hub[hf_transfer]" 2>/dev/null \
  || pip install --quiet --upgrade "huggingface_hub[hf_transfer]"
echo ">> downloading $MODEL → $HF_HOME (skips files already present)"
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -m huggingface_hub.commands.huggingface_cli \
  download "$MODEL" --exclude "*.pth" "original/*" >/dev/null
echo ">> model present"

# ── 2. Rebuild the OpenHands harness venv on the pod ─────────────────────────
# The HPC venv's bin/python symlinks an EasyBuild path that does not exist here,
# so it cannot be relocated — rebuild native to the pod. Reuse if already valid.
cd "$OH"
if [ -x .venv/bin/python ] && .venv/bin/python -c "import openhands, pie_client" 2>/dev/null; then
  echo ">> venv already present and importable — skipping rebuild"
else
  echo ">> building .venv (python3 -m venv + editable installs)"
  rm -rf .venv
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install -e '.[dev]'
  .venv/bin/pip install -e ../../client/python
  echo ">> venv built"
fi

# ── 3. Verify the whole runtime surface the job will touch ───────────────────
echo "== verify =="
command -v pie >/dev/null && pie driver list | grep -q cuda_native \
  && echo "  [ok] pie binary has cuda_native driver" \
  || { echo "  [FAIL] pie binary missing or not built with driver-cuda (image problem)"; exit 1; }
.venv/bin/python - <<'PY'
import importlib, sys
for m in ("openhands", "pie_client", "litellm"):
    try: importlib.import_module(m); print(f"  [ok] import {m}")
    except Exception as e: print(f"  [FAIL] import {m}: {e}"); sys.exit(1)
PY
snap="$HF_HOME/hub/models--${MODEL//\//--}/snapshots"
[ -d "$snap" ] && [ -n "$(ls -A "$snap" 2>/dev/null)" ] \
  && echo "  [ok] model snapshot at $snap" \
  || { echo "  [FAIL] no model snapshot under $snap"; exit 1; }
[ -f "$WASM" ] && echo "  [ok] inferlet wasm present"

echo ""
echo "== staging complete — volume ready =="
echo "   Run a job (from the HPC login node) with e.g.:"
echo "     PIE_BIN=/usr/local/bin/pie HF_HOME=$HF_HOME \\"
echo "     CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml \\"
echo "     MODEL=$MODEL BACKEND=pie-session \\"
echo "       ./runpod_submit.sh -- bash integrations/openhands/run_pie_backend.sh \\"
echo "         --instance-id django__django-13028 --python-tool-parser"
