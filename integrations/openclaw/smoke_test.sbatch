#!/usr/bin/env bash
#SBATCH --job-name=pie-oc-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_h200
#SBATCH --qos=normal
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openclaw/logs/smoke_%j.out

set -euo pipefail

ROOT=/nfs/roberts/project/pi_ql324/ly337/pie
OPENCLAW_DIR="$ROOT/integrations/openclaw"
cd "$OPENCLAW_DIR"

module load nodejs/24.12.0-GCCcore-13.3.0

export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
export PIE_BIN="$ROOT/target/release/pie"
export PIE_WASM="$ROOT/inferlets/openclaw-chat/target/wasm32-wasip2/release/openclaw_chat.wasm"
export PIE_MANIFEST="$ROOT/inferlets/openclaw-chat/Pie.toml"
export PIE_CONFIG="$OPENCLAW_DIR/tests/fixtures/pie_dummy_config.toml"

echo "=== Pie binary ==="
ls -l "$PIE_BIN"
"$PIE_BIN" --version || true

echo "=== WASM binary ==="
ls -l "$PIE_WASM"

echo "=== Node version ==="
node --version

echo "=== HF cache ==="
echo "HF_HOME=$HF_HOME"

echo "=== Running unit tests ==="
node --test tests/test_conversions.mjs

echo "=== Running E2E smoke test ==="
node --test --test-timeout 300000 tests/test_e2e_smoke.mjs

echo "=== Done ==="
