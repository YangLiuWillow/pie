#!/usr/bin/env bash
# Run the qwen-code benchmark task set against one serving arm.
#
#   ARM=pie   BASE_URL=http://127.0.0.1:8123/v1  MODEL=qwen3-coder bash run_arm.sh
#   ARM=vllm  BASE_URL=http://127.0.0.1:18000/v1 MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct bash run_arm.sh
#
# For each task in tasks.jsonl: fresh workspace, audited launch profile
# (docs/qwen-code-rl-audit.md §6) with temperature pinned to 0 so the two
# arms are trajectory-comparable, `--openai-logging` for wire captures.
# Results land in $OUT_DIR/<task-id>/ (wall.txt, rc.txt, check.txt,
# stdout.log, logs/openai/*.json). Summarize with summarize.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=${ARM:?set ARM=pie|vllm}
BASE_URL=${BASE_URL:?set BASE_URL}
MODEL=${MODEL:?set MODEL (client-side model name)}
QWEN_VERSION=${QWEN_VERSION:-0.21.6}
OUT_DIR=${OUT_DIR:-$SCRIPT_DIR/results/$ARM-$(echo "$MODEL" | tr '/' '_')}
TASKS=${TASKS:-$SCRIPT_DIR/tasks.jsonl}
MAX_TURNS=${MAX_TURNS:-12}
WALL_LIMIT=${WALL_LIMIT:-10m}

mkdir -p "$OUT_DIR"

run_one() {
    local id="$1" prompt="$2" setup="$3" check="$4"
    local ws tdir
    tdir="$OUT_DIR/$id"
    rm -rf "$tdir" && mkdir -p "$tdir"
    ws="$(mktemp -d)/repo" && mkdir -p "$ws"

    (cd "$ws" && bash -c "$setup")
    mkdir -p "$ws/.qwen"
    cat > "$ws/.qwen/settings.json" <<'EOF'
{
  "model": {
    "skipStartupContext": true,
    "maxToolCallsPerTurn": 0,
    "generationConfig": { "contextWindowSize": 100000000, "temperature": 0 }
  },
  "context": { "clearContextOnIdle": {
    "toolResultsThresholdMinutes": -1, "toolResultsTotalCharsThreshold": -1 } },
  "memory": { "enableManagedAutoMemory": false, "enableManagedAutoDream": false,
              "enableAutoSkill": false },
  "privacy": { "usageStatisticsEnabled": false },
  "tools": { "truncateToolOutputThreshold": 30000 }
}
EOF

    local qhome="$tdir/qwen-home" qruntime="$tdir/qwen-runtime"
    local start end rc
    start=$(python3 -c 'import time; print(time.time())')
    set +e
    (cd "$ws" && \
      QWEN_HOME="$qhome" QWEN_RUNTIME_DIR="$qruntime" \
      QWEN_USAGE_STATISTICS_ENABLED=false QWEN_CODE_SKIP_UPDATE_CHECK_ONCE=1 \
      QWEN_CODE_DISABLE_PRECONNECT=1 QWEN_DISABLE_AUTO_TITLE=1 \
      QWEN_CODE_TOOL_CALL_STYLE=general QWEN_CODE_SUPPRESS_YOLO_WARNING=1 \
      OPENAI_BASE_URL="$BASE_URL" OPENAI_API_KEY=bench OPENAI_MODEL="$MODEL" \
      npx -y @qwen-code/qwen-code@"$QWEN_VERSION" \
        --yolo --bare --safe-mode --auth-type openai \
        --max-session-turns "$MAX_TURNS" --max-wall-time "$WALL_LIMIT" \
        --max-tool-calls 40 --chat-recording false --openai-logging \
        -p "$prompt" < /dev/null > "$tdir/stdout.log" 2>&1)
    rc=$?
    set -e
    end=$(python3 -c 'import time; print(time.time())')

    echo "$rc" > "$tdir/rc.txt"
    python3 -c "print(f'{$end - $start:.2f}')" > "$tdir/wall.txt"
    cp -r "$ws/logs/openai" "$tdir/logs" 2>/dev/null || mkdir -p "$tdir/logs"

    set +e
    (cd "$ws" && bash -c "$check") > "$tdir/check.txt" 2>&1
    echo "check_rc=$?" >> "$tdir/check.txt"
    set -e

    printf '%-16s wall=%ss rc=%s %s\n' "$id" "$(cat "$tdir/wall.txt")" "$rc" \
        "$(grep -q 'check_rc=0' "$tdir/check.txt" && echo PASS || echo FAIL)"
}

echo "arm=$ARM base=$BASE_URL model=$MODEL out=$OUT_DIR"
while IFS= read -r line <&3; do
    [ -z "$line" ] && continue
    id=$(printf '%s' "$line" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
    prompt=$(printf '%s' "$line" | python3 -c 'import json,sys; print(json.load(sys.stdin)["prompt"])')
    setup=$(printf '%s' "$line" | python3 -c 'import json,sys; print(json.load(sys.stdin)["setup"])')
    check=$(printf '%s' "$line" | python3 -c 'import json,sys; print(json.load(sys.stdin)["check"])')
    run_one "$id" "$prompt" "$setup" "$check"
done 3< "$TASKS"
echo "done -> $OUT_DIR"
