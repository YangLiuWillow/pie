#!/usr/bin/env python3
"""Capture vLLM's own rendered token stream for the fixture corpus.

`check_render.py` compares pie against `apply_chat_template`, which answers
"does our renderer match my reading of the template". That was the right
reference while the template *was* the specification (docs §12). For the
Qwen3.6 A/B it is the wrong one: what the benchmark compares against is what
the **vLLM arm actually feeds the model**, and vLLM 0.27 will simply tell us
via `POST /v1/chat/completions/render`, which returns the exact `token_ids`.

Using the served reference also removes a whole class of error: the MLX build
ships an older template than the official repo (§12), `hf_reference()` has to
re-implement vLLM's content normalization by hand — and got the multi-part
separator wrong until vLLM's own render caught it, 3 bytes over 43 KB.

Both stacks want ~20 GB, so they cannot be up at once on a 48 GB machine.
This script captures the reference to disk; `check_render.py --reference`
then compares pie against the file with vLLM long since shut down.

The server MUST run with `--enable-auto-tool-choice --tool-call-parser
qwen3_xml` (alias `qwen3_coder`), or a tools-bearing request is rejected 400
— and, more to the point, the benchmark arm would emit no `tool_calls` at
all, which is run 2's failure mode reproduced on the vLLM side.

Usage:
    vllm serve mlx-community/Qwen3.6-35B-A3B-4bit --port 18000 \
        --max-model-len 16384 --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml &
    python3 capture_vllm_render.py --base http://127.0.0.1:18000 \
        --model mlx-community/Qwen3.6-35B-A3B-4bit -o vllm_render_qwen36.json
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES = REPO / "tests/inferlets/fixtures/rl_completions/wire"

# Fields that describe the conversation. Everything else in a qwen-code
# capture (max_tokens, stream, stream_options, …) is generation policy and
# has no effect on the rendered prompt, so it is dropped rather than passed
# through, where it could make /render reject a request it would otherwise
# have answered.
PROMPT_FIELDS = ("messages", "tools", "tool_choice", "chat_template_kwargs")


def render_one(base, model, request, timeout=300):
    body = {k: request[k] for k in PROMPT_FIELDS if k in request}
    body["model"] = model
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions/render",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000")
    ap.add_argument("--model", required=True,
                    help="the model id the server was launched with")
    ap.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    fixtures = sorted(Path(args.fixtures).glob("episode*/openai-*.json"))
    if not fixtures:
        print(f"no fixtures under {args.fixtures}", file=sys.stderr)
        return 2

    out = {"model": args.model, "source": "vllm /v1/chat/completions/render",
           "renders": {}}
    for f in fixtures:
        key = f"{f.parent.name}/{f.name}"
        request = json.loads(f.read_text())["request"]
        try:
            r = render_one(args.base, args.model, request)
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            print(f"[HTTP {e.code}]  {key}\n  {detail}", file=sys.stderr)
            if e.code == 400 and "tool choice" in detail:
                print("\nStart the server with --enable-auto-tool-choice "
                      "--tool-call-parser qwen3_xml.", file=sys.stderr)
            return 1
        ids = r.get("token_ids")
        if not ids:
            print(f"no token_ids for {key}: keys={list(r)}", file=sys.stderr)
            return 1
        out["renders"][key] = ids
        print(f"[{len(ids):>6} tok]  {key}")

    Path(args.out).write_text(json.dumps(out))
    print(f"\n{len(out['renders'])} renders → {args.out}")
    print("vLLM can be shut down now; check_render.py --reference reads the file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
