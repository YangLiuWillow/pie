#!/usr/bin/env python3
"""Prompt parity: pie's render vs what vLLM ACTUALLY sends the model.

## Why this exists next to `check_render.py`

`check_render.py` diffs pie against HF `apply_chat_template`, which is the right
reference for "does pie render the template correctly". It is **not** the right
reference for a pie-vs-vLLM benchmark, because it assumes vLLM sends exactly
what `apply_chat_template` produces. vLLM has its own chat-template plumbing
(tool-schema injection, `add_generation_prompt`, reasoning-channel handling),
and the only authority on what it feeds the model is vLLM itself.

vLLM exposes `/v1/chat/completions/render`, which returns the token ids for a
request without running it. So this asks both stacks the same question with the
same request body and diffs the answers.

## Why it matters

A cross-stack timing comparison is meaningless if the two stacks see different
prompts — the handover is emphatic, because qwen-code's run 2 lost a whole
benchmark to exactly this. Measured on our bench fixtures before this script
existed: pie 7268 tokens vs vLLM 7235 on the Coder-30B (0.5%), 7351 vs 7471 on
the 35B (1.6%). Small in size, and evidently large in effect — pie ran to
`max_tokens` every turn while vLLM stopped after 11 tokens, which is a
behavioural divergence rather than a timing one.

Usage:
    # vLLM must be serving the same checkpoint
    python3 check_render_vllm.py \
        --bin <target>/release/render-tokens \
        --base-url http://127.0.0.1:8000 --model coder30b
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIXTURES = REPO / "tests/inferlets/fixtures/opencode/wire"


def load_body(path):
    d = json.loads(path.read_text())
    body = d["body"]
    return json.loads(body) if isinstance(body, str) else body


def vllm_render(base_url, model, body, timeout=120):
    """Ask vLLM for the token ids it would prefill for this request.

    Returns (token_ids, raw_response). The endpoint's shape has moved between
    vLLM versions, so several field names are probed rather than assumed.
    """
    req_body = dict(body)
    req_body["model"] = model
    # Strip everything that cannot affect the PROMPT. `max_tokens` in
    # particular is validated against `max_model_len` even by the render
    # endpoint, and the captured opencode fixtures carry max_tokens=32000 —
    # which 400s against a 16384 context and has nothing to do with rendering.
    for k in ("stream", "stream_options", "max_tokens", "max_completion_tokens",
              "temperature", "top_p", "stop", "n", "logprobs", "seed"):
        req_body.pop(k, None)
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions/render",
        data=json.dumps(req_body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer parity"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read())
    for key in ("prompt_token_ids", "token_ids", "tokens", "rendered_ids"):
        if isinstance(obj, dict) and key in obj:
            return obj[key], obj
    # Some builds nest it under a list of prompts.
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for key in ("prompt_token_ids", "token_ids"):
            if key in obj[0]:
                return obj[0][key], obj
    return None, obj


def pie_render(binary, tokenizer, fixture_path):
    out = subprocess.run(
        [str(binary), str(tokenizer), str(fixture_path)],
        capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0:
        raise RuntimeError(f"render-tokens failed: {out.stderr[:400]}")
    d = json.loads(out.stdout)
    # The bin emits a bare array of ids; tolerate a dict wrapper too.
    if isinstance(d, list):
        return d
    for key in ("token_ids", "tokens", "ids", "rendered_ids"):
        if key in d:
            return d[key]
    raise RuntimeError(f"render-tokens output had no token ids: {list(d)[:8]}")


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="path to the render-tokens binary")
    ap.add_argument("--tokenizer", required=True,
                    help="tokenizer.json for the SAME checkpoint vLLM is serving")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures-dir", default=str(FIXTURES))
    ap.add_argument("--fixture", default=None, help="just this one, e.g. req-004.json")
    args = ap.parse_args()

    fdir = Path(args.fixtures_dir)
    names = (
        [args.fixture]
        if args.fixture
        else sorted(p.name for p in fdir.glob("req-*.json"))
    )

    worst = 0
    failures = 0
    for name in names:
        path = fdir / name
        body = load_body(path)
        if not body.get("messages"):
            continue
        try:
            v_ids, raw = vllm_render(args.base_url, args.model, body)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: vLLM render FAILED: {e!r}")
            failures += 1
            continue
        if v_ids is None:
            print(f"{name}: vLLM render returned no token ids; keys={list(raw)[:10]}")
            failures += 1
            continue
        try:
            p_ids = pie_render(args.bin, args.tokenizer, path)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: pie render FAILED: {e!r}")
            failures += 1
            continue

        idx = first_divergence(p_ids, v_ids)
        delta = len(v_ids) - len(p_ids)
        worst = max(worst, abs(delta))
        if idx == -1:
            print(f"{name}: IDENTICAL ({len(p_ids)} tokens)")
            continue
        failures += 1
        print(
            f"{name}: DIVERGES at token {idx} — pie {len(p_ids)} tok, "
            f"vLLM {len(v_ids)} tok (delta {delta:+d})"
        )
        lo = max(0, idx - 5)
        print(f"    pie [{lo}:{idx+5}]  = {p_ids[lo:idx+5]}")
        print(f"    vLLM[{lo}:{idx+5}]  = {v_ids[lo:idx+5]}")

    print()
    print(f"{len(names)} fixtures, {failures} with a divergence, worst length delta {worst}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
