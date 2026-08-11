#!/usr/bin/env python3
"""Renderer-parity check: pie's rendering vs HF apply_chat_template.

For each wire capture in tests/inferlets/fixtures/opencode/wire/req-*.json
(chat-completions POSTs only):

  1. HF side: build messages+tools straight from the captured body and call
     tokenizer.apply_chat_template(messages, tools=tools,
     add_generation_prompt=True, enable_thinking=False, tokenize=True)
     on Qwen/Qwen3-0.6B.
  2. pie side: run the render-tokens bin (pie-openai-serving plan_render ->
     QwenInstruct -> pie-tokenizer) on the same fixture file.
  3. Diff the token id sequences; on mismatch print a first-divergence report
     (token index, id/text both sides, +/-5 context tokens, decoded windows).

Run from anywhere:

  python3 check_render.py \
      --bin  /path/to/target/debug/render-tokens \
      [--fixtures-dir .../tests/inferlets/fixtures/opencode/wire] \
      [--model Qwen/Qwen3-0.6B] [--cache-dir DIR] [--fixture req-005.json]

Requires: transformers (tokenizer-only; no torch), huggingface_hub.
Downloads tokenizer files only (tokenizer.json/tokenizer_config.json/...),
never weights.
"""

import argparse
import collections
import difflib
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES = REPO_ROOT / "tests/inferlets/fixtures/opencode/wire"

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "config.json",
    "generation_config.json",
]


def load_body(path: pathlib.Path):
    """Fixture wrapper or raw body -> (request body dict, request path or None)."""
    data = json.loads(path.read_text())
    if "body" in data:
        body = data["body"]
        if isinstance(body, str):
            body = json.loads(body)
        return body, data.get("path")
    return data, None


def tok_repr(tok, tid: int) -> str:
    """Human-readable form of one token id (merge-level piece, escaped)."""
    piece = tok.convert_ids_to_tokens([tid])[0]
    return repr(piece)


def context(tok, ids, i, radius=5):
    lines = []
    for j in range(max(0, i - radius), min(len(ids), i + radius + 1)):
        marker = ">>>" if j == i else "   "
        lines.append(f"    {marker} [{j:5d}] {ids[j]:>7d}  {tok_repr(tok, ids[j])}")
    return "\n".join(lines)


def report_divergence(name, tok, pie_ids, hf_ids, pie_text, hf_text):
    n = min(len(pie_ids), len(hf_ids))
    i = next((k for k in range(n) if pie_ids[k] != hf_ids[k]), n)
    print(f"  FIRST DIVERGENCE at token index {i}")
    if i < len(pie_ids):
        print(f"  pie: id {pie_ids[i]} {tok_repr(tok, pie_ids[i])}")
    else:
        print("  pie: <end of sequence>")
    if i < len(hf_ids):
        print(f"  hf : id {hf_ids[i]} {tok_repr(tok, hf_ids[i])}")
    else:
        print("  hf : <end of sequence>")
    print("  pie context:")
    print(context(tok, pie_ids, min(i, len(pie_ids) - 1)))
    print("  hf  context:")
    print(context(tok, hf_ids, min(i, len(hf_ids) - 1)))
    # Character-level view: decoded strings around the shared prefix's end.
    p = 0
    limit = min(len(pie_text), len(hf_text))
    while p < limit and pie_text[p] == hf_text[p]:
        p += 1
    lo = max(0, p - 120)
    print(f"  decoded common prefix: {p} chars; divergence context (char level):")
    print(f"    shared tail : {pie_text[lo:p]!r}")
    print(f"    pie next    : {pie_text[p:p + 160]!r}")
    print(f"    hf  next    : {hf_text[p:p + 160]!r}")
    print(f"  lengths: pie {len(pie_ids)} tokens / {len(pie_text)} chars, "
          f"hf {len(hf_ids)} tokens / {len(hf_text)} chars")
    # The full divergence list, char level, grouped: a handful of systematic
    # template differences shows up as a few high-count rows rather than one
    # unreadable first-diff. (SequenceMatcher on two ~30k-char prompts is
    # fine; autojunk must be off or long common runs get misaligned.)
    sm = difflib.SequenceMatcher(None, pie_text, hf_text, autojunk=False)
    groups = collections.Counter()
    order = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        key = (tag, pie_text[i1:i2], hf_text[j1:j2])
        if key not in order:
            order[key] = i1
        groups[key] += 1
    print(f"  all divergence regions ({sum(groups.values())} total, "
          f"{len(groups)} distinct), char level:")
    for (tag, pa, pb), count in sorted(groups.items(), key=lambda kv: order[kv[0]]):
        print(f"    x{count:4d} {tag:8s} first@char {order[(tag, pa, pb)]}: "
              f"pie={pa[:80]!r} hf={pb[:80]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="path to the render-tokens binary")
    ap.add_argument("--fixtures-dir", default=str(DEFAULT_FIXTURES))
    ap.add_argument("--fixture", action="append",
                    help="specific fixture filename(s); default: all req-*.json")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--cache-dir", default=None,
                    help="huggingface_hub cache dir (default: ~/.cache/huggingface)")
    ap.add_argument("--dump-dir", default=None,
                    help="write per-fixture decoded prompts (pie/hf) here")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snap = snapshot_download(args.model, allow_patterns=TOKENIZER_FILES,
                             cache_dir=args.cache_dir)
    snap = pathlib.Path(snap)
    tok = AutoTokenizer.from_pretrained(snap)
    tokenizer_json = snap / "tokenizer.json"

    fixtures_dir = pathlib.Path(args.fixtures_dir)
    names = args.fixture or sorted(p.name for p in fixtures_dir.glob("req-*.json"))

    results = {}
    for name in names:
        path = fixtures_dir / name
        body, req_path = load_body(path)
        if req_path is not None and not req_path.endswith("/chat/completions"):
            print(f"== {name}: SKIP (path {req_path})")
            continue

        messages = body["messages"]
        tools = body.get("tools") or None

        # Render text, then tokenize with add_special_tokens=False — exactly
        # what apply_chat_template(tokenize=True) does internally, but with a
        # return shape stable across transformers 4.x/5.x.
        hf_text = tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            enable_thinking=False, tokenize=False)
        hf_ids = tok(hf_text, add_special_tokens=False)["input_ids"]

        proc = subprocess.run(
            [args.bin, str(tokenizer_json), str(path)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"== {name}: render-tokens FAILED\n{proc.stderr}")
            results[name] = "bin-error"
            continue
        pie_ids = json.loads(proc.stdout)
        pie_text = proc.stderr  # decoded prompt, special tokens included

        if args.dump_dir:
            d = pathlib.Path(args.dump_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{path.stem}.pie.txt").write_text(pie_text)
            (d / f"{path.stem}.hf.txt").write_text(hf_text)

        if pie_ids == hf_ids:
            print(f"== {name}: OK token-exact ({len(pie_ids)} tokens)")
            results[name] = "exact"
        else:
            print(f"== {name}: MISMATCH")
            report_divergence(name, tok, pie_ids, hf_ids, pie_text, hf_text)
            results[name] = "mismatch"

    print("\nSummary:")
    for name, verdict in results.items():
        print(f"  {name}: {verdict}")
    return 0 if all(v == "exact" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
