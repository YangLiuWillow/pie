#!/usr/bin/env python3
"""C3 renderer parity: pie's chat-completions renderer vs HF apply_chat_template.

For every checked-in qwen-code wire capture, ask the running daemon for its
rendered token stream (`echo_tokens` debug flag → decoded text + ids) and
compare byte-for-byte against `tokenizer.apply_chat_template(...)` for the
same conversation — the load-bearing verification both integration plans
deferred (`docs/qwen-code-dev-port.md` §3, old plan §C3).

Known deliberate divergences are normalized away and REPORTED, never
silently accepted:

  no-think   pie renders the Qwen3 soft switch (" /no_think" on every user
             turn, position-independent so history replays byte-identically);
             HF's `enable_thinking=False` instead plants an empty
             "<think>\\n\\n</think>\\n\\n" block after the generation header.
  args-str   qwen-code sends tool_call arguments as a JSON *string*; vLLM
             parses it to an object before templating (`| tojson`). We mirror
             vLLM by parsing before the HF call, so this never shows as a
             diff — listed here for the record.

Anything else is a MISMATCH and fails the run.

Prereqs: dummy-driver stack up (rendering needs only the tokenizer):
    PIE_CONFIG=integrations/qwen-code/pie_config_dummy.toml pie serve &
    python3 integrations/qwen-code/shim.py ... &
Usage:
    python3 check_render.py [--base http://127.0.0.1:8123]
                            [--hf-model Qwen/Qwen3-0.6B] [--fixtures DIR]
"""

import argparse
import difflib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES = REPO / "tests/inferlets/fixtures/rl_completions/wire"

THINK_CUE = "<think>\n\n</think>\n\n"


def post_echo(base, request_body):
    body = dict(request_body)
    body["echo_tokens"] = True
    body["stream"] = False
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def hf_reference(tok, request_body):
    """Reproduce what vLLM would feed the model for this request."""
    messages = []
    for m in request_body["messages"]:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):  # parts → text, matching vLLM normalization
            m["content"] = "\n".join(
                p.get("text", "") for p in c if p.get("type") == "text")
        if m.get("content") is None:
            m["content"] = ""
        for tc in m.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments")
            if isinstance(args, str):  # vLLM parses before templating
                try:
                    tc["function"]["arguments"] = json.loads(args)
                except ValueError:
                    pass
        m.pop("reasoning_content", None)  # template ignores it; be explicit
        messages.append(m)

    kwargs = {}
    ctk = request_body.get("chat_template_kwargs") or {}
    if "enable_thinking" in ctk:
        kwargs["enable_thinking"] = ctk["enable_thinking"]
    return tok.apply_chat_template(
        messages,
        tools=request_body.get("tools") or None,
        add_generation_prompt=True,
        tokenize=False,
        **kwargs,
    )


def normalize(pie_text, hf_text):
    """Strip the known-divergence bytes from both sides; report what fired."""
    fired = []
    if " /no_think" in pie_text:
        pie_text = pie_text.replace(" /no_think", "")
        fired.append("no-think(pie soft switch)")
    if hf_text.endswith(THINK_CUE):
        hf_text = hf_text[: -len(THINK_CUE)]
        fired.append("no-think(HF empty think block)")
    return pie_text, hf_text, fired


def first_diff(a, b, ctx=90):
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k] != b[k]), n)
    if i == len(a) == len(b):
        return None
    return (f"  first divergence at byte {i}:\n"
            f"    pie: …{a[max(0, i - ctx):i + ctx]!r}…\n"
            f"    hf:  …{b[max(0, i - ctx):i + ctx]!r}…")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8123")
    ap.add_argument("--hf-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    args = ap.parse_args()

    from transformers import AutoTokenizer  # deferred: slow import
    tok = AutoTokenizer.from_pretrained(args.hf_model)

    fixtures = sorted(Path(args.fixtures).glob("episode*/openai-*.json"))
    if not fixtures:
        print(f"no fixtures under {args.fixtures}", file=sys.stderr)
        return 2

    exact = normalized = mismatched = 0
    for f in fixtures:
        capture = json.loads(f.read_text())
        request = capture["request"]
        echo = post_echo(args.base, request)
        pie_text = echo["rendered_text"]
        hf_text = hf_reference(tok, request)

        if pie_text == hf_text:
            exact += 1
            print(f"[EXACT]      {f.parent.name}/{f.name}")
            continue
        pie_n, hf_n, fired = normalize(pie_text, hf_text)
        if pie_n == hf_n:
            normalized += 1
            print(f"[KNOWN-DIV]  {f.parent.name}/{f.name} — {', '.join(fired)}")
            continue
        mismatched += 1
        print(f"[MISMATCH]   {f.parent.name}/{f.name}")
        print(first_diff(pie_n, hf_n))
        # A compact unified diff of the tails often localizes render bugs.
        tail = lambda s: s[-1200:].splitlines(keepends=True)
        sys.stdout.writelines(
            f"    {l}" for l in difflib.unified_diff(
                tail(pie_n), tail(hf_n), "pie", "hf", n=1))

    print(f"\n{exact} exact, {normalized} known-divergence, "
          f"{mismatched} mismatched of {len(fixtures)} fixtures")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
