#!/usr/bin/env python3
"""Renderer-parity harness — Risk R1, ported to pie 0.5.

The rllm training gateway seeds its cumulative accumulator from Pie's turn-0
rendering, then extends later turns with the `renderers` package
(`bridge_to_next_turn`). Multi-turn merging is trustworthy only if Pie's
chat template agrees with `renderers` token-for-token. Asserts:

  A. TURN-0 PARITY: bridge /v1/chat/completions prompt_token_ids ==
     renderers.render_ids([system, user], add_generation_prompt=True).
     A mismatch here IS the chat-template drift R1 warns about.
  B. BRIDGE CONSISTENCY: renderers.bridge_to_next_turn(pie_prompt,
     pie_completion, [tool result]) succeeds and starts with pie_prompt +
     pie_completion (§3.4 append-only) — the invariants cumulative-token
     training depends on, tested in the shape qwen-code actually produces
     (turns >= 1 extend with TOOL results). A new USER query is refused by
     design under qwen3's "tool_cycle" thinking retention (the template
     drops prior <think> blocks there, so verbatim extension is impossible;
     the gateway resets its accumulator = a segment break, not an error) —
     asserted as the expected policy.

Run on the pod (needs `pip install renderers transformers` and a live
bridge): python3 test_renderer_parity.py [--model Qwen/Qwen3-1.7B]
"""
import argparse
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8123"


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), None, None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--renderer-family", default="qwen3")
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from renderers import Message, config_from_name, create_renderer

    tok = AutoTokenizer.from_pretrained(args.model)
    renderer = create_renderer(tok, config_from_name(args.renderer_family))

    system = "You are a terse coding assistant."
    user1 = "What language is the Linux kernel written in?"
    user2 = "And its build system?"

    req = urllib.request.Request(
        f"{args.base}/v1/chat/completions",
        data=json.dumps({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user1},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
        }).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    pie_prompt = body["prompt_token_ids"]
    pie_completion = body["choices"][0]["token_ids"]
    assert pie_prompt, "bridge returned no prompt_token_ids"

    # A. Turn-0 parity
    ref0 = renderer.render_ids(
        [Message(role="system", content=system), Message(role="user", content=user1)],
        add_generation_prompt=True,
    )
    div = first_divergence(pie_prompt, list(ref0))
    if div is not None:
        i = div[0]
        print(f"TURN-0 DRIFT at index {i}: pie={div[1]} ref={div[2]}")
        print(f"  pie[{max(0, i-3)}:{i+3}] = {pie_prompt[max(0, i-3):i+3]}")
        print(f"  ref[{max(0, i-3)}:{i+3}] = {list(ref0)[max(0, i-3):i+3]}")
        print(f"  pie len={len(pie_prompt)} ref len={len(ref0)}")
        print(f"  pie tail decoded: {tok.decode(pie_prompt[max(0, i-8):])!r}")
        print(f"  ref tail decoded: {tok.decode(list(ref0)[max(0, i-8):])!r}")
        return 1
    print(f"A. turn-0 parity: {len(pie_prompt)} tokens identical")

    # B. Bridge consistency, in the shape training traffic has: turns >= 1
    # extend the conversation with TOOL results (qwen-code's loop). This is
    # the path Phase 1a measured holding 59/59 append-only pairs.
    tool_result = "drwxr-xr-x 2 root root 4096 kernel/"
    bridged = renderer.bridge_to_next_turn(
        pie_prompt, pie_completion,
        [Message(role="tool", content=tool_result, tool_call_id="call_0")],
    )
    assert bridged is not None, "bridge_to_next_turn refused a tool continuation"
    bridged_ids = list(bridged.token_ids)
    head = pie_prompt + pie_completion
    assert bridged_ids[: len(head)] == head, (
        "bridge output does not start with pie prompt+completion (append-only broken)"
    )
    tail = tok.decode(bridged_ids[len(head):])
    assert tool_result in tail, f"bridged tail does not carry the tool result: {tail!r}"
    print(f"B. bridge consistency: tool turn bridges, append-only prefix holds, "
          f"tail renders the result ({len(bridged_ids) - len(head)} tokens)")

    # Expected policy: a NEW USER QUERY is refused under tool_cycle thinking
    # retention (template drops prior <think> there — the gateway resets its
    # accumulator, a segment break by design).
    refused = renderer.bridge_to_next_turn(
        pie_prompt, pie_completion, [Message(role="user", content=user2)]
    )
    assert refused is None, (
        "expected bridge refusal on a new user query under tool_cycle retention"
    )
    print("   user-query continuation correctly refused (accumulator segment break)")
    print("RENDERER_PARITY_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
