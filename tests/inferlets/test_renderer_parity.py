"""Renderer-parity harness — the Risk R1 de-risk (spec Track B item 4).

The gateway seeds its cumulative accumulator from Pie's turn-0 rendering,
then extends every later turn with rllm's `renderers` package
(`bridge_to_next_turn`). For multi-turn merging to be trustworthy, Pie's
chat template must agree with `renderers` token-for-token. This asserts:

  A. TURN-0 PARITY. Pie's `/v1/chat/completions` prompt_token_ids ==
     renderers.render_ids([system, user], add_generation_prompt=True).
     A mismatch here IS the chat-template drift R1 warns about.

  B. BRIDGE CONSISTENCY. Given Pie's turn-0 (prompt_ids + completion_ids),
     renderers.bridge_to_next_turn(..., [new user msg]) must equal a full
     renderers.render_ids([system, user, assistant(completion), new user],
     add_generation_prompt=True) — i.e. the bridge that the gateway will
     apply to Pie's tokens reproduces a ground-truth re-render, so turns
     merge (§3.4 append-only invariant holds through real templating).

Runs on the pod (needs both the Pie daemon and the `renderers` package):

    pip install renderers transformers
    python3 tests/inferlets/test_renderer_parity.py \\
        --driver portable --model /root/qwen17b-f16 --device auto \\
        --renderer-family qwen3
"""
from __future__ import annotations

import asyncio

import httpx

from conftest import make_parser, _run
from test_rl_completions import _launch


def _first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), None, None
    return None


async def test_renderer_parity(client, args):
    from transformers import AutoTokenizer
    from renderers import Message, config_from_name, create_renderer

    tok = AutoTokenizer.from_pretrained(args.model)
    renderer = create_renderer(tok, config_from_name(args.renderer_family))

    system = "You are a terse coding assistant."
    user1 = "What language is the Linux kernel written in?"
    user2 = "And its build system?"

    base = await _launch(client)
    async with httpx.AsyncClient(timeout=args.timeout) as http:
        # ── Pie turn-0 ──
        r0 = await http.post(
            f"{base}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user1},
                ],
                "max_tokens": 24,
                "temperature": 0.0,
            },
        )
        assert r0.status_code == 200, f"turn-0: {r0.status_code} {r0.text[:200]}"
        b0 = r0.json()
        pie_prompt = b0["prompt_token_ids"]
        pie_completion = b0["choices"][0]["token_ids"]

        # ── A. Turn-0 parity ──
        ref0 = renderer.render_ids(
            [Message(role="system", content=system), Message(role="user", content=user1)],
            add_generation_prompt=True,
        )
        div = _first_divergence(pie_prompt, ref0)
        assert div is None, (
            f"TURN-0 DRIFT: Pie ({len(pie_prompt)} tok) vs renderers "
            f"({len(ref0)} tok) diverge at index {div[0]}: {div[1]} != {div[2]}\n"
            f"  pie[{max(0,div[0]-3)}:{div[0]+3}]={pie_prompt[max(0,div[0]-3):div[0]+3]}\n"
            f"  ref[{max(0,div[0]-3)}:{div[0]+3}]={ref0[max(0,div[0]-3):div[0]+3]}"
        )
        print(f"A. turn-0 parity: {len(pie_prompt)} tokens identical")

        # ── B. Bridge consistency ──
        # The completion Pie returns includes the stop token (§3.1); strip it
        # for the assistant *content* the renderer re-renders from text, but
        # feed the raw ids to bridge (which works on the exact token stream).
        assistant_text = tok.decode(pie_completion).replace("<|im_end|>", "").strip()

        bridged = renderer.bridge_to_next_turn(
            pie_prompt,
            pie_completion,
            [Message(role="user", content=user2)],
        )
        assert bridged is not None, "bridge_to_next_turn returned None (prefix contract failed)"
        bridged_ids = list(bridged.token_ids)

        ref1 = renderer.render_ids(
            [
                Message(role="system", content=system),
                Message(role="user", content=user1),
                Message(role="assistant", content=assistant_text),
                Message(role="user", content=user2),
            ],
            add_generation_prompt=True,
        )
        div = _first_divergence(bridged_ids, ref1)
        assert div is None, (
            f"BRIDGE DRIFT: bridge ({len(bridged_ids)} tok) vs full re-render "
            f"({len(ref1)} tok) diverge at index {div[0]}: {div[1]} != {div[2]}"
        )
        # And the append-only invariant the trainer checks (§3.4):
        assert bridged_ids[: len(pie_prompt) + len(pie_completion)] == pie_prompt + pie_completion, (
            "bridge output does not start with pie prompt+completion (append-only broken)"
        )
        print(f"B. bridge consistency: {len(bridged_ids)} tokens match full re-render; "
              f"append-only prefix holds")


def tests():
    return [test_renderer_parity]


if __name__ == "__main__":
    parser = make_parser("Renderer parity")
    parser.add_argument("--renderer-family", default="qwen3")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(tests(), args)))
