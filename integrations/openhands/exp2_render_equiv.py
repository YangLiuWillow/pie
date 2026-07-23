"""Exp 2 — render equivalence. Isolates layer (C): does the inferlet's OWN Rust
``render_prompt`` (coder-session lib.rs:258) produce the SAME prompt tokens as
vLLM's chat-template render for the same messages+tools? A mismatch = the model
sees a different prompt on the two backends -> guaranteed trajectory divergence,
independent of engine nondeterminism or parser.

Method: send ONE fixed (messages, tools) through the live inferlet via PieLLM and
read the session telemetry ``len`` (+ ``hash``) = the inferlet's render token
count. Independently apply the HF tokenizer's chat template to the same
messages+tools = vLLM's render. Compare token counts (and decoded text) to spot
divergence. Requires a running ``pie serve`` (booted by the sbatch).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

URI = "ws://127.0.0.1:18099"
INFERLET = "openhands-coder-session@0.1.0"
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# A small but representative multi-turn conversation with a tool call + result,
# exercising system prompt, tool schema, assistant tool_call, and tool result.
MESSAGES = [
    {"role": "system", "content": "You are a coding agent. Fix issues with minimal changes."},
    {"role": "user", "content": "Fix the failing test in foo.py."},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c0", "type": "function",
         "function": {"name": "terminal", "arguments": '{"command": "pytest -x"}'}},
    ]},
    {"role": "tool", "tool_call_id": "c0", "content": "E   AssertionError in test_bar"},
    {"role": "user", "content": "Now inspect foo.py and propose a fix."},
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "the command"}},
            "required": ["command"],
        },
    },
}]


def inferlet_render_len() -> dict:
    """Send the fixed prompt via PieLLM (session on) -> inferlet render len/hash."""
    from pie_openhands import PieLLM
    llm = PieLLM(
        model=MODEL, pie_uri=URI, pie_username="local-dev", pie_inferlet=INFERLET,
        pie_session=True, native_tool_calling=True, pie_python_tool_parser=True,
        pie_kv_verify=True, num_retries=1, retry_min_wait=0, retry_max_wait=0,
    )
    try:
        llm._transport_call(messages=MESSAGES, tools=TOOLS, max_tokens=4, temperature=0.0)
        s = llm._pie_session_stats[-1]
        return {"len": s["prompt_len"], "mode": s["mode"],
                "hash": getattr(llm, "_pie_session_hash", None)}
    finally:
        try: llm.close_pie_session()
        except Exception: pass


def _postprocess_messages(messages: list) -> list:
    """Mirror vLLM's _postprocess_messages (chat_utils.py): assistant tool_call
    arguments arrive as OpenAI JSON strings but the Qwen3-Coder chat template
    does ``arguments|items`` and needs dicts. This is exactly what the litellm
    baseline's vLLM server does before apply_chat_template — replicate it so the
    render comparison is faithful."""
    import copy
    msgs = copy.deepcopy(messages)
    for m in msgs:
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            for item in m["tool_calls"]:
                fn = item.get("function", {})
                content = fn.get("arguments")
                if content:
                    if not isinstance(content, (dict, list)):
                        fn["arguments"] = json.loads(content)
                else:
                    fn["arguments"] = {}
    return msgs


def vllm_render() -> dict:
    """vLLM-side render: HF tokenizer chat template over the same messages+tools."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    msgs = _postprocess_messages(MESSAGES)
    # add_generation_prompt=True matches a pre-generation render (cue excluded on
    # the inferlet side; we report both text and token count for inspection).
    ids = tok.apply_chat_template(
        msgs, tools=TOOLS, add_generation_prompt=True, tokenize=True,
    )
    text = tok.apply_chat_template(
        msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False,
    )
    return {"len": len(ids), "text": text, "ids_head": ids[:40], "ids_tail": ids[-40:]}


def main() -> int:
    print("=== Exp 2: render equivalence (inferlet Rust vs vLLM chat template) ===")
    inf = {}
    try:
        inf = inferlet_render_len()
        print("inferlet render:", inf)
    except Exception as e:
        print("inferlet render FAILED:", repr(e))
    vl = vllm_render()
    print(f"vllm render len: {vl['len']}")
    print("--- vllm rendered text (first 1500 chars) ---")
    print(vl["text"][:1500])

    result = {
        "inferlet_len": inf.get("len"),
        "inferlet_hash": inf.get("hash"),
        "vllm_len": vl["len"],
        "len_delta": (inf.get("len") - vl["len"]) if inf.get("len") is not None else None,
        "vllm_ids_head": vl["ids_head"],
        "vllm_ids_tail": vl["ids_tail"],
        "vllm_text": vl["text"],
    }
    Path("logs/exp2_render_equiv_result.json").write_text(json.dumps(result, indent=2))
    print("\nlen_delta (inferlet - vllm):", result["len_delta"])
    print("NOTE: inferlet excludes the trailing generation cue; a small constant "
          "delta may reflect the cue, a large/irregular delta = real render "
          "divergence (layer C). Inspect vllm_text vs the inferlet render.")
    print("wrote logs/exp2_render_equiv_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
