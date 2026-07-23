"""Exp 1 — parser equivalence (offline). Tests the /btw hypothesis: does the
tool-call PARSER cause pie-vs-litellm divergence?

For every raw model output captured in a pie run's PIE_DEBUG_LOG jsonl
(``debug_full_text`` = the model's native Qwen3-Coder XML), parse it with:
  (i)  our Python port  pie_openhands.qwen3coder_parser.extract_tool_calls
  (ii) the REAL vLLM     Qwen3CoderToolParser.extract_tool_calls (if importable)
and diff the extracted (name, arguments) tuples.

Prediction if the user is right (both backends share the qwen3_coder parser):
(i) == (ii) on 100% of outputs -> parser exonerated; divergence is upstream
(render / engine), not the parser.
"""
from __future__ import annotations
import glob, json, sys, traceback
from pathlib import Path

DEBUG_GLOBS = [
    "logs/pie_debug_aligned_session_sub4_19023313.jsonl",
    "logs/pie_debug_*29*.jsonl",
    "logs/pie_debug_*.jsonl",
]


def find_debug_logs() -> list[str]:
    seen: list[str] = []
    for g in DEBUG_GLOBS:
        for p in sorted(glob.glob(g)):
            if p not in seen:
                seen.append(p)
    return seen


def load_raw_outputs(paths: list[str]) -> list[str]:
    outs = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("debug_full_text") or d.get("full_text") or d.get("text")
            if isinstance(t, str) and "<function=" in t:
                outs.append(t)
    return outs


def norm(calls) -> list[tuple]:
    """Normalize either parser's output to a comparable list of (name, args)."""
    res = []
    for c in (calls or []):
        # our port returns dicts; vLLM returns ToolCall objects
        if isinstance(c, dict):
            fn = c.get("function", c)
            name = fn.get("name"); args = fn.get("arguments")
        else:
            fn = getattr(c, "function", c)
            name = getattr(fn, "name", None); args = getattr(fn, "arguments", None)
        # arguments may be a json string or dict — canonicalize
        if isinstance(args, str):
            try: args = json.loads(args)
            except Exception: pass
        res.append((name, json.dumps(args, sort_keys=True) if args is not None else None))
    return res


def main() -> int:
    paths = find_debug_logs()
    print("debug logs:", paths)
    outs = load_raw_outputs(paths)
    print(f"raw outputs with <function=>: {len(outs)}")
    if not outs:
        print("NO raw outputs found — check PIE_DEBUG_LOG jsonl paths")
        return 2

    # (i) our python port — load the module file directly to avoid the
    # pie_openhands package __init__ (which pulls in litellm, absent here).
    import importlib.util
    _pp = Path(__file__).parent / "pie_openhands" / "qwen3coder_parser.py"
    _spec = importlib.util.spec_from_file_location("qwen3coder_parser", _pp)
    port = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(port)
    port_fn = getattr(port, "extract_tool_calls", None)
    print("port entry:", port_fn)

    # (ii) real vLLM parser (best-effort import)
    vllm_parser = None
    try:
        try:  # vllm >= 0.16 moved the module
            from vllm.tool_parsers.qwen3coder_tool_parser import (
                Qwen3CoderToolParser,
            )
        except ImportError:  # older vllm path
            from vllm.entrypoints.openai.tool_parsers.qwen3coder_tool_parser import (
                Qwen3CoderToolParser,
            )
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
        vllm_parser = Qwen3CoderToolParser(tok)
        print("REAL vLLM parser loaded")
    except Exception:
        print("could not load real vLLM parser:\n" + traceback.format_exc())

    tools: list = []  # both parsers get the same (empty) tools -> fair diff
    n = len(outs)
    port_ok = vllm_ok = both = agree = 0
    mismatches = []
    for i, text in enumerate(outs):
        pc = vc = None
        try:
            pc = norm(port_fn(text, tools)); port_ok += 1
        except Exception:
            try:
                pc = norm(port_fn(model_output=text, tools=tools)); port_ok += 1
            except Exception:
                pc = None
        if vllm_parser is not None:
            try:
                class _Req:  # minimal request stub — same (empty) tools as port
                    tools = []
                vc_raw = vllm_parser.extract_tool_calls(text, _Req())
                vc = norm(getattr(vc_raw, "tool_calls", vc_raw)); vllm_ok += 1
            except Exception:
                vc = None
        if pc is not None and vc is not None:
            both += 1
            if pc == vc:
                agree += 1
            else:
                if len(mismatches) < 25:
                    mismatches.append({"idx": i, "port": pc, "vllm": vc, "text_head": text[:400]})

    print("\n=== Exp1 parser-equivalence summary ===")
    print(f"raw outputs           : {n}")
    print(f"port parsed ok        : {port_ok}")
    print(f"vllm parsed ok        : {vllm_ok}")
    print(f"both parsed           : {both}")
    print(f"port == vllm (agree)  : {agree}")
    if both:
        print(f"agreement rate        : {100.0*agree/both:.1f}%")
    print(f"mismatches (<=25 shown): {len(mismatches)}")

    out = {
        "n": n, "port_ok": port_ok, "vllm_ok": vllm_ok, "both": both,
        "agree": agree, "mismatches": mismatches,
    }
    Path("logs/exp1_parser_equiv_result.json").write_text(json.dumps(out, indent=2))
    print("wrote logs/exp1_parser_equiv_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
