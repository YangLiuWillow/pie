"""Cold-vs-resumed divergence at temp 0 on a hybrid, strategy A.

Request 1 (cold): full prefill -- the fold covers the whole prompt; the turn
parks the history prefix. Requests 2 and 3 (identical): resume from the
parked KV; the recurrent layers fold only the tiny unparked suffix from a
zero state. If the fold mattered, C2 != C1. C2 vs C3 is the control: both
resumed, so any C2==C3 && C1!=C2 pattern is systematic, not noise.
"""
import json, urllib.request, sys

URL = "http://127.0.0.1:8080/v1/chat/completions"
src = open('/Users/liuyang/Documents/Liszt_ai/pie-opencode/runtime/engine/src/store/kv.rs').read()
FILLER = src[:36000]  # ~10k tokens of real code

msgs = [
    {"role": "system", "content": "You are a precise code summarizer."},
    {"role": "user", "content": f"Here is a Rust source file:\n\n{FILLER}\n\nIn one sentence, what is the single most important invariant this file maintains?"},
]
body = json.dumps({"model": "qwen3.6-35b-a3b", "messages": msgs,
                   "max_tokens": 64, "temperature": 0.0}).encode()

def ask(tag):
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer pie-local"})
    with urllib.request.urlopen(req, timeout=900) as r:
        o = json.load(r)
    ch = o["choices"][0]; u = o.get("usage") or {}
    text = ch["message"]["content"]
    print(f"{tag}: cached_tokens={ (u.get('prompt_tokens_details') or {}).get('cached_tokens', u.get('cached_tokens', '?')) } "
          f"prompt={u.get('prompt_tokens')} finish={ch.get('finish_reason')}")
    print(f"  text: {text[:200]!r}")
    return text

c1 = ask("C1 (cold)   ")
c2 = ask("C2 (resumed)")
c3 = ask("C3 (resumed)")
print()
print(f"C1 == C2: {c1 == c2}")
print(f"C2 == C3: {c2 == c3}")
if c1 != c2 and c2 == c3:
    print("VERDICT: systematic divergence — the resumed state is NOT the cold state")
elif c1 == c2 == c3:
    print("VERDICT: no divergence observed on this prompt")
else:
    print("VERDICT: inconsistent — noise or nondeterminism; needs repeats")
