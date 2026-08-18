"""Walk one conversation's context upward until the session dies, and say where.

The repro for `finding-inferlet-killed-at-large-context.md`. No agent, no
Docker, no dataset -- one growing chat against a booted server, ~25 minutes.

    tools/boot_pie.sh ramp PIE_STRATEGY=b PIE_MODEL=qwen3.6-35b-a3b \\
        PIE_MAX_MODEL_LEN=65536 PIE_MAX_FORWARD_TOKENS=4096
    python3 tools/ramp_context.py [max_tokens]

`max_tokens` is the variable that MOVES the failure point, which is the whole
reason this script takes it as an argument: it enters the guest in exactly one
place, the pool reservation in `opencode-session/src/engine.rs`. Run it at 8
and at 4096 and the death moves by ~8k tokens.
"""
import json, urllib.request, sys, time
URL="http://127.0.0.1:8080/v1/chat/completions"
src=open('/Users/liuyang/Documents/Liszt_ai/pie-opencode/runtime/engine/src/store/kv.rs').read()
BLOCK=src[:8000]   # ~2.2k tokens per step: fine-grained around the edge
MAX_TOKENS=int(sys.argv[1]) if len(sys.argv)>1 else 8
print(f"ramp: max_tokens={MAX_TOKENS}", flush=True)
msgs=[{"role":"system","content":"You are terse."}]
last=0
for i in range(40):
    msgs.append({"role":"user","content":f"Chunk {i}:\n\n{BLOCK}\n\nReply just OK."})
    body=json.dumps({"model":"qwen3.6-35b-a3b","messages":msgs,"max_tokens":MAX_TOKENS,
                     "temperature":0.0}).encode()
    req=urllib.request.Request(URL,data=body,headers={
        "Content-Type":"application/json","Authorization":"Bearer pie-local"})
    try:
        with urllib.request.urlopen(req,timeout=900) as r: o=json.load(r)
    except Exception as e:
        print(f"step {i}: DIED after last good prompt={last}  ({type(e).__name__}: {e})")
        sys.exit(2)
    ch=o["choices"][0]; u=o.get("usage") or {}
    pt=u.get("prompt_tokens") or 0
    bad = ch.get("finish_reason")=="length" and (u.get("completion_tokens") or 0)==0
    print(f"step {i}: prompt={pt:>6} finish={ch.get('finish_reason')!r:10} {'DEGRADED' if bad else 'ok'}", flush=True)
    if bad:
        print(f"DEGRADED at prompt={pt} (previous good {last})"); sys.exit(3)
    last=pt
    msgs.append({"role":"assistant","content":(ch["message"].get("content") or "").strip()})
print("ramp completed without dying; last prompt =", last)
