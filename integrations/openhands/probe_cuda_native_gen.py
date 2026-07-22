#!/usr/bin/env python3
"""Direct generation probe against a running pie server.

Runs the text-completion inferlet with a few prompts and prints the raw
model output plus whether generation stopped naturally (Return/Done) or ran
to max_tokens. Used to confirm the native CUDA driver decodes coherently and
emits stop tokens — isolating driver decode from agent/tool-format concerns.
"""
import asyncio, json, os, sys
from pie_client import Event, PieClient

URI = os.environ.get("PIE_URI", "ws://127.0.0.1:18082")
INFERLET = os.environ.get("PROBE_INFERLET", "text-completion@0.2.15")

PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number. Only the code.",
    "In one sentence, what is a transformer in machine learning?",
    "Reverse the string 'hello world' and explain briefly.",
]

async def main():
    async with PieClient(URI) as c:
        await c.authenticate("local-dev")
        for i, p in enumerate(PROMPTS):
            payload = {"prompt": p, "max_tokens": 200, "temperature": 0.0, "top_p": 1.0}
            proc = await c.launch_process(INFERLET, input=payload)
            ret, err, stdout = None, None, []
            while True:
                event, value = await asyncio.wait_for(proc.recv(), timeout=600)
                if event == Event.Return:
                    if isinstance(value, (bytes, bytearray)): value = value.decode()
                    ret = value
                    break
                if event == Event.Error:
                    err = value
                    break
                # capture any stdout/print events
                if isinstance(value, (bytes, bytearray)): value = value.decode(errors="replace")
                stdout.append(str(value))
            print(f"\n===== PROMPT {i}: {p}")
            if err is not None:
                print(f"  ERROR: {err!r}")
                continue
            try:
                parsed = json.loads(ret) if isinstance(ret, str) else ret
            except Exception:
                parsed = ret
            text = parsed if isinstance(parsed, str) else json.dumps(parsed)
            print(f"  ---- OUTPUT ({len(text)} chars) ----")
            print(text[:1200])
            if stdout:
                joined = "".join(stdout)
                print(f"  ---- STDOUT ({len(joined)} chars) ----")
                print(joined[:1200])

if __name__ == "__main__":
    asyncio.run(main())
