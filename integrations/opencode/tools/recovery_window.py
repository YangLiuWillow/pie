"""How long after saturation does the engine start serving again?

A yes/no probe answered this wrong once already. Immediately after a
conversation saturated the pool and the guest freed it, one request was refused
and the run was written up as "the wedge is back" — but the gateway's admission
gate does not read the pool. It reads `RoutingTable.coarse_load
.kv_pressure_bucket`, a value the worker PUSHES every `REPORT_INTERVAL = 2s`
(`worker/src/link/control.rs`), and the controller only advances the gateway
epoch when the bucket crosses. So a probe fired sub-second after the guest
frees pages is racing a 2-second report and will be refused whether or not
anything is wrong.

The distinction that matters is not served-vs-refused, it is BOUNDED-vs-NOT:

    a window of a few seconds  -> reporting lag, working as designed
    refused indefinitely       -> a real wedge, the guest's pages are still held

So this probes once a second and reports the first success and the elapsed
time, which answers both at once.

    PIE_PROBE_OK=1 python3 tools/recovery_window.py [seconds]   # default 30

Run it straight after `ramp_context.py` has died, against the same server.

`PIE_PROBE_OK=1` is required, and it is not ceremony. This tool sends LIVE
requests to whatever is listening on :8080, and this machine is shared — a peer
session boots its own server on that port for kernel A/Bs. Smoke-testing this
file without the guard put one stray request into someone else's measurement
and cost them a round. A probe that fires blind is a probe that eventually
fires into somebody's experiment, so the port is not proof of ownership and
the caller has to say so.
"""
import json
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
DEADLINE_S = int(sys.argv[1]) if len(sys.argv) > 1 else 30

import os

if os.environ.get("PIE_PROBE_OK") != "1":
    sys.exit(
        "refusing to probe :8080 without PIE_PROBE_OK=1 — this machine is "
        "shared and the port is not proof the server is yours"
    )


def probe():
    body = json.dumps(
        {
            "model": "qwen3.6-35b-a3b",
            "messages": [{"role": "user", "content": "Say OK."}],
            "max_tokens": 8,
        }
    ).encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer pie-local",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return True, json.load(r)["choices"][0]["message"]["content"][:30]
    except urllib.error.HTTPError as e:
        return False, f"{e.code}"
    except Exception as e:  # transport, not admission
        return False, f"{type(e).__name__}"


t0 = time.time()
attempts = 0
while time.time() - t0 < DEADLINE_S:
    attempts += 1
    ok, detail = probe()
    elapsed = time.time() - t0
    print(f"t+{elapsed:5.1f}s  attempt {attempts:2d}  {'SERVED' if ok else 'refused'}  {detail}",
          flush=True)
    if ok:
        print(f"\nRECOVERY WINDOW: {elapsed:.1f}s after the first probe "
              f"({attempts} attempts)")
        if elapsed <= 6:
            print("VERDICT: bounded, and consistent with the 2s coarse-load "
                  "report interval. Not a wedge.")
        else:
            print("VERDICT: bounded but SLOWER than the report interval "
                  "explains — worth a look.")
        sys.exit(0)
    time.sleep(1)

print(f"\nNO RECOVERY in {DEADLINE_S}s ({attempts} attempts)")
print("VERDICT: this is a real wedge — pages are still held by something the "
      "guest did not release, and no report interval explains it.")
sys.exit(2)
