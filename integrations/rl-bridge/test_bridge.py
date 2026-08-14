#!/usr/bin/env python3
"""Contract test for pie-openai-bridge: the §3.1 rows a training gateway parses.

Asserts, over real HTTP against a running bridge:
  1. /health -> 200 "ok"
  2. non-streaming /v1/completions with prompt=list[int]:
     exact prompt_token_ids echo, choices[0].token_ids present,
     logprobs.token_logprobs aligned, finish_reason stop|length,
     usage.completion_tokens == len(token_ids), weight_version present
  3. cumulative turn 2 (prompt extends turn 1's prompt+completion):
     usage.prompt_tokens_details.cached_tokens > 0 (bridge hints + snapshots)
  4. streaming form: first chunk prompt_token_ids, delta chunk token_ids,
     usage chunk, [DONE] terminator
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8123"

P = [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 9625, 374, 1083,
     264, 3283, 429, 374, 3881, 369, 1181, 6722, 13, 576, 6722, 315, 9625,
     374, 12095, 13, 576, 6722, 315, 9625, 374, 1083, 264, 3283, 13]
EXTRA = [576, 1196, 4588, 911, 419, 13]


def post(path: str, body: dict, raw: bool = False):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
        return data if raw else json.loads(data)


def main() -> int:
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
        assert r.status == 200
        health = json.loads(r.read())
        assert health["status"] == "ok", health
        assert isinstance(health["weight_version"], int), health
    print(f"1. /health ok (weight_version {health['weight_version']})")

    body = {"prompt": P, "max_tokens": 12, "temperature": 0.0,
            "logprobs": True, "return_token_ids": True}
    r1 = post("/v1/completions", body)
    assert r1["prompt_token_ids"] == P, "prompt_token_ids is not an exact echo"
    assert "weight_version" in r1
    c = r1["choices"][0]
    ids = c["token_ids"]
    assert isinstance(ids, list) and ids
    assert c["finish_reason"] in ("stop", "length")
    lps = c["logprobs"]["token_logprobs"]
    assert len(lps) == len(ids) and all(lp <= 0.0 for lp in lps)
    assert r1["usage"]["completion_tokens"] == len(ids)
    assert r1["usage"]["prompt_tokens"] == len(P)
    print(f"2. turn 1 envelope ok ({len(ids)} tokens, cached="
          f"{r1['usage']['prompt_tokens_details']['cached_tokens']})")

    p2 = P + ids + EXTRA
    r2 = post("/v1/completions", {**body, "prompt": p2})
    cached = r2["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert r2["prompt_token_ids"] == p2
    assert cached > 0, "expected KV reuse on the cumulative turn"
    assert cached == len(P) + len(ids) - 1, (cached, len(P), len(ids))
    print(f"3. turn 2 cumulative reuse ok (cached {cached}/{len(p2)})")

    raw = post("/v1/completions", {**body, "stream": True}, raw=True).decode()
    events = [json.loads(l[6:]) for l in raw.splitlines()
              if l.startswith("data: ") and l != "data: [DONE]"]
    assert raw.rstrip().endswith("data: [DONE]"), "missing [DONE] terminator"
    assert events[0]["prompt_token_ids"] == P, "first chunk must echo prompt ids"
    delta_ids = [t for e in events for ch in e.get("choices", [])
                 for t in ch.get("token_ids", [])]
    assert delta_ids, "no token_ids in stream deltas"
    usage_events = [e for e in events if e.get("usage")]
    assert usage_events and usage_events[-1]["usage"]["completion_tokens"] == len(delta_ids)
    print(f"4. streaming form ok ({len(events)} chunks, {len(delta_ids)} tokens)")

    # R-G3: the response body's weight_version overrides the gateway's, so the
    # bridge is the one that has to tell the truth about which weights served.
    baseline = health["weight_version"]
    try:
        assert post("/admin/weight_version", {"weight_version": 7})["weight_version"] == 7
        # First request after the flush, so no hint of this lineage can have
        # been re-recorded: a version change must invalidate the boundaries
        # taken under the old weights, and this same prompt hit above.
        r5 = post("/v1/completions", {**body, "prompt": p2})
        assert r5["weight_version"] == 7, r5["weight_version"]
        assert r5["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, (
            "KV hints survived a weight-version change")
    finally:
        post("/admin/weight_version", {"weight_version": baseline})
    print("5. weight_version advance ok (stamped + hints flushed)")

    print("BRIDGE_CONTRACT_TEST_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
