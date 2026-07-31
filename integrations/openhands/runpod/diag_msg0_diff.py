import sys, subprocess, json
from pathlib import Path
sys.path.insert(0, "/workspace/pie/integrations/openhands")
from benchmarks.swe_bench import build_agent, build_llm
from openhands.sdk import Conversation

HERE = Path(__file__).parent

def dump(iid):
    ws = HERE / f"sweb-{iid}/repo"; ws.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(ws)])
    llm = build_llm("test")

    captured = {}
    orig = llm.completion
    def spy(messages=None, *a, **kw):
        if messages is not None and "messages" not in captured:
            captured["messages"] = [m.model_dump() if hasattr(m, "model_dump") else m
                                    for m in messages]
            captured["tools"] = [t if isinstance(t, dict) else t.model_dump()
                                 for t in (kw.get("tools") or [])]
        return orig(messages=messages, *a, **kw)
    object.__setattr__(llm, "completion", spy)

    agent = build_agent(llm, enable_condenser=False)
    conv = Conversation(agent=agent, workspace=str(ws), max_iteration_per_run=1, visualizer=None)
    conv.send_message("Fix the bug described in the issue.")
    conv.run()

    msgs = captured.get("messages", [])
    return {
        "static": agent.static_system_message,
        "dynamic": agent.dynamic_context,
        "msg0": msgs[0] if msgs else None,
        "tools": captured.get("tools", []),
        "n_messages": len(msgs),
    }

A = dump("astropy__astropy-12907")
B = dump("django__django-11039")

for tag, d in (("A", A), ("B", B)):
    (HERE / f"{tag}.msg0.json").write_text(json.dumps(d["msg0"], indent=1, default=str))
    (HERE / f"{tag}.static.txt").write_text(d["static"] or "")
    (HERE / f"{tag}.dynamic.txt").write_text(d["dynamic"] or "(None)")
    (HERE / f"{tag}.tools.json").write_text(json.dumps(d["tools"], indent=1, default=str))

j = lambda x: json.dumps(x, sort_keys=True, default=str)
print("RESULT static identical: ", A["static"] == B["static"])
print("RESULT dynamic identical:", A["dynamic"] == B["dynamic"],
      "| A:", repr((A["dynamic"] or "")[:120]), "| B:", repr((B["dynamic"] or "")[:120]))
print("RESULT msg0 identical:   ", j(A["msg0"]) == j(B["msg0"]))
print("RESULT tools identical:  ", j(A["tools"]) == j(B["tools"]))
print("n_messages:", A["n_messages"], B["n_messages"])
