import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

snap = "/root/.cache/huggingface/hub/models--bigai-NPR--NPR-4B/snapshots/4e108aa0b95103716e4a914ce2323f53bb68e048"
out = "/workspace/NPR-4B-bf16"
os.makedirs(out, exist_ok=True)
total = 0
for f in sorted(os.listdir(snap)):
    src = os.path.join(snap, f)
    if f.endswith(".safetensors"):
        sd = load_file(src)
        sd = {
            k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v)
            for k, v in sd.items()
        }
        total += sum(v.numel() * v.element_size() for v in sd.values())
        save_file(sd, os.path.join(out, f), metadata={"format": "pt"})
        print("converted", f, flush=True)
    elif os.path.isfile(src):
        shutil.copy(src, os.path.join(out, f))

cfgp = os.path.join(out, "config.json")
cfg = json.load(open(cfgp))
cfg["torch_dtype"] = "bfloat16"
json.dump(cfg, open(cfgp, "w"), indent=2)

idxp = os.path.join(out, "model.safetensors.index.json")
if os.path.exists(idxp):
    idx = json.load(open(idxp))
    idx.setdefault("metadata", {})["total_size"] = total
    json.dump(idx, open(idxp, "w"))
print("DONE total bytes:", total)
