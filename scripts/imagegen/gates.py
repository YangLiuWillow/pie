#!/usr/bin/env python3
"""
gates.py -- every parity gate the image/video effort built, in one command.

Six families landed with their own harnesses and their own idioms; this runs
all of them against ONE tree, in a fixed order, and prints one table plus a
machine-readable summary line.  A person should be able to tell at a glance
whether `dev` is healthy.

    CUDA_VISIBLE_DEVICES=3 python scripts/imagegen/gates.py            # all of it
    python scripts/imagegen/gates.py --list                            # the roster
    python scripts/imagegen/gates.py --only flux2-klein --only wan-mini

WHAT A GATE IS HERE
-------------------
A row of the table: a family's harness invoked exactly the way its README
states, with a config file of its OWN (`~/.pie/config.gates-<name>.toml`,
its own `[server] port`, its own scratch dir), and one headline number lifted
out of the harness's own output.  This runner never re-implements a gate's
arithmetic -- every number in the table is printed by the harness or by
`compare.py`, and every PASS/FAIL is the harness's own exit status.

Three rules it does enforce itself, because the failures they prevent have
all been paid for once already:

- **A missing artifact or golden is a SKIP, not a FAIL.**  Each gate names
  the files it needs; when one is absent the row says `skip` and the reason
  names the file and the command that would make it (`pie model import ...`,
  `python <golden>.py --mini`).  A red table should mean a regression.
- **Every gate reaps its server in a `finally`.**  A failed `pie run` has
  left a process holding 35 GiB of a card, and the NEXT gate then died with
  a misleading elastic-memory message.  Each step runs in its own process
  session so the whole tree can be killed, and afterwards the runner sweeps
  `/proc` for any `pie` still carrying THIS gate's config path.  It never
  touches a pie belonging to anyone else -- the config path is the marker.
- **One card.**  `CUDA_VISIBLE_DEVICES` is inherited and every config says
  `device = ["cuda:0"]`, so the runner drives whatever card the caller gave
  it and nothing else.

EXIT CODE
---------
0 when no gate failed (skips do not fail the run), 1 otherwise.  The last
line is `gates:` followed by `key=value` pairs and a per-gate `name=status`
list, for a CI step that wants to read it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GOLDEN = os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden")
ARTIFACTS = os.environ.get("PIE_IMAGEGEN_ARTIFACTS", "/root/.cache/pie-imagegen")
VENV = "/root/.venv/imagegen/bin/python"

PASS, FAIL, SKIP, ERROR = "pass", "FAIL", "skip", "ERROR"


# ----------------------------------------------------------------------------
# the shape of a gate
# ----------------------------------------------------------------------------

class Gate:
    """One row of the table.

    `needs` is a list of `(path, how to make it)`; a path may be a glob.  If
    any is missing the gate is skipped and the first missing one's `how` is
    the reason.  `steps` are `(label, argv)` builders taking the runner's
    context; `readout` turns the concatenated output into the measured
    string the table prints.
    """

    def __init__(self, name, wraps, expected, needs, steps, readout,
                 config=None, timeout=1800, note=""):
        self.name = name
        self.wraps = wraps
        self.expected = expected
        self.needs = needs
        self.steps = steps
        self.readout = readout
        self.config = config          # None, or the kwargs for `write_config`
        self.timeout = timeout
        self.note = note


# ----------------------------------------------------------------------------
# configs: one file, one port, one scratch dir, per gate
# ----------------------------------------------------------------------------

CONFIG = """\
# Written by `scripts/imagegen/gates.py` for the `{name}` gate. PRIVATE to this
# run: its own port, its own scratch dir. Rewritten every time; edit gates.py,
# not this file.

[server]
host = "127.0.0.1"
port = {port}
verbose = false

[model]
name = "gates-{name}"
model = "{model}"

[engine]
graphs = "{graphs}"
type = "cuda_native"
device = ["cuda:0"]          # whatever CUDA_VISIBLE_DEVICES selects
tensor_parallel_size = 1
activation_dtype = "bfloat16"
gpu_mem_utilization = {mem}
# rows x submit depth, never the 4096 default: a 1024^2 job dies on its second
# fire at the default ("wants 8192 kv tokens in one slot").
max_model_len = {rows}

[runtime]
request_timeout = "{timeout}"

[sandbox]
allow_fs = true
allow_network = false
fs_scratch_dir = "{scratch}"
"""


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def write_config(ctx, gate) -> str:
    """The gate's own config file, at the first free port from its base."""
    spec = dict(gate.config)
    port = spec.pop("port")
    while not port_free(port):
        port += 1
    # The sandbox's scratch root is a directory of its OWN, never the gate's
    # `--out`. `flux2_klein_parity.py` hands its case to the running instance
    # by watching this root for the per-process directory pie makes and taking
    # the newest entry; sharing the root with `--out` makes the harness's own
    # `pie.stdout` the newest entry and it copies the case into a file.
    scratch = os.path.join(ctx.out, gate.name, "scratch")
    os.makedirs(scratch, exist_ok=True)
    text = CONFIG.format(
        name=gate.name, port=port, scratch=scratch,
        model=spec.pop("model"),
        graphs=spec.pop("graphs", "on"),
        mem=spec.pop("mem", 0.90),
        rows=spec.pop("rows", 32768),
        timeout=spec.pop("timeout", "600s"),
    )
    assert not spec, f"{gate.name}: unknown config keys {sorted(spec)}"
    path = os.path.join(ctx.config_dir, f"config.gates-{gate.name}.toml")
    with open(path, "w") as f:
        f.write(text)
    return path


# ----------------------------------------------------------------------------
# running a step, and reaping what it leaves
# ----------------------------------------------------------------------------

def stray_pies(marker: str) -> list[tuple[int, str]]:
    """Every live `pie` whose command line carries `marker`.

    The marker is this gate's own config path, so a server another agent on
    this box is running is never a candidate: it was started with a different
    file.  A `pie run` boots its engine in-process, so the pie process IS the
    server holding the card.
    """
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                raw = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        argv = [piece for piece in raw.split("\0") if piece]
        if not argv or marker not in raw:
            continue
        if os.path.basename(argv[0]) != "pie":
            continue
        found.append((int(entry), " ".join(argv)[:160]))
    return found


def reap(pid: int | None, marker: str, log) -> None:
    """Kill the step's whole session, then sweep for a pie it orphaned."""
    if pid is not None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    for _ in range(3):
        strays = stray_pies(marker)
        if not strays:
            return
        for spid, cmd in strays:
            log(f"[reap] killing stray pie {spid}: {cmd}")
            try:
                os.kill(spid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(1.0)
    left = stray_pies(marker)
    if left:
        log(f"[reap] WARNING {len(left)} pie process(es) survived SIGKILL: "
            f"{[pid for pid, _ in left]}")


def step(ctx, argv, cwd, timeout, marker, log) -> tuple[int, str]:
    """One command, in its own session, reaped whatever happens."""
    log("$ " + " ".join(argv))
    env = dict(os.environ)
    env.setdefault("PIE_IMAGEGEN_GOLDEN", GOLDEN)
    env.setdefault("PIE_IMAGEGEN_ARTIFACTS", ARTIFACTS)
    proc = None
    text = ""
    try:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            text = proc.communicate(timeout=timeout)[0] or ""
            code = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            text = (proc.communicate()[0] or "") + f"\n[gates] TIMEOUT after {timeout}s\n"
            code = 124
    except FileNotFoundError as why:
        text, code = f"[gates] {why}\n", 127
    finally:
        reap(proc.pid if proc else None, marker, log)
    for line in text.splitlines():
        log("  " + line)
    return code, text


# ----------------------------------------------------------------------------
# readouts: the numbers the harnesses print, lifted out
# ----------------------------------------------------------------------------

COS = re.compile(r"worst cos ([0-9]*\.?[0-9]+)")
PSNR = re.compile(r"PSNR\(([^)]*)\)\s*=\s*(-?[0-9.]+|inf|nan) dB")
CLAIM = re.compile(r"\[claim\] (PASS|FAIL) (.+)")
SECTION = re.compile(r"=+ ([a-z_0-9]+): ")


def worst_cos(text: str) -> str:
    hits = [float(m) for m in COS.findall(text)]
    return f"cos {min(hits):.6f}" if hits else "cos ?"


def psnrs(text: str, keep: str = "") -> list[tuple[str, float]]:
    out = []
    for what, value in PSNR.findall(text):
        if keep and keep not in what:
            continue
        out.append((what.strip(), float(value)))
    return out


def row_cos(text: str, key: str) -> float | None:
    """The cosine column of one `compare.py` row (its last field)."""
    hit = re.search(rf"^\s*{re.escape(key)}\s+\(.*?([0-9]\.[0-9]+)(?:\s+FAIL:\S*)?\s*$",
                    text, re.M)
    return float(hit.group(1)) if hit else None


def claims(text: str) -> list[str]:
    return [f"{verdict.lower()}: {what.split(':')[0].strip()}"
            for verdict, what in CLAIM.findall(text)]


def sectioned_cos(text: str) -> str:
    """`zimage_parity.py gate` banners each mode; report each mode's own cos."""
    parts, name = {}, None
    for line in text.splitlines():
        hit = SECTION.search(line)
        if hit:
            name = hit.group(1)
            continue
        for value in COS.findall(line):
            if name:
                parts[name] = min(parts.get(name, 9.0), float(value))
    if not parts:
        return worst_cos(text)
    return "  ".join(f"{k} {v:.6f}" for k, v in parts.items())


def plain(text: str) -> str:
    return worst_cos(text)


def guidance_readout(text: str) -> str:
    """`mini_dit_parity.py guidance`: the three claims, worst case each."""
    def verdict(pattern: str, label: str) -> str | None:
        hits = re.findall(pattern, text)
        if not hits:
            return None
        ok = all(v == "PASS" for v in hits)
        return f"{label} {'pass' if ok else 'FAIL'}"

    bits = []
    for pattern, label in (
        (r"(PASS|FAIL) s=0 the combine (?:is|differs)", "peer is the other group:"),
        (r"(PASS|FAIL) guidance moves|(PASS|FAIL) s=0 and s=1 agree", "guidance moves:"),
        (r"(PASS|FAIL) s=2 is", "affine in s:"),
    ):
        got = verdict(pattern.replace("|(PASS|FAIL)", "|"), label)
        if got:
            bits.append(got)
    moved = re.search(r"moves the velocity by rel ([0-9.]+)", text)
    if moved:
        bits.append(f"moves {moved.group(1)}")
    return "  ".join(bits) or "no guidance lines"


def zimage_turbo_readout(text: str) -> str:
    bits = [sectioned_cos(text)]
    for _, value in psnrs(text, keep="golden latent decode"):
        bits.append(f"PSNR {value:.2f} dB")
    return "  ".join(bits)


def klein_readout(text: str) -> str:
    # `worst cos` here is the minimum over the GATED readings: `text.hidden`,
    # `dit.step0.out` and `probe.step{0..3}.out`, all at --cos-tol 0.999.
    bits = [worst_cos(text)]
    walk = re.search(r"latent\.final\s+cos ([0-9.]+)", text)
    if walk:
        bits.append(f"latent.final {float(walk.group(1)):.6f}")
    for _, value in psnrs(text, keep="pie.png"):
        bits.append(f"PSNR {value:.2f} dB")
    return "  ".join(bits)


def rust_cos_readout(text: str) -> str:
    bits = []
    for side in ("decode", "encode"):
        hit = re.search(rf"^{side} .*cos ([0-9.]+)", text, re.M)
        if hit:
            bits.append(f"{side} {float(hit.group(1)):.6f}")
    if not bits:
        return worst_cos(text)
    return "  ".join(bits)


def zimage_vae_readout(text: str) -> str:
    bits = []
    guest = re.search(r"\[compare\] cos ([0-9.]+)", text)
    if guest:
        bits.append(f"guest {float(guest.group(1)):.6f}")
    host = rust_cos_readout(text)
    if host != "cos ?":
        bits.append(f"host {host}")
    return "  ".join(bits) or "cos ?"


def wan_readout(text: str) -> str:
    bits = [worst_cos(text)]
    move = re.search(r"moves the velocity by ([0-9.]+)", text)
    if move:
        bits.append(f"prompt moves {float(move.group(1)):.4f}")
    return "  ".join(bits)


def hy3_readout(text: str) -> str:
    velocity = row_cos(text, "image_out.velocity.rows")
    bits = [f"velocity {velocity:.6f}"] if velocity is not None else []
    bits.append(worst_cos(text) + (" worst" if velocity is not None else ""))
    bits += claims(text)
    return "  ".join(bits)


def ltx_readout(text: str) -> str:
    bits = [worst_cos(text)]
    if "MUST MOVE" in text:
        bits.append("conditioning moves" if "FAILED" not in text
                    else "conditioning DID NOT move")
    return "  ".join(bits)


def wan_video_readout(text: str) -> str:
    bits = []
    host = rust_cos_readout(text)
    if host != "cos ?":
        bits.append(f"vae {host}")
    cold = re.search(r"cacheless frame \d+: cos ([0-9.]+)", text)
    if cold:
        bits.append(f"cacheless {float(cold.group(1)):.4f}")
    clips = re.findall(r"\[gates\] clip: (\S+)", text)
    size = re.search(r"\[gates\] (\d+)x(\d+) mp4, (\d+) frames", text)
    if size:
        bits.append(f"{size.group(1)}x{size.group(2)}x{size.group(3)} mp4")
    # The ungated run and the guided one, in step order. Guidance must MOVE
    # the latent, or the second branch conditioned nothing: `s = 5` on an
    # undistilled row widens the distribution, and a std that did not budge
    # is a guided run that was not guided.
    stds = [float(v) for v in re.findall(r'"std":([0-9.]+)', text)]
    guided = re.findall(r'"cfg":(true|false)', text)
    if len(stds) >= 2 and guided[:2] == ["false", "true"]:
        moved = abs(stds[1] - stds[0]) / max(stds[0], 1e-9)
        verdict = "pass" if moved > 0.05 else "FAIL"
        bits.append(f"guided std {stds[0]:.3f} -> {stds[1]:.3f} ({verdict})")
    elif guided:
        bits.append(f"cfg runs {','.join(guided)}")
    if clips:
        bits.append(clips[-1])
    return "  ".join(bits) or "no clip"


def t2i_readout(text: str) -> str:
    png = re.search(r"\[gates\] picture: (\S+)", text)
    size = re.search(r"\[gates\] (\d+)x(\d+) PNG", text)
    bits = []
    if size:
        bits.append(f"{size.group(1)}x{size.group(2)} PNG")
    if png:
        bits.append(png.group(1))
    return "  ".join(bits) or "no picture"


# ----------------------------------------------------------------------------
# the roster
# ----------------------------------------------------------------------------

def g(*parts) -> str:
    return os.path.join(GOLDEN, *parts)


def a(name) -> str:
    return os.path.join(ARTIFACTS, name)


IMPORT = "pie model import"


def harness(script, *args):
    """A step that runs one of this directory's harnesses."""
    def build(ctx, gate):
        argv = [ctx.python, os.path.join(HERE, script)]
        argv += [piece.format(out=os.path.join(ctx.out, gate.name),
                              config=ctx.configs[gate.name]) for piece in args]
        return argv, REPO
    return build


def cargo_test(name):
    def build(ctx, gate):
        return ([ctx.cargo, "test", "-p", "engine-cuda", "--features", "cuda",
                 "--test", name, "--", "--nocapture"], REPO)
    return build


def t2v_step(prompt_ids, width, height, frames, steps, seed,
             negative_ids=None, guidance=None, out_name="clip"):
    def build(ctx, gate):
        out = os.path.join(ctx.out, gate.name, out_name)
        os.makedirs(out, exist_ok=True)
        argv = [ctx.python, os.path.join(HERE, "gates.py"), "--text-to-video",
                "--config", ctx.configs[gate.name], "--out", out,
                "--pie", ctx.pie, "--prompt-ids", prompt_ids,
                "--width", str(width), "--height", str(height),
                "--frames", str(frames), "--steps", str(steps),
                "--seed", str(seed)]
        if negative_ids is not None:
            argv += ["--negative-ids", negative_ids]
        if guidance is not None:
            argv += ["--guidance", str(guidance)]
        return (argv, REPO)
    return build


def t2i_step(prompt, width, height, steps, seed):
    def build(ctx, gate):
        out = os.path.join(ctx.out, gate.name, "picture")
        os.makedirs(out, exist_ok=True)
        return ([ctx.python, os.path.join(HERE, "gates.py"), "--text-to-image",
                 "--config", ctx.configs[gate.name], "--out", out,
                 "--pie", ctx.pie, "--prompt", prompt,
                 "--width", str(width), "--height", str(height),
                 "--steps", str(steps), "--seed", str(seed)], REPO)
    return build


def roster() -> list[Gate]:
    return [
        Gate(
            name="mini-dit",
            wraps="mini_dit_parity.py all, and --euler",
            expected="cos >= 0.9999 (landed 0.99996)",
            needs=[(g("mini-dit", "mini_dit_dump_bf16.npz"),
                    "python scripts/imagegen/mini_dit_ref.py --dump --euler"),
                   (g("mini-dit", "mini_dit_euler_bf16.npz"),
                    "python scripts/imagegen/mini_dit_ref.py --euler"),
                   (a("mini-dit.zt"),
                    f"{IMPORT} $PIE_IMAGEGEN_GOLDEN/mini-dit/ --sku mini-dit-bf16-kv-bf16 "
                    f"--out {a('mini-dit.zt')}")],
            config=dict(port=8601, model=a("mini-dit.zt"), rows=65536, mem=0.60),
            steps=[("one step", harness("mini_dit_parity.py", "all", "--out", "{out}",
                                        "--config", "{config}")),
                   ("four Euler steps", harness("mini_dit_parity.py", "all", "--euler",
                                                "--out", "{out}", "--config", "{config}"))],
            readout=plain,
            timeout=1800,
        ),
        Gate(
            name="guidance",
            wraps="mini_dit_parity.py guidance",
            expected="s=0 IS the unconditional branch of its own fire, guidance "
                     "moves the answer, and the combine is affine in s",
            note="classifier-free guidance ON THE DEVICE: six lanes, two attention "
                 "groups, one fire, combined in the epilogue off "
                 "`intrinsics::peer_velocity`. Needs no golden: every claim is "
                 "checked WITHIN one fire's own answers, because a six-lane fire "
                 "and a three-lane one give the same lane different bf16 answers "
                 "(rel 0.0056) and a cross-fire identity is not one.",
            needs=[(a("mini-dit.zt"),
                    f"{IMPORT} $PIE_IMAGEGEN_GOLDEN/mini-dit/ --sku mini-dit-bf16-kv-bf16 "
                    f"--out {a('mini-dit.zt')}")],
            config=dict(port=8601, model=a("mini-dit.zt"), rows=65536, mem=0.60),
            steps=[("two scales", harness("mini_dit_parity.py", "guidance", "--out", "{out}",
                                          "--config", "{config}"))],
            readout=guidance_readout,
            timeout=1800,
        ),
        Gate(
            name="zimage-mini",
            wraps="zimage_parity.py --mode mini, --mode mini_pad",
            expected=">= 0.9999 (landed 0.999985)",
            needs=[(g("z-image", "zimage_mini.npz"), "python scripts/imagegen/zimage_golden.py --mini"),
                   (g("z-image", "zimage_mini_pad.npz"), "python scripts/imagegen/zimage_golden.py --mini-pad"),
                   (a("z-image-mini.zt"),
                    f"{IMPORT} $PIE_IMAGEGEN_GOLDEN/z-image/ --sku z-image-mini-bf16-kv-bf16 "
                    f"--out {a('z-image-mini.zt')}")],
            config=dict(port=8602, model=a("z-image-mini.zt"), rows=65536, mem=0.60),
            steps=[("mini", harness("zimage_parity.py", "all", "--mode", "mini",
                                    "--out", "{out}", "--config", "{config}")),
                   ("mini-pad", harness("zimage_parity.py", "all", "--mode", "mini_pad",
                                        "--out", "{out}", "--config", "{config}"))],
            readout=plain,
            timeout=1800,
        ),
        Gate(
            name="zimage-turbo",
            wraps="zimage_parity.py gate (text/turbo/chain/steps, then decode)",
            expected="text 0.99999, turbo 0.9998, steps 0.9961-0.9972, PSNR 32.8-33.3 dB",
            needs=[(g("z-image", "zimage_golden.npz"), "python scripts/imagegen/zimage_golden.py --full"),
                   (a("z-image-turbo.zt"),
                    f"{IMPORT} <Z-Image-Turbo snapshot> --sku z-image-turbo-bf16-kv-bf16 "
                    f"--out {a('z-image-turbo.zt')}")],
            config=dict(port=8603, model=a("z-image-turbo.zt"), rows=65536, mem=0.90),
            steps=[("gate", harness("zimage_parity.py", "gate", "--out", "{out}",
                                    "--config", "{config}"))],
            readout=zimage_turbo_readout,
            timeout=5400,
            note="the 8-step endpoint is BELOW the bf16 floor by design; see README",
        ),
        Gate(
            name="flux2-mini",
            wraps="flux2_parity.py all",
            expected="0.99999",
            needs=[(g("flux2", "flux2_mini.npz"), "python scripts/imagegen/flux2_golden.py --mini"),
                   (a("flux2-mini.zt"),
                    f"{IMPORT} $PIE_IMAGEGEN_GOLDEN/flux2/ --sku flux2-mini-bf16-kv-bf16 "
                    f"--out {a('flux2-mini.zt')}")],
            config=dict(port=8604, model=a("flux2-mini.zt"), rows=65536, mem=0.60),
            steps=[("one step", harness("flux2_parity.py", "all", "--out", "{out}",
                                        "--config", "{config}"))],
            readout=plain,
            timeout=1800,
        ),
        Gate(
            name="flux2-klein",
            wraps="flux2_klein_parity.py all",
            expected="per-step velocities >= 0.999, PSNR >= 34 dB",
            needs=[(g("flux2", "flux2_golden.npz"), "python scripts/imagegen/flux2_golden.py --full"),
                   (a("flux2-klein-4b.zt"),
                    f"{IMPORT} <FLUX.2-klein-4B snapshot> --sku flux2-klein-4b-bf16-kv-bf16 "
                    f"--out {a('flux2-klein-4b.zt')}")],
            config=dict(port=8605, model=a("flux2-klein-4b.zt"), rows=32768, mem=0.90),
            steps=[("all", harness("flux2_klein_parity.py", "all", "--out", "{out}",
                                   "--config", "{config}"))],
            readout=klein_readout,
            timeout=5400,
            note="the 4-step endpoint is BELOW the bf16 floor by design; see README",
        ),
        Gate(
            name="flux2-vae",
            wraps="cargo test -p engine-cuda --test the_flux_2_vae_answers_the_reference",
            expected="decode 0.99999, encode 0.99995",
            needs=[(g("flux2", "flux2_vae", "shapes.json"),
                    "python scripts/imagegen/flux2_golden.py --vae"),
                   (a("flux2-klein-4b.zt"),
                    f"{IMPORT} <FLUX.2-klein-4B snapshot> --sku flux2-klein-4b-bf16-kv-bf16 "
                    f"--out {a('flux2-klein-4b.zt')}")],
            config=None,
            steps=[("host gate", cargo_test("the_flux_2_vae_answers_the_reference"))],
            readout=rust_cos_readout,
            timeout=3600,
        ),
        Gate(
            name="zimage-vae",
            wraps="zimage_vae_parity.py all, then the_z_image_vae_answers_the_reference",
            expected="0.99998",
            needs=[(g("z-image", "zimage_vae", "shapes.json"),
                    "python scripts/imagegen/zimage_golden.py --vae"),
                   (a("z-image-turbo.zt"),
                    f"{IMPORT} <Z-Image-Turbo snapshot> --sku z-image-turbo-bf16-kv-bf16 "
                    f"--out {a('z-image-turbo.zt')}")],
            config=dict(port=8606, model=a("z-image-turbo.zt"), rows=32768, mem=0.90),
            steps=[("from a guest", harness("zimage_vae_parity.py", "all", "--out", "{out}",
                                            "--config", "{config}")),
                   ("from the host", cargo_test("the_z_image_vae_answers_the_reference"))],
            readout=zimage_vae_readout,
            timeout=3600,
        ),
        Gate(
            name="wan-mini",
            wraps="wan22_parity.py all (d128), all --pertoken; both include `conditioning`",
            expected="0.999925, and the conditioning must MOVE",
            needs=[(g("wan22", "wan22_mini.npz"), "python scripts/imagegen/wan22_golden.py --mini"),
                   (a("wan22-mini-d128.zt"),
                    f"{IMPORT} <dir with wan22_mini_d128.safetensors> "
                    f"--sku wan22-mini-d128-bf16-kv-bf16 --out {a('wan22-mini-d128.zt')}")],
            config=dict(port=8607, model=a("wan22-mini-d128.zt"), rows=16384, mem=0.60),
            steps=[("scalar timestep", harness("wan22_parity.py", "all", "--out", "{out}",
                                               "--config", "{config}")),
                   ("per-token timestep", harness("wan22_parity.py", "all", "--pertoken",
                                                  "--out", "{out}", "--config", "{config}"))],
            readout=wan_readout,
            timeout=2400,
        ),
        Gate(
            name="h3-mini",
            wraps="h3_parity.py all",
            expected="refined-text / video / audio >= 0.9999",
            needs=[(g("minimax_h3", "h3_mini.npz"), "python scripts/imagegen/h3_golden.py --mini"),
                   (a("h3-mini.zt"),
                    f"{IMPORT} $PIE_IMAGEGEN_GOLDEN/minimax_h3/ "
                    f"--sku minimax-h3-mini-bf16-kv-bf16 --out {a('h3-mini.zt')}")],
            config=dict(port=8608, model=a("h3-mini.zt"), rows=65536, mem=0.60),
            steps=[("refine + denoise", harness("h3_parity.py", "all", "--out", "{out}",
                                                "--config", "{config}"))],
            readout=plain,
            timeout=1800,
        ),
        Gate(
            name="hy3-mini",
            wraps="hy3_parity.py all",
            expected="velocity 0.999997, prefix-conditions claim, prefix-KV-reuse exact",
            needs=[(g("hy3", "hy3_mini.npz"), "python scripts/imagegen/hy3_golden.py --mini"),
                   (g("hy3", "hy3-mini.zt"),
                    f"{IMPORT} $PIE_IMAGEGEN_GOLDEN/hy3/artifact "
                    f"--sku hunyuanimage3-mini-bf16-kv-bf16 --out {g('hy3', 'hy3-mini.zt')}")],
            config=dict(port=8609, model=g("hy3", "hy3-mini.zt"), rows=32768, mem=0.30),
            steps=[("three fires + two claims", harness("hy3_parity.py", "all", "--out", "{out}",
                                                        "--config", "{config}"))],
            readout=hy3_readout,
            timeout=1800,
        ),
        Gate(
            name="ltx2-mini",
            wraps="ltx2_parity.py all, all --refine, matters",
            expected="cos >= 0.9999 (the harness's own gate; the README states no number)",
            needs=[(g("ltx25", "ltx2_mini.npz"), "python scripts/imagegen/ltx2_golden.py --mini"),
                   (a("ltx2-mini.zt"),
                    f"{IMPORT} $PIE_IMAGEGEN_GOLDEN/ltx25 --sku ltx25-mini-bf16-kv-bf16 "
                    f"--out {a('ltx2-mini.zt')}")],
            config=dict(port=8610, model=a("ltx2-mini.zt"), rows=4096, mem=0.40),
            steps=[("joint step", harness("ltx2_parity.py", "all", "--out", "{out}",
                                          "--config", "{config}")),
                   ("connectors", harness("ltx2_parity.py", "all", "--refine",
                                          "--out", "{out}", "--config", "{config}")),
                   ("conditioning moves", harness("ltx2_parity.py", "matters",
                                                  "--out", "{out}", "--config", "{config}"))],
            readout=ltx_readout,
            timeout=2400,
        ),
        Gate(
            name="wan-video",
            wraps=("the_wan_2_vae_answers_the_reference, then the model-agnostic "
                   "text-to-video guest on wan22-ti2v-5b.zt"),
            expected="decode cos >= 0.999 per chunk and clip (landed 0.999986); a real mp4, "
                     "and guidance at 5.0 must MOVE the latent",
            needs=[(g("wan22", "wan22_vae", "shapes.json"),
                    "python scripts/imagegen/wan22_golden.py --vae"),
                   (a("wan22-ti2v-5b.zt"),
                    f"{IMPORT} <Wan2.2-TI2V-5B-Diffusers snapshot, with a tokenizer.json "
                    f"pie can compile beside it> --sku wan22-ti2v-5b-bf16-kv-bf16 "
                    f"--out {a('wan22-ti2v-5b.zt')}")],
            # 1950 latent rows plus a 512-row context lane, times the submit
            # depth: a video job wants far more headroom than an image one.
            config=dict(port=8612, model=a("wan22-ti2v-5b.zt"), rows=131072, mem=0.90,
                        timeout="1800s"),
            steps=[("the VAE against the reference",
                    cargo_test("the_wan_2_vae_answers_the_reference")),
                   # umT5's SentencePiece Unigram tokenizer does not compile
                   # in pie (`models::wan_2::tokenizer`), so the prompt goes
                   # in as ids: the reference tokenizer's encoding of
                   # "a red bicycle leaning on a blue wall", the golden's
                   # own prompt.
                   ("prompt -> clip",
                    t2v_step("289,4062,188625,346,291,1350,369,289,15258,21006,1",
                             832, 480, 17, 20, 0)),
                   # THE SAME CLIP, GUIDED. Four lanes — two attention
                   # groups of a context and a video lane — in one fire,
                   # combining `u + s(c - u)` in the epilogue off
                   # `intrinsics::peer_velocity`. This is the undistilled
                   # row, so guidance is the setting the reference actually
                   # uses; the readout reports both latents' std, and they
                   # must differ or the second branch changed nothing.
                   ("the same prompt, guided at 5.0",
                    t2v_step("289,4062,188625,346,291,1350,369,289,15258,21006,1",
                             832, 480, 17, 20, 0,
                             negative_ids="1", guidance=5.0, out_name="guided"))],
            readout=wan_video_readout,
            timeout=5400,
            note=("the prompt goes in as IDS: umT5's SentencePiece Unigram tokenizer does not "
                  "compile in pie, and the artifact was imported from a staged snapshot "
                  "(/root/.cache/pie-imagegen/wan22-stage) whose tokenizer/ is one pie CAN "
                  "compile, because `pie model import` refuses the real one"),
        ),
        Gate(
            name="text-to-image",
            wraps="the model-agnostic guest on flux2-klein-4b.zt, 4 steps at 1024^2",
            expected="a real PNG",
            needs=[(a("flux2-klein-4b.zt"),
                    f"{IMPORT} <FLUX.2-klein-4B snapshot> --sku flux2-klein-4b-bf16-kv-bf16 "
                    f"--out {a('flux2-klein-4b.zt')}")],
            config=dict(port=8611, model=a("flux2-klein-4b.zt"), rows=32768, mem=0.90),
            steps=[("prompt -> picture",
                    t2i_step("a red bicycle leaning on a blue wall", 1024, 1024, 4, 0))],
            readout=t2i_readout,
            timeout=3600,
        ),
    ]


# ----------------------------------------------------------------------------
# the text-to-image gate, which is a `pie run` and then maybe a VAE decode
# ----------------------------------------------------------------------------

def text_to_image(args) -> int:
    """`--text-to-image`: drive the generic guest and end with a real PNG.

    Two exits, and which one the row takes is its own fact (D11): a row whose
    `vae.decode` is a declared reading hands back `image.png` directly; FLUX.2
    today hands back the final latent plus its JSON sidecar, and
    `decode_latent.py` finishes the job with the checkpoint's own autoencoder.
    Either way this prints `[gates] picture: <path>` and its pixel size.
    """
    inferlet = os.path.join(REPO, "tests/inferlets/text-to-image")
    workspace = os.path.dirname(inferlet)
    if not os.environ.get("PIE_INFERLETS_NO_BUILD"):
        done = subprocess.run(["cargo", "build", "-p", "text-to-image", "--release",
                               "--target", "wasm32-wasip2"],
                              cwd=workspace, capture_output=True, text=True)
        if done.returncode != 0:
            sys.stderr.write(done.stderr)
            return 1
    candidates = [os.path.join(workspace, "target/wasm32-wasip2", flavour, "text_to_image.wasm")
                  for flavour in ("release", "debug")]
    present = [p for p in candidates if os.path.exists(p)]
    if not present:
        print(f"[gates] no text-to-image wasm; tried {candidates}")
        return 1
    wasm = max(present, key=os.path.getmtime)
    cmd = [args.pie, "--config", args.config, "run", "--path", wasm,
           "--manifest", os.path.join(inferlet, "Pie.toml"),
           "-o", args.out, "--",
           "--prompt", args.prompt, "--width", str(args.width),
           "--height", str(args.height), "--steps", str(args.steps),
           "--seed", str(args.seed), "--out", "image"]
    print("$ " + " ".join(cmd), flush=True)
    done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    if done.returncode != 0:
        print(f"[gates] pie run failed ({done.returncode})")
        return 1

    png = os.path.join(args.out, "image.png")
    if not os.path.exists(png):
        latent = os.path.join(args.out, "image.latent.f32")
        sidecar = os.path.join(args.out, "image.json")
        if not os.path.exists(latent):
            print(f"[gates] neither {png} nor {latent} exists")
            return 1
        if not os.path.exists(sidecar):
            # the report is the guest's stdout; keep it as the sidecar
            body = [line for line in done.stdout.splitlines() if line.startswith("{")]
            if not body:
                print("[gates] no sidecar and no JSON report on stdout")
                return 1
            with open(sidecar, "w") as f:
                f.write(body[-1])
        snap = sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/*")))
        if not snap:
            print("[gates] no FLUX.2-klein-4B snapshot to decode the latent with")
            return 1
        decode = [sys.executable, os.path.join(HERE, "decode_latent.py"),
                  "--latent", latent, "--sidecar", sidecar,
                  "--model-dir", snap[-1], "--out", png, "--family", "flux2"]
        print("$ " + " ".join(decode), flush=True)
        done = subprocess.run(decode, cwd=REPO, capture_output=True, text=True)
        sys.stdout.write(done.stdout)
        sys.stderr.write(done.stderr)
        if done.returncode != 0 or not os.path.exists(png):
            print(f"[gates] decode_latent.py failed ({done.returncode})")
            return 1

    # A real PNG: the magic, a plausible size, and more than one colour.
    with open(png, "rb") as f:
        head = f.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        print(f"[gates] {png} is not a PNG")
        return 1
    w = int.from_bytes(head[16:20], "big")
    h = int.from_bytes(head[20:24], "big")
    size = os.path.getsize(png)
    print(f"[gates] {w}x{h} PNG, {size} bytes")
    print(f"[gates] picture: {png}")
    if size < 100_000:
        print(f"[gates] suspiciously small for a {w}x{h} photograph")
        return 1
    return 0


# ----------------------------------------------------------------------------
# the text-to-video gate, which is a `pie run` and then a real mp4
# ----------------------------------------------------------------------------

def text_to_video(args) -> int:
    """`--text-to-video`: drive the generic video guest and end with an mp4.

    One exit, not two: a video row must declare a `vae.decode` reading, or
    there is nothing here that can turn a latent volume into frames from
    inside pie, and the guest refuses by name.

    The prompt goes in as IDS.  Wan 2.2's text encoder is umT5, whose
    tokenizer is a SentencePiece Unigram model and `crates/tokenizer`
    compiles BPE pipelines alone (`models::wan_2::tokenizer`), so the
    artifact carries somebody else's vocabulary and `--prompt` on this row
    would condition the DiT on ids it has never seen.  The ids below are
    `T5Tokenizer(Wan2.2-TI2V-5B/tokenizer)("a red bicycle leaning on a blue
    wall")` -- the golden's own prompt, so the clip is comparable with
    `$PIE_IMAGEGEN_GOLDEN/wan22/wan22_golden.mp4`.
    """
    inferlet = os.path.join(REPO, "tests/inferlets/text-to-video")
    workspace = os.path.dirname(inferlet)
    if not os.environ.get("PIE_INFERLETS_NO_BUILD"):
        done = subprocess.run(["cargo", "build", "-p", "text-to-video", "--release",
                               "--target", "wasm32-wasip2"],
                              cwd=workspace, capture_output=True, text=True)
        if done.returncode != 0:
            sys.stderr.write(done.stderr)
            return 1
    candidates = [os.path.join(workspace, "target/wasm32-wasip2", flavour, "text_to_video.wasm")
                  for flavour in ("release", "debug")]
    present = [p for p in candidates if os.path.exists(p)]
    if not present:
        print(f"[gates] no text-to-video wasm; tried {candidates}")
        return 1
    wasm = max(present, key=os.path.getmtime)
    cmd = [args.pie, "--config", args.config, "run", "--path", wasm,
           "--manifest", os.path.join(inferlet, "Pie.toml"),
           "-o", args.out, "--",
           "--prompt-ids", args.prompt_ids, "--width", str(args.width),
           "--height", str(args.height), "--frames", str(args.frames),
           "--steps", str(args.steps), "--seed", str(args.seed), "--out", "clip"]
    # Real guidance: a negative prompt at a scale above 1 runs the second
    # lane pair and the combine happens in the epilogue.
    if args.guidance is not None:
        cmd += ["--guidance", str(args.guidance)]
    if args.negative_ids:
        # The guest names this one `negative_prompt_ids`, and takes it as a
        # comma-separated STRING. A single id needs a trailing comma, or the
        # argv parser hands it over as a JSON integer and the guest refuses.
        ids = args.negative_ids if "," in args.negative_ids else f"{args.negative_ids},"
        cmd += ["--negative_prompt_ids", ids]
    print("$ " + " ".join(cmd), flush=True)
    done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    if done.returncode != 0:
        print(f"[gates] pie run failed ({done.returncode})")
        return 1

    mp4 = os.path.join(args.out, "clip.mp4")
    if not os.path.exists(mp4):
        print(f"[gates] {mp4} does not exist")
        return 1
    # A real mp4: the ISO-BMFF brand, a plausible size, and the frame count
    # the guest reported.
    with open(mp4, "rb") as f:
        head = f.read(12)
    if head[4:8] != b"ftyp":
        print(f"[gates] {mp4} is not ISO base media")
        return 1
    size = os.path.getsize(mp4)
    print(f"[gates] {args.width}x{args.height} mp4, {args.frames} frames, {size} bytes")
    print(f"[gates] clip: {mp4}")
    if size < 20_000:
        print(f"[gates] suspiciously small for a {args.frames}-frame clip")
        return 1
    return 0


# ----------------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------------

class Context:
    def __init__(self, args):
        self.out = args.out
        self.config_dir = args.config_dir
        self.python = args.python
        self.pie = args.pie
        self.cargo = args.cargo
        self.configs: dict[str, str] = {}


def missing(gate) -> tuple[str, str] | None:
    for path, how in gate.needs:
        if glob.glob(path):
            continue
        return path, how
    return None


def run_gate(ctx, gate, log) -> tuple[str, str, str]:
    """-> (status, measured, note).  Never raises: a gate's own explosion is
    an ERROR row, not the end of the table."""
    gone = missing(gate)
    if gone:
        path, how = gone
        return SKIP, "", f"missing {os.path.basename(path.rstrip('/'))} -- make it with `{how}`"
    marker = ctx.configs.get(gate.name, f"gates-{gate.name}")
    text, code = "", 0
    for label, build in gate.steps:
        argv, cwd = build(ctx, gate)
        log(f"\n--- {gate.name}: {label}")
        rc, out = step(ctx, argv, cwd, gate.timeout, marker, log)
        text += out
        code = code or rc
        if rc != 0:
            break
    measured = gate.readout(text)
    if "skipping the VAE parity gate" in text:
        why = re.search(r"skipping the VAE parity gate: ([^\n]+)", text)
        return SKIP, measured, why.group(1) if why else "the host gate skipped itself"
    if code == 124:
        return ERROR, measured, f"timed out after {gate.timeout}s"
    return (PASS if code == 0 else FAIL), measured, gate.note


def render(rows) -> str:
    head = ("gate", "status", "measured", "expected", "note")
    body = [head] + [(r["gate"], r["status"], r["measured"], r["expected"], r["note"])
                     for r in rows]
    width = [max(len(str(row[i])) for row in body) for i in range(len(head))]
    rule = "  ".join("-" * w for w in width)
    lines = ["  ".join(str(cell).ljust(width[i]) for i, cell in enumerate(head)), rule]
    for row in body[1:]:
        lines.append("  ".join(str(cell).ljust(width[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=None,
                    help="run just this gate (repeatable); `--list` names them")
    ap.add_argument("--list", action="store_true", help="the roster, and what each expects")
    ap.add_argument("--out", default="/tmp/imagegen-gates",
                    help="working directory; each gate gets a subdirectory")
    ap.add_argument("--config-dir", default=os.path.expanduser("~/.pie"),
                    help="where the per-gate config files are written")
    ap.add_argument("--python", default=None, help="the interpreter the harnesses run under")
    ap.add_argument("--pie", default=None, help="the pie binary")
    ap.add_argument("--cargo", default="cargo")
    ap.add_argument("--json", default=None, help="also write the table here as JSON")
    ap.add_argument("--log", default=None, help="the full transcript (default: <out>/gates.log)")
    # the text-to-image / text-to-video gates re-enter this file as a step
    ap.add_argument("--text-to-image", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--text-to-video", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--prompt", default="", help=argparse.SUPPRESS)
    ap.add_argument("--prompt-ids", default="", help=argparse.SUPPRESS)
    ap.add_argument("--negative-ids", default="", help=argparse.SUPPRESS)
    ap.add_argument("--guidance", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--width", type=int, default=1024, help=argparse.SUPPRESS)
    ap.add_argument("--height", type=int, default=1024, help=argparse.SUPPRESS)
    ap.add_argument("--frames", type=int, default=17, help=argparse.SUPPRESS)
    ap.add_argument("--steps", type=int, default=4, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    gates = roster()
    if args.list:
        for gate in gates:
            print(f"{gate.name:<15} {gate.wraps}")
            print(f"{'':<15} expects {gate.expected}")
        return 0

    args.python = args.python or (VENV if os.path.exists(VENV) else sys.executable)
    args.pie = args.pie or shutil.which("pie") or os.path.join(REPO, "target/debug/pie")

    if args.text_to_image:
        return text_to_image(args)

    if args.text_to_video:
        return text_to_video(args)

    if args.only:
        names = {gate.name for gate in gates}
        for want in args.only:
            if want not in names:
                print(f"no such gate: {want} (try --list)", file=sys.stderr)
                return 2
        gates = [gate for gate in gates if gate.name in set(args.only)]

    os.makedirs(args.out, exist_ok=True)
    log_path = args.log or os.path.join(args.out, "gates.log")
    transcript = open(log_path, "w")

    def log(line: str) -> None:
        transcript.write(line + "\n")
        transcript.flush()

    if not os.path.exists(args.pie):
        print(f"{args.pie}: no pie binary. `cargo build -p pie --features cuda`, or --pie.",
              file=sys.stderr)
        return 2

    ctx = Context(args)
    print(f"[gates] pie      {args.pie}")
    print(f"[gates] python   {args.python}")
    print(f"[gates] golden   {GOLDEN}")
    print(f"[gates] card     CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}")
    print(f"[gates] log      {log_path}")
    print()

    rows = []
    started_all = time.time()
    for gate in gates:
        if gate.config:
            ctx.configs[gate.name] = write_config(ctx, gate)
        started = time.time()
        print(f"[gates] {gate.name} ...", end=" ", flush=True)
        try:
            status, measured, note = run_gate(ctx, gate, log)
        except Exception as why:                      # a gate never kills the table
            status, measured, note = ERROR, "", f"{type(why).__name__}: {why}"
            log(f"[gates] {gate.name} raised: {why!r}")
        finally:
            if gate.name in ctx.configs:
                reap(None, ctx.configs[gate.name], log)
        took = time.time() - started
        print(f"{status} ({took:.0f}s)")
        rows.append(dict(gate=gate.name, status=status, measured=measured,
                         expected=gate.expected, note=note, seconds=round(took, 1),
                         wraps=gate.wraps,
                         config=ctx.configs.get(gate.name, "")))
    transcript.close()

    print()
    print(render(rows))
    print()
    tally = {PASS: 0, FAIL: 0, SKIP: 0, ERROR: 0}
    for row in rows:
        tally[row["status"]] += 1
    bad = tally[FAIL] + tally[ERROR]
    verdict = "HEALTHY" if bad == 0 else "REGRESSED"
    summary = (f"gates: verdict={verdict} total={len(rows)} pass={tally[PASS]} "
               f"fail={tally[FAIL]} skip={tally[SKIP]} error={tally[ERROR]} "
               f"seconds={time.time() - started_all:.0f} "
               + " ".join(f"{row['gate']}={row['status']}" for row in rows))
    print(summary)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(verdict=verdict, tally=tally, gates=rows), f, indent=2)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
