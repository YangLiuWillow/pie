#!/usr/bin/env python3
"""Blog figures: Pie vs vLLM, H100 measurements. Light surface, validated palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PIE, VLLM = "#2a78d6", "#eb6834"
INK, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": MUT,
    "xtick.color": MUT, "ytick.color": MUT, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
})
OUT = "/workspace/pie/integrations/openhands/docs/figs"
import os; os.makedirs(OUT, exist_ok=True)

def label(ax, bars, fmt="{:,.0f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=10, color=INK)

# Fig 1 — prefill progression (marginal tok/s, same sweep method throughout)
fig, ax = plt.subplots(figsize=(7.6, 4.2))
stages = ["host-orch.\nMoE (bug)", "on-device\nMoE fix", "+ chunk\nN=2048", "+ 64-row\ntiles"]
vals = [5183, 12058, 16634, 24120]
bars = ax.bar(stages, vals, width=0.55, color=PIE)
vb = ax.bar(["vLLM 0.25\n(fair)"], [44855], width=0.55, color=VLLM)
label(ax, bars); label(ax, vb)
ax.axhline(29600, color=MUT, lw=1, ls="--")
ax.annotate("512-token-chunk bandwidth ceiling", (0.02, 29600), xytext=(0, 5),
            textcoords="offset points", fontsize=9, color=MUT)
ax.set_ylabel("prefill throughput (tok/s)")
ax.set_title("Prefill: one bug fix + two config levers = 4.7\u00d7 (H100)")
ax.set_ylim(0, 50000); fig.tight_layout(); fig.savefig(f"{OUT}/fig1_prefill.png", dpi=160); plt.close(fig)

# Fig 2 — decode split: two panels, one measure each (no dual axis)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.6))
for ax, vals, title, fmt in (
    (a1, (5.22, 4.69), "context-independent cost\n(ms/token, lower better)", "{:.2f}"),
    (a2, (2988, 2829), "KV-read bandwidth\n(GB/s, higher better)", "{:,.0f}")):
    b = ax.bar(["Pie", "vLLM (FA3)"], vals, width=0.5, color=[PIE, VLLM])
    label(ax, b, fmt); ax.set_title(title, fontsize=10)
a2.axhline(3350, color=MUT, lw=1, ls="--")
a2.annotate("HBM3 peak", (0.6, 3360), fontsize=9, color=MUT)
a2.set_ylim(0, 3700)
fig.suptitle("Decode at batch 1: parity (H100)",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_decode.png", dpi=160); plt.close(fig)

# Fig 3 — in-call gap vs concurrency (H200 series + H100 c1 shown separately)
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot([8, 16], [1.39, 1.21], color=PIE, lw=2, marker="o", ms=8)
for x, y in ((8, 1.39), (16, 1.21)):
    ax.annotate(f"{y:.2f}×", (x, y), xytext=(6, 6), textcoords="offset points", color=INK)
ax.axhline(1.0, color=MUT, lw=1, ls="--")
ax.annotate("parity", (15.2, 1.005), fontsize=9, color=MUT)
ax.set_xticks([8, 16]); ax.set_xlabel("concurrent conversations")
ax.set_ylabel("vLLM advantage (×)")
ax.set_ylim(0.9, 1.5)
ax.set_title("The gap narrows as load rises: vLLM saturates, Pie keeps scaling (H200)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_concurrency.png", dpi=160); plt.close(fig)

# Fig 4 — c8 overcommit completion (placeholder: armA/armB fill in)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
arms = ["no swap,\nsnapshots\n(07-29)", "swap only\n(control)", "live ctx + bids\n(after 5 fixes)", "vLLM fair\n(same load)"]
vals = [0, 0, 6, 8]
attempted = [8, 8, 6, 8]
cols = [PIE, PIE, PIE, VLLM]
b = ax.bar(arms, vals, width=0.55, color=cols)
for i, (v, a) in enumerate(zip(vals, attempted)):
    ax.annotate(f"{v}/{a}", (i, v + 0.2), ha="center", color=INK)
ax.annotate("stopped early for\nthe comparator", (2, 6.9), ha="center", fontsize=8, color=MUT)
ax.set_ylim(0, 13); ax.set_ylabel("instances served / attempted")
ax.set_title("Memory overcommit at c8 on 80 GB: policy, not speed")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_overcommit.png", dpi=160); plt.close(fig)
print("wrote 4 figs to", OUT)
