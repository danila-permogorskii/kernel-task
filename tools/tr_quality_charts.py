#!/usr/bin/env python3
"""Figures for the quality study (needs matplotlib; the kernel venv has none: run it in any
Python with matplotlib).

    python tools/tr_quality_charts.py results/h100/qwen/quality.json \\
        results/h100/qwen/quality_visual.json results/h100/qwen/quality
  q1_error_vs_size.png   kept energy vs compression: ring (every real matrix), SVD same size, controls
  q1_spectra.png         singular values of one layer's matrices vs a random matrix
  q1_patches.png         a 48 x 48 block of a real matrix next to its approximations
  q2_tensorization.png   kept energy per tensorization variant, R = 8 and 16
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

qual = json.loads(Path(sys.argv[1]).read_text())
vis = json.loads(Path(sys.argv[2]).read_text())
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
rows = qual["rows"]
kept = lambda e: 100 * (1 - e * e)  # noqa: E731
C = {"ring": "#2a6fdb", "svd": "#e07b00", "gauss": "#888888", "syn": "#1a9e5b"}

# ---- Q1: kept energy vs compression --------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
real = [r for r in rows if r["layer"] >= 0 and r["tensorization"] == "balanced"]
ax.scatter([r["compression"] for r in real], [kept(r["err_tr"]) for r in real], s=12,
           color=C["ring"], alpha=0.5, label="tensor ring, real Qwen matrices")
ax.scatter([r["compression"] for r in real], [kept(r["err_svd_eq"]) for r in real], s=12,
           color=C["svd"], alpha=0.5, marker="s", label="truncated SVD, same parameter count")
for name, col, lab in (("gaussian", C["gauss"], "ring on a random matrix (no structure)"),
                       ("synthetic_ring_R8", C["syn"], "ring on a matrix that IS a rank-8 ring")):
    g = sorted((r for r in rows if r["matrix"] == name), key=lambda r: r["R"])
    comp = [5120 * 17408 / r["params"] for r in g]
    ax.plot(comp, [kept(r["err_tr"]) for r in g], "-o", color=col, label=lab)
for R in sorted({r["R"] for r in real}):
    c = next(r["compression"] for r in real if r["R"] == R)
    ax.axvline(c, color="#ccc", lw=0.8, zorder=0)
    ax.text(c, 101, f"R={R}", ha="center", va="bottom", fontsize=9)
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("compression (dense parameters / factor parameters)")
ax.set_ylabel("energy of W kept, %  (100 = exact)")
ax.set_ylim(-3, 108)
ax.set_title("Q1: how much of a real Qwen3.8-27B matrix a tensor ring keeps")
ax.legend(fontsize=8, loc="center right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "q1_error_vs_size.png", dpi=150)

# ---- spectra --------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for name, s in vis["spectra"].items():
    x = [(i + 1) / s["n"] for i in s["index"]]
    ax.plot(x, s["sigma"], lw=1.5, color=C["gauss"] if "gaussian" in name else None,
            ls="--" if "gaussian" in name else "-", label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("singular value index / min(rows, cols)")
ax.set_ylabel("σ_i / σ_1")
ax.set_title(f"Singular values, layer {vis['layer']}: flat like noise, no low-rank structure")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(out / "q1_spectra.png", dpi=150)

# ---- patches --------------------------------------------------------------------------------
pt = vis["patches"]
lim = max(abs(v) for row in pt["original"] for v in row)
fig, axes = plt.subplots(1, len(pt), figsize=(4 * len(pt), 4.2))
for axi, (name, m) in zip(axes, pt.items()):
    im = axi.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axi.set_title(name, fontsize=9)
    axi.set_xticks([])
    axi.set_yticks([])
fig.colorbar(im, ax=axes, shrink=0.8)
fig.suptitle(f"Layer {vis['layer']} {vis['matrix']} {vis['shape'][0]}x{vis['shape'][1]}: "
             "top-left 48x48 block, same colour scale")
fig.savefig(out / "q1_patches.png", dpi=150, bbox_inches="tight")

# ---- Q2: tensorization ----------------------------------------------------------------------
q2 = [r for r in rows if r["layer"] >= 0 and r["R"] in (8, 16)]
layers = sorted({r["layer"] for r in q2 if r["tensorization"] != "balanced"})
variants = ["balanced", "balanced_out_reversed", "skewed", "balanced_random_perm"]
labels = ["balanced", "out reversed", "skewed", "random perm"]
if layers:
    groups = defaultdict(dict)
    for r in q2:
        if r["layer"] in layers:
            groups[(r["layer"], r["matrix"], r["R"])][r["tensorization"]] = r
    keys = [k for k in sorted(groups) if len(groups[k]) == len(variants)]
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.9 * len(keys) / 2 + 3), 7), sharey=False)
    for axi, R in zip(axes, (8, 16)):
        ks = [k for k in keys if k[2] == R]
        w = 0.8 / len(variants)
        for vi, (v, lab) in enumerate(zip(variants, labels)):
            axi.bar([i + vi * w for i in range(len(ks))], [kept(groups[k][v]["err_tr"]) for k in ks],
                    width=w, label=lab)
        axi.plot([i + 0.4 - w / 2 for i in range(len(ks))],
                 [kept(groups[k]["balanced"]["err_svd_eq"]) for k in ks], "k_", ms=14,
                 mew=2, label="SVD same size")
        axi.set_xticks([i + 0.4 - w / 2 for i in range(len(ks))])
        axi.set_xticklabels([f"L{k[0]}\n{k[1]}" for k in ks], fontsize=7)
        axi.set_ylabel(f"R={R}: energy kept, %")
        axi.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8, ncol=5)
    axes[0].set_title("Q2: does the tensorization change what the ring keeps?")
    fig.tight_layout()
    fig.savefig(out / "q2_tensorization.png", dpi=150)
print("figures in", out)
