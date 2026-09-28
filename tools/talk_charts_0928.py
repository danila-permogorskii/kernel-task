#!/usr/bin/env python3
"""Two charts for the talk from the 2026-09-28 H100 measurements (results/h100/qwen/*.json).

  step_speed_energy[_en].png  one decode step of all linear layers of Qwen3.8-27B + lm_head:
                              ms per step and J per token, dense BF16 / FP8 / INT4 / ring R 8, 16
  quality_real[_en].png       left: share of a real Qwen matrix a ring / SVD of the same size
                              keeps (median over MLP matrices of 12 layers) vs a random matrix;
                              right: output error on real text, layer 0 (ring, SVD, both
                              activation-aware)

    python tools/talk_charts_0928.py          # Russian labels
    python tools/talk_charts_0928.py --en     # English labels
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EN = "--en" in sys.argv
ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "results" / "h100" / "qwen"
OUT = ROOT / "kernel-design" / "physics"
SUF = "_en" if EN else ""


def L(ru, en):
    return en if EN else ru


# ---- 1. decode step ---------------------------------------------------------------------------
chain = json.loads((Q / "model_chain_lowbit.json").read_text())["rows"]
names = {"dense": L("dense BF16", "dense BF16"), "fp8": "dense FP8", "int4": "dense INT4",
         "ring8": L("кольцо R=8", "ring R=8"), "ring16": L("кольцо R=16", "ring R=16")}
colors = {"dense": "#7f7f7f", "fp8": "#a0a0a0", "int4": "#c8c8c8", "ring8": "#2a6fdb",
          "ring16": "#8fb3ee"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, key, ylab in ((axes[0], "graph_ms", L("мс на шаг", "ms per step")),
                      (axes[1], "joules_per_token", L("Дж на токен", "J per token"))):
    for ti, T in enumerate((1, 8)):
        for vi, v in enumerate(names):
            r = next(x for x in chain if x["variant"] == v and x["tokens"] == T)
            xpos = ti * 6 + vi
            ax.bar(xpos, r[key], color=colors[v], label=names[v] if ti == 0 else None)
            ax.text(xpos, r[key], f"{r[key]:.1f}" if key == "graph_ms" else f"{r[key]:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks([2, 8])
    ax.set_xticklabels([L("1 токен за шаг", "1 token per step"), L("8 токенов за шаг", "8 tokens per step")])
    ax.set_ylabel(ylab)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)  # room for the value labels
    ax.grid(alpha=0.3, axis="y")
axes[0].legend(fontsize=8)
axes[0].set_title(L("Время шага: все линейные слои + lm_head", "Step time: all linear layers + lm_head"))
axes[1].set_title(L("Энергия на токен (плата H100)", "Energy per token (H100 board)"))
fig.suptitle(L("Qwen3.8-27B, H100, у каждого слоя свои веса, CUDA graph (измерено 28.09)",
               "Qwen3.8-27B, H100, every layer its own weights, CUDA graph (measured 28 Sep)"),
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT / f"step_speed_energy{SUF}.png", dpi=150)

# ---- 2. quality -----------------------------------------------------------------------------
qual = json.loads((Q / "quality.json").read_text())["rows"]
act = json.loads((Q / "activation.json").read_text())["rows"]
kept = lambda e: 100 * (1 - e * e)  # noqa: E731
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
Rs = [8, 16, 32, 64]
mlp = [r for r in qual if r["layer"] >= 0 and r["tensorization"] == "balanced"
       and r["matrix"].startswith("mlp")]
ring = [statistics.median(kept(r["err_tr"]) for r in mlp if r["R"] == R) for R in Rs]
svd = [statistics.median(kept(r["err_svd_eq"]) for r in mlp if r["R"] == R) for R in Rs]
gau = [kept(next(r["err_tr"] for r in qual if r["matrix"] == "gaussian" and r["R"] == R)) for R in Rs]
comp = [statistics.median(r["compression"] for r in mlp if r["R"] == R) for R in Rs]
xl = [f"R={R}\n÷{c:.0f}" for R, c in zip(Rs, comp)]
ax.plot(xl, ring, "-o", color="#2a6fdb", label=L("кольцо, настоящие веса MLP", "ring, real MLP weights"))
ax.plot(xl, svd, "-s", color="#e07b00", label=L("SVD того же размера", "SVD of the same size"))
ax.plot(xl, gau, "--", color="#888", label=L("кольцо на случайной матрице", "ring on a random matrix"))
for i, (x, y) in enumerate(zip(xl, ring)):  # label above all three curves at this R
    ax.annotate(f"{y:.1f}%", (i, max(ring[i], svd[i], gau[i])), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8, color="#2a6fdb")
ax.set_ylabel(L("сохранено энергии матрицы, %", "energy of the matrix kept, %"))
ax.set_title(L("Готовые веса: кольцо почти ничего не сохраняет", "Pretrained weights: the ring keeps almost nothing"))
ax.set_ylim(0, 100)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
Ra = [r["R"] for r in act]
xa = [f"R={r['R']}\n÷{r['compression']:.0f}" for r in act]
for key, lab, st, col in (("out_err_ring", L("кольцо", "ring"), "-o", "#2a6fdb"),
                          ("out_err_ring_act", L("кольцо, с учётом активаций", "ring, activation-aware"), "--o", "#2a6fdb"),
                          ("out_err_svd", "SVD", "-s", "#e07b00"),
                          ("out_err_svd_act", L("SVD, с учётом активаций", "SVD, activation-aware"), "--s", "#e07b00")):
    ax.plot(xa, [r[key] for r in act], st, color=col, label=lab)
ax.set_ylabel(L("ошибка выхода на реальном тексте", "output error on real text"))
ax.set_ylim(0, 1.05)
ax.set_title(L("Слой 0, настоящие входы: SVD ≫ кольцо", "Layer 0, real inputs: SVD ≫ ring"))
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)  # below the axes
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / f"quality_real{SUF}.png", dpi=150)
print("wrote", OUT / f"step_speed_energy{SUF}.png", OUT / f"quality_real{SUF}.png")
