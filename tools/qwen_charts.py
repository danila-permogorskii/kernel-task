#!/usr/bin/env python3
"""Charts on the real Qwen3.8-27B shapes (tools/qwen_estimate.py, balanced digits).

  qwen1_memory.png   one A100 80GB: weights, KV + GDN pool, reserve; sessions of 55K
  qwen2_node.png     sessions, one-user speed, total per card: today vs R = 8 / 16 / 32

    python tools/qwen_charts.py          # Russian labels
    python tools/qwen_charts.py --en     # English labels, *_en.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import physics_charts as pc
import qwen_estimate as qe

RANKS = (8, 16, 32)
COLORS = {8: pc.C_R8, 16: pc.C_R16, 32: "#eda100"}
GiB = 2 ** 30
EN = "--en" in sys.argv
SUF = "_en" if EN else ""


def L(ru, en):
    return en if EN else ru


def ring(R, k):
    return qe.node(R, qe.balanced, k)


def fig_memory():
    rows = [(L("dense BF16\n(сегодня)", "dense BF16\n(today)"), pc.W_TARGET,
             pc.KV_POOL + pc.GDN_SLOTS * pc.GDN_SLOT, pc.sessions_dense())]
    for R in RANKS:
        d = ring(R, qe.TARGET)
        w = d["weights"] * GiB
        rows.append((L(f"кольцо R = {R}", f"ring R = {R}"), w, pc.BUDGET - w - pc.OTHER_MEM,
                     d["sess"]))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    parts = [(L("веса", "weights"), pc.INK2),
             (L("KV-кэш + состояния GDN", "KV cache + GDN states"), pc.C_VIOLET),
             (L("драфтер, графы, резерв", "drafter, graphs, reserve"), pc.LIGHT)]
    print("qwen1: weights GiB, pool GiB, sessions")
    for i, (name, w, pool, n) in enumerate(rows):
        y = len(rows) - 1 - i
        left = 0.0
        for size, (lab, color) in zip((w, pool, pc.OTHER_MEM), parts):
            ax.barh(y, size / GiB, left=left / GiB, color=color, height=0.55, edgecolor=pc.SURFACE,
                    lw=2, label=lab if i == 0 else None)
            if size / GiB > 4:
                ax.text((left + size / 2) / GiB, y, f"{size / GiB:.1f}", ha="center", va="center",
                        color=pc.SURFACE if color != pc.LIGHT else pc.INK, fontsize=10,
                        fontweight="bold")
            left += size
        ax.text(left / GiB + 1, y, L(f"≈ {n:.1f} сессий по 55K", f"≈ {n:.1f} sessions of 55K"),
                va="center", fontsize=10.5, color=pc.INK)
        print(f"  {name.splitlines()[0]:14s} {w / GiB:5.1f} {pool / GiB:5.1f} {n:5.1f}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set(xlim=(0, 100),
           xlabel=L("GiB на одной A100 80GB (бюджет 0.94 × 79.3)",
                    "GiB on one A100 80GB (budget 0.94 × 79.3)"),
           title=L("Qwen3.8-27B на A100: память при реальных формах слоёв",
                   "Qwen3.8-27B on A100: memory with the real layer shapes"))
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, fontsize=9.5)
    fig.text(0.5, 0.955, L("Как читать: одна полоса — одна карта. Тёмное — веса, фиолетовое — "
                           "место под сессии. Справа — сколько сессий помещается.",
                           "How to read: one bar is one card. Dark is weights, violet is room "
                           "for sessions. On the right: how many sessions fit."),
             ha="center", fontsize=10.5, color=pc.INK2)
    pc.save(fig, f"qwen1_memory{SUF}.png")


def fig_node():
    groups = ([L("сегодня:\ndense BF16", "today:\ndense BF16")]
              + [L(f"кольцо\nR = {R}", f"ring\nR = {R}") for R in RANKS])
    colors = [pc.C_DENSE] + [COLORS[R] for R in RANKS]
    now = {R: ring(R, qe.NOW) for R in RANKS}
    tgt = {R: ring(R, qe.TARGET) for R in RANKS}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    print("qwen2: sessions, one now/target, total now/target")
    for R in RANKS:
        print(f"  R={R}: {tgt[R]['sess']:.1f}  {now[R]['one']:.0f}/{tgt[R]['one']:.0f}  "
              f"{now[R]['total']:.0f}/{tgt[R]['total']:.0f}")

    def panel(ax, today, key, title, unit, fmt, pair=True):
        ax.bar(0, today, color=colors[0], width=0.6)
        ax.text(0, today, fmt(today), ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        for i, R in enumerate(RANKS, start=1):
            if not pair:
                v = tgt[R][key]
                ax.bar(i, v, color=colors[i], width=0.6)
                ax.text(i, v, fmt(v), ha="center", va="bottom", fontsize=10.5, fontweight="bold")
                continue
            a, b = now[R][key], tgt[R][key]
            ax.bar(i - 0.16, a, color=colors[i], alpha=0.35, width=0.3, hatch="//",
                   edgecolor=colors[i], lw=0)
            ax.bar(i + 0.16, b, color=colors[i], width=0.3)
            ax.text(i - 0.16, a, fmt(a), ha="center", va="bottom", fontsize=9, color=pc.INK2)
            ax.text(i + 0.16, b, fmt(b), ha="center", va="bottom", fontsize=10.5,
                    fontweight="bold")
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.margins(y=0.15)

    panel(axes[0], pc.sessions_dense(), "sess",
          L("Сколько сессий по 55K\nпомещается на карту", "How many 55K sessions\nfit on one card"),
          L("сессий", "sessions"), lambda v: f"{v:.1f}", pair=False)
    panel(axes[1], pc.MEAS_PER[1], "one",
          L("Скорость для одного\nпользователя", "Speed seen by\none user"),
          L("токенов/с", "tokens/s"), lambda v: f"{v:.0f}")
    panel(axes[2], pc.MEAS_AGG[4], "total",
          L("Сколько карта выдаёт\nвсем вместе", "Total output\nof one card"),
          L("токенов/с на карту", "tokens/s per card"), lambda v: f"{v:.0f}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=pc.C_DENSE),
               plt.Rectangle((0, 0), 1, 1, color=pc.MUTED, alpha=0.35, hatch="//"),
               plt.Rectangle((0, 0), 1, 1, color=pc.MUTED)]
    fig.legend(handles, [L("сегодня: замер на узле (DFlash2, ~55K)",
                           "today: measured on the node (DFlash2, ~55K)"),
                         L("кольцо, кернел как есть (оценка)", "ring, kernel as is (estimate)"),
                         L("кольцо, доведённый кернел (оценка)", "ring, tuned kernel (estimate)")],
               loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(L("Qwen3.8-27B на одной A100, контекст 55K: реальные формы слоёв",
                   "Qwen3.8-27B on one A100, 55K context: real layer shapes"),
                 fontsize=14, fontweight="bold", y=1.03)
    fig.text(0.5, 0.965, L("Как читать: три вопроса, у каждого своя шкала. Синий — сегодня. "
                           "Светлый столбик — наш кернел как есть, яркий — после доводки. "
                           "Сбалансированное разложение на цифры.",
                           "How to read: three questions, each on its own scale. Blue is today. "
                           "Light bar: our kernel as is; bright bar: after tuning. "
                           "Balanced factorisation into digits."),
             ha="center", fontsize=10.5, color=pc.INK2)
    pc.save(fig, f"qwen2_node{SUF}.png")


if __name__ == "__main__":
    fig_memory()
    fig_node()
