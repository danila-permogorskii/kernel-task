#!/usr/bin/env python3
"""Two plain charts for the talk, built on the numbers in physics_charts.py.

  simple1_kernel_window.png   one call of the operator: now, realistic target, limits (ns)
  simple2_node.png            one A100 with Qwen3.8-27B: today vs ring R = 8 / 16

The target kernel is the same in both: ~2.5 us per call at t = 1 and 30% of Tensor Core
peak on bulk work. "Now" is our kernel as measured: ~7 us per call, ~10% of peak.

    python tools/simple_charts.py          # Russian labels
    python tools/simple_charts.py --en     # English labels, simple1_kernel_window_en.png only
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import physics_charts as pc

NS = 1e3  # us -> ns
EN = "--en" in sys.argv


def L(ru, en):
    return en if EN else ru


def num(v):
    s = f"{v:,.0f}"
    return s if EN else s.replace(",", " ")

# ---------------------------------------------------------------- kernel, one call, t = 1
# Measured on H100 (results/h100/kernels.json): design B kernel, V3 alone, empty kernel.
LAUNCH = pc.KERNEL_EMPTY_US


def now_parts(R):
    b = pc.measured_us(R, 1, "B")
    a = pc.case(R, 1, "A")["fused_kernel_us"]   # the fused V3 kernel alone, without the finish
    assert 0 < a < b
    return {"launch": LAUNCH, "work": a - LAUNCH, "finish": b - a}


# ESTIMATE, from H100 microbenchmarks (kernel_work/hopper_floors): a 16-24 KB L2 -> smem copy
# takes ~0.5 us, a dependent mma ~24 cycles; the finish done inside a thread-block cluster.
TARGET_WORK = {8: 1.2, 16: 1.9}   # load cores ~0.5-0.6 us + mma chain ~0.7-1.3 us
TARGET_FINISH = 0.4


def target_parts(R):
    return {"launch": LAUNCH, "work": TARGET_WORK[R], "finish": TARGET_FINISH}


def physics_us(R):
    return pc.ring_flop(R) / pc.H100["peak"] * 1e6


def simple1_kernel_window():
    ns = L("нс", "ns")
    seg = [("launch", pc.INK2, L("запуск кернела", "kernel launch")),
           ("work", pc.C_R8, L("работа внутри кернела: загрузка ядер, вычисления, ожидание",
                               "work inside the kernel: loading cores, compute, waiting")),
           ("finish", pc.C_VIOLET, L("сборка результата из кусков",
                                     "assembling the result from pieces"))]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.4), sharex=True)
    print("simple1: ns per call")
    for ax, R in zip(axes, (8, 16)):
        rows = [(L("сейчас (замер)", "now (measured)"), now_parts(R)),
                (L("цель: доведённый кернел\n(оценка)", "target: tuned kernel\n(estimate)"),
                 target_parts(R)),
                (L("предел одного вызова", "limit of one call"),
                 {"launch": LAUNCH, "work": physics_us(R)}),
                (L("предел физики\n(много слоёв в одном кернеле)",
                   "physics limit\n(many layers in one kernel)"), {"work": physics_us(R)})]
        totals = []
        for i, (name, parts) in enumerate(rows):
            y = len(rows) - 1 - i
            left = 0.0
            for key, color, _ in seg:
                if key in parts:
                    w = parts[key] * NS
                    ax.barh(y, w, left=left, color=color, height=0.6, edgecolor=pc.SURFACE, lw=2)
                    if w > 600:
                        ax.text(left + w / 2, y, num(w), ha="center",
                                va="center", color=pc.SURFACE, fontsize=9.5, fontweight="bold")
                    left += w
            totals.append(left)
            ax.text(left + 120, y, f"{num(left)} {ns}", va="center",
                    fontsize=11, fontweight="bold", color=pc.INK)
            print(f"  R={R} {name.splitlines()[0]:28s} {left:8.0f}  "
                  + "  ".join(f"{k}={v * NS:.0f}" for k, v in parts.items()))
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.set_title(f"R = {R}", loc="left")
        gain = totals[0] - totals[1]
        ax.annotate("", xy=(totals[1], 2.5), xytext=(totals[0], 2.5),
                    arrowprops=dict(arrowstyle="<->", color=pc.INK, lw=1.3))
        ax.text(totals[0] + 150, 2.5,
                L("окно инженерии", "engineering window")
                + f" ≈ {num(gain)} {ns} (×{totals[0] / totals[1]:.1f})",
                ha="left", va="center", fontsize=10.5, color=pc.INK)
        print(f"  R={R} window {gain:.0f} ns (x{totals[0] / totals[1]:.2f})")
    axes[1].set_xlabel(L("время одного вызова оператора на H100, наносекунды (1 µs = 1000 нс)",
                         "time of one operator call on H100, nanoseconds (1 µs = 1000 ns)"))
    axes[1].set_xlim(0, 14000)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in seg]
    fig.legend(handles, [s[2] for s in seg], loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(L("Один вызов кернела при t = 1: сколько сейчас и сколько можно выжать",
                   "One kernel call at t = 1: where we are and how much is left"),
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.965, L("Как читать: длина полосы — время вызова, короче = лучше. "
                           "Цвет — на что уходит время. Стрелка — сколько снимет инженерия.",
                           "How to read: bar length is the call time, shorter is better. "
                           "Colour shows where the time goes. The arrow is what engineering "
                           "can remove."),
             ha="center", fontsize=10.5, color=pc.INK2)
    pc.save(fig, L("simple1_kernel_window.png", "simple1_kernel_window_en.png"))


# ---------------------------------------------------------------- the A100 node
NOW = {"eta": 0.10, "lat": 7.0e-6}      # our kernel as measured: ~11% peak at R16 t32, ~7 us/call
TARGET = {"eta": 0.30, "lat": 2.5e-6}   # the target of simple1
CALLS = 384                             # factorized linears per decode step (ASSUMPTION)


def step_ring(R, n, eta, lat):
    flop = pc.LINEAR * pc.ring_flop(R) / pc.DENSE_FLOP
    fixed = (pc.LM_HEAD + pc.LINEAR * pc.ring_bytes(R) / pc.DENSE_BYTES) / pc.BW_EFF
    return CALLS * lat + fixed + n * (flop / (eta * pc.A100["peak"]) + pc.TRAFFIC / pc.BW_EFF)


def ring_numbers(R, k):
    n = int(pc.sessions_ring(R))
    one = 1 / step_ring(R, 1, **k)
    total = n / step_ring(R, n, **k)
    return n, one, total


def simple2_node():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6))
    labels = ["сейчас:\ndense BF16", "кольцо\nR = 8", "кольцо\nR = 16"]
    colors = [pc.C_DENSE, pc.C_R8, pc.C_R16]
    r = {R: {name: ring_numbers(R, k) for name, k in (("now", NOW), ("target", TARGET))}
         for R in (8, 16)}
    print("\nsimple2: sessions, tok/s one session, tok/s total at max sessions")
    print(f"  dense measured: ~{pc.sessions_dense():.1f} sessions, one {pc.MEAS_PER[1]}, "
          f"total {pc.MEAS_AGG[4]} at 4")
    for R in (8, 16):
        for name in ("now", "target"):
            print(f"  R={R} {name:6s}: n={r[R][name][0]}  one={r[R][name][1]:.0f}  total={r[R][name][2]:.0f}")

    def bars(ax, dense, key, title, unit, fmt):
        x = np.arange(3)
        ax.bar(0, dense, color=colors[0], width=0.6)
        ax.text(0, dense, fmt(dense), ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        for i, R in enumerate((8, 16), start=1):
            now, tgt = r[R]["now"][key], r[R]["target"][key]
            if key == 0:
                tgt = pc.sessions_ring(R)
                ax.bar(i, tgt, color=colors[i], width=0.6)
                ax.text(i, tgt, fmt(tgt), ha="center", va="bottom", fontsize=10.5, fontweight="bold")
                continue
            ax.bar(i - 0.16, now, color=colors[i], alpha=0.35, width=0.3, hatch="//",
                   edgecolor=colors[i], lw=0)
            ax.bar(i + 0.16, tgt, color=colors[i], width=0.3)
            ax.text(i - 0.16, now, fmt(now), ha="center", va="bottom", fontsize=9, color=pc.INK2)
            ax.text(i + 0.16, tgt, fmt(tgt), ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.margins(y=0.15)

    bars(axes[0], pc.sessions_dense(), 0, "Сколько сессий по 55K\nпомещается на карту",
         "сессий", lambda v: f"{v:.1f}")
    bars(axes[1], pc.MEAS_PER[1], 1, "Скорость для одного\nпользователя", "токенов/с",
         lambda v: f"{v:.0f}")
    bars(axes[2], pc.MEAS_AGG[4], 2, "Сколько карта выдаёт\nвсем вместе", "токенов/с на карту",
         lambda v: f"{v:.0f}")
    axes[2].text(0, 12, "4 сессии", ha="center", color=pc.SURFACE, fontsize=9)
    for i, R in enumerate((8, 16), start=1):
        axes[2].text(i, 12, f"{r[R]['target'][0]} сессий", ha="center", color=pc.INK, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=pc.C_DENSE),
               plt.Rectangle((0, 0), 1, 1, color=pc.MUTED, alpha=0.35, hatch="//"),
               plt.Rectangle((0, 0), 1, 1, color=pc.MUTED)]
    fig.legend(handles, ["сейчас: замер на вашем узле (DFlash2, ~55K контекста)",
                         "кольцо с нашим кернелом как есть (оценка)",
                         "кольцо с доведённым кернелом (оценка)"],
               loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Одна A100 с Qwen3.8-27B, контекст 55K: сегодня и с кольцом",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.text(0.5, 0.965, "Как читать: три отдельных вопроса, у каждого своя шкала. "
                         "Синий — то, что есть сейчас. Светлый столбик — наш кернел сегодня, "
                         "яркий — после доводки.", ha="center", fontsize=10.5, color=pc.INK2)
    pc.save(fig, "simple2_node.png")


if __name__ == "__main__":
    simple1_kernel_window()
    if not EN:
        simple2_node()
