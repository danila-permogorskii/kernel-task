#!/usr/bin/env python3
"""Where kernel engineering ends and physics begins, for the tensor-ring operator.

Five charts, written to kernel-design/physics/:
  fig1_roofline.png     measured kernels against the H100 roofline
  fig2_floors.png       kernel time vs tokens: measured, physical floors, launch floor
  fig3_break_even.png   break-even token count t* vs rank, H100 and A100
  fig4_memory.png       A100 80GB memory of the Qwen3.8-27B node: dense vs ring
  fig5_serving.png      decode throughput per A100 vs resident sessions (a model)

Figures 1-3 use only measured H100 data (results/h100/kernels.json) and exact FLOP
counts. Figures 4-5 are an estimate: the inference-ops Qwen3.8-27B node's measured
numbers plus the ASSUMPTIONs marked below. The script prints every derived number.

    python tools/physics_charts.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "kernel-design" / "physics"
K = json.loads((ROOT / "results" / "h100" / "kernels.json").read_text())

# Palette: the dataviz reference palette, light mode.
C_DENSE, C_R8, C_R16 = "#2a78d6", "#eb6834", "#1baf7a"
C_GREEN, C_VIOLET = "#008300", "#4a3aa7"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8d8c87", "#e4e3df"
SURFACE, BAND, BAND_WARN, LIGHT = "#fcfcfb", "#ecebe7", "#f6e3da", "#d6d5cf"
RANK_COLOR = {8: C_R8, 16: C_R16}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "legend.frameon": False,
})

# ------------------------------------------------------------------ hardware
H100 = {"name": "H100 SXM5", "peak": K["peaks"]["fp16_tc_tflops"] * 1e12,
        "bw": K["peaks"]["hbm_gbs"] * 1e9}
A100 = {"name": "A100 80GB PCIe", "peak": 312e12, "bw": 1.935e12}   # datasheet
KERNEL_EMPTY_US = K["floors"]["kernel_empty_us"]

# ------------------------------------------------------------------ the operator
N_IN, N_OUT = 1920, 2880
DENSE_FLOP = 2 * N_IN * N_OUT        # per token
DENSE_BYTES = 2 * N_IN * N_OUT       # FP16 weight
IO_BYTES = 2 * (N_IN + N_OUT)        # x and y, per token


def case(rank, tokens, design):
    return next(c for c in K["cases"]
                if c["rank"] == rank and c["tokens"] == tokens and c["design"] == design)


_r8 = case(8, 1, "B")
STAGE13 = _r8["flops"]["stage1"] + _r8["flops"]["stage3"]   # grows as R^2
STAGE2 = _r8["flops"]["stage2"]                              # grows as R^3
CORE8 = _r8["min_bytes"] - IO_BYTES                          # packed cores at R = 8


def ring_flop(R):
    return STAGE13 * (R / 8) ** 2 + STAGE2 * (R / 8) ** 3


def ring_bytes(R):
    return CORE8 * (R / 8) ** 2


assert case(8, 1, "dense")["flops"]["total"] == DENSE_FLOP
assert abs(ring_flop(16) - case(16, 1, "B")["flops"]["total"]) < 1
assert abs(ring_bytes(16) - (case(16, 1, "B")["min_bytes"] - IO_BYTES)) < 1


def floor_dense(t, hw):
    return np.maximum(DENSE_FLOP * t / hw["peak"], (DENSE_BYTES + IO_BYTES * t) / hw["bw"])


def floor_ring(R, t, hw, eta=1.0):
    return np.maximum(ring_flop(R) * t / (eta * hw["peak"]),
                      (ring_bytes(R) + IO_BYTES * t) / hw["bw"])


def t_star(R, hw, eta=1.0):
    """Tokens per call at which a ring kernel running at eta x Tensor Core peak costs as
    much as dense reading its weight at full HBM bandwidth (ring core bytes neglected)."""
    return eta * hw["peak"] * DENSE_BYTES / (hw["bw"] * ring_flop(R))


def measured_us(rank, tokens, design):
    return case(rank, tokens, design)["kernel_us_per_call"]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT.relative_to(ROOT) / name)


def fig1_roofline():
    hw = H100
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    I = np.logspace(-0.5, 4.6, 400)
    ridge = hw["peak"] / hw["bw"]
    ax.axvspan(I[0], ridge, color=BAND, zorder=0, lw=0)
    ax.plot(I, np.minimum(hw["peak"], hw["bw"] * I) / 1e12, color=INK, lw=2.2, zorder=3)
    ax.axvline(ridge, color=MUTED, ls=":", lw=1.2)
    ax.text(ridge / 1.1, 1.3, f"точка баланса\n≈ {ridge:.0f} FLOP/байт", color=INK2, fontsize=10, ha="right")
    ax.text(0.4, 2600, "ждём память:\nвремя = байты ÷ ПСП", color=INK2, fontsize=10, va="top")
    ax.text(1.6e3, 2600, "ждём арифметику:\nвремя = FLOP ÷ пик", color=INK2, fontsize=10, va="top")

    series = [
        ("dense (cuBLAS)", C_DENSE, "o", 8, "dense", (1, 8, 32),
         lambda t: (DENSE_FLOP * t, DENSE_BYTES + IO_BYTES * t)),
        ("кольцо R = 8", C_R8, "s", 8, "B", (1, 8, 32),
         lambda t: (ring_flop(8) * t, ring_bytes(8) + IO_BYTES * t)),
        ("кольцо R = 16", C_R16, "D", 16, "B", (1, 32),
         lambda t: (ring_flop(16) * t, ring_bytes(16) + IO_BYTES * t)),
    ]
    print("\nfig1: intensity FLOP/B, achieved TFLOP/s, roof TFLOP/s, % of roof")
    for label, color, marker, rank, design, ts, fb in series:
        xs, ys = [], []
        for t in ts:
            f, b = fb(t)
            x = f / b
            got = f / (measured_us(rank, t, design) * 1e-6) / 1e12
            roof = min(hw["peak"], hw["bw"] * x) / 1e12
            print(f"  {label:16s} t={t:<3d} {x:8.1f} {got:8.2f} {roof:8.1f} {100 * got / roof:5.1f}%")
            ax.plot([x, x], [got, roof], color=color, lw=1.2, alpha=0.6, zorder=2)
            ax.plot(x, roof, marker=marker, ms=9, mfc=SURFACE, mec=color, mew=2, ls="none", zorder=4)
            ax.annotate(f"t={t}", (x, got), textcoords="offset points", xytext=(9, -4),
                        fontsize=9, color=INK2)
            xs.append(x)
            ys.append(got)
        ax.plot(xs, ys, marker=marker, ms=9, color=color, ls="none", mec=SURFACE, mew=1.5,
                label=label, zorder=5)

    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="s", color=INK2, ls="none", ms=8, label="закрашено: измерено"),
                Line2D([], [], marker="s", mfc=SURFACE, mec=INK2, mew=2, ls="none", ms=8,
                       label="пусто: потолок для этой точки")]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.text(0.4, 300, "Отрезок от точки до крыши — запас,\nкоторый может выбрать инженерия.\n"
                     "Крыша — физика: выше не бывает.", fontsize=10, color=INK)
    ax.set(xscale="log", yscale="log", xlim=(0.3, 4e4), ylim=(1, 3000),
           xlabel="арифметическая интенсивность, FLOP на байт из памяти (лог)",
           ylabel="TFLOP/s (лог)", title="H100: где наши кернелы относительно потолка")
    fig.text(0.01, -0.02, "Время кернела из results/h100/kernels.json. Dense измерен с весом в L2 "
                          "(повторные вызовы); в модели его вес идёт из HBM.", fontsize=8.5, color=INK2)
    save(fig, "fig1_roofline.png")


def fig2_floors():
    hw = H100
    t = np.logspace(0, np.log10(512), 400)
    ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    print("\nfig2: t* (H100, 100% TC), headroom at t=1")
    for ax, R, ts in ((axes[0], 8, (1, 8, 32)), (axes[1], 16, (1, 32))):
        color = RANK_COLOR[R]
        fr = floor_ring(R, t, hw) * 1e6
        reach = np.maximum(fr, KERNEL_EMPTY_US)
        ts_ = t_star(R, hw)
        ax.fill_between(t, 1e-3, reach, color=BAND, lw=0, zorder=0)
        ax.axvspan(ts_, t[-1], color=BAND_WARN, lw=0, zorder=0)
        ax.plot(t, floor_dense(t, hw) * 1e6, color=C_DENSE, ls="--", label="dense: пол (вес из HBM)")
        ax.plot(t, fr, color=color, ls="--", label=f"кольцо R = {R}: пол (100% TC)")
        ax.axhline(KERNEL_EMPTY_US, color=MUTED, ls=":", lw=1.2)
        ax.text(40, KERNEL_EMPTY_US * 0.72, f"пустой кернел {KERNEL_EMPTY_US:.2f} µs",
                color=INK2, fontsize=9)
        dm = [measured_us(R, x, "dense") for x in ts]
        rm = [measured_us(R, x, "B") for x in ts]
        ax.plot(ts, dm, "o", color=C_DENSE, ms=8, mec=SURFACE, mew=1.5, label="dense измерено (вес в L2)")
        ax.plot(ts, rm, "s-", color=color, ms=8, lw=1.4, mec=SURFACE, mew=1.5,
                label=f"наш кернел R = {R}, измерено")
        r1 = rm[0]
        f1 = max(float(floor_ring(R, 1, hw)) * 1e6, KERNEL_EMPTY_US)
        ax.annotate("", xy=(1.22, f1), xytext=(1.22, r1),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
        ax.text(1.32, 1.15, f"×{r1 / f1:.1f} запас\nинженерии", color=INK, fontsize=9, va="bottom")
        rl = rm[-1]
        fl = float(floor_ring(R, ts[-1], hw)) * 1e6
        print(f"  R={R}: t*={ts_:.1f}  t=1: {r1:.2f} us vs reachable {f1:.2f} (x{r1 / f1:.1f});"
              f"  t={ts[-1]}: {rl:.1f} us vs floor {fl:.2f} (x{rl / fl:.1f})")
        ax.axvline(ts_, color=INK2, lw=1, ls="-.")
        ax.text(ts_ * 1.08, 0.05, f"t* ≈ {ts_:.0f}:\nправее dense\nбыстрее любого\nкернела кольца",
                color=INK, fontsize=9, va="bottom")
        ax.text(1.05, 0.013, "недостижимо: физика + пол запуска", color=INK2, fontsize=9)
        ax.set(xscale="log", yscale="log", xlim=(1, 512), ylim=(0.01, 400),
               title=f"R = {R}", xlabel="токенов за вызов, t (лог)")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(x) for x in ticks])
    axes[0].set_ylabel("время кернела, µs (лог)")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + [h1[1], h1[3]], l0 + [l1[1], l1[3]], loc="lower center",
               bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9.5)
    fig.suptitle("Где кончается инженерия и начинается физика (H100, один вызов оператора)",
                 fontsize=14, fontweight="bold")
    fig.text(0.01, -0.14, "Пол = max(FLOP ÷ пик, байты ÷ ПСП). Серая зона недостижима никаким "
                          "кернелом. Правее t* кольцо проигрывает dense даже на 100% Tensor Core.",
             fontsize=9, color=INK2)
    save(fig, "fig2_floors.png")


def fig3_break_even():
    R = np.linspace(6, 32, 300)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.fill_between(R, t_star(R, H100), 2e4, color=BAND_WARN, lw=0, zorder=0)
    for hw, color in ((H100, INK), (A100, C_GREEN)):
        for eta, ls in ((1.0, "-"), (0.3, "--")):
            ax.plot(R, t_star(R, hw, eta), color=color, ls=ls,
                    label=f"{hw['name']}, кернел на {eta:.0%} пика TC")
    refs = [(1, "1 сессия, обычный decode"), (8, "1 сессия + DFlash2 (блок 8)"),
            (32, "4 сессии × блок 8: ваш прод"), (4096, "чанк prefill 4096")]
    for y, txt in refs:
        ax.axhline(y, color=MUTED, lw=1, ls=":")
        ax.text(32.4, y, txt, va="center", fontsize=9, color=INK2, clip_on=False)
    print("\nfig3: FLOP overhead and t* per rank")
    for Rm in (8, 16):
        over = ring_flop(Rm) / DENSE_FLOP
        vals = {f"{hw['name']} {eta:.0%}": t_star(Rm, hw, eta) for hw in (H100, A100) for eta in (1.0, 0.3)}
        print(f"  R={Rm}: FLOP x{over:.2f}, memory /{DENSE_BYTES / ring_bytes(Rm):.1f}, "
              + ", ".join(f"{k}: {v:.1f}" for k, v in vals.items()))
        ax.axvline(Rm, color=MUTED, lw=1)
        ax.text(Rm + 0.3, 3500, f"R = {Rm}\nFLOP ×{over:.1f}\nt* H100 ≈ {t_star(Rm, H100):.0f}\n"
                                f"t* A100 ≈ {t_star(Rm, A100):.0f}", fontsize=9, color=INK, va="top")
    ax.text(20.5, 250, "t* = η · (пик ÷ ПСП) ÷ (FLOP кольца ÷ FLOP dense)\n"
                     "     = КПД · точка баланса ÷ накладные FLOP", fontsize=10.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=MUTED))
    ax.text(24, 1200, "выше линий: dense быстрее\nпо физике", fontsize=10, color=INK2)
    ax.set(yscale="log", xlim=(6, 32), ylim=(0.5, 1.5e4), xlabel="ранг кольца R",
           ylabel="t*: до скольких токенов за вызов\nкольцо может обогнать dense (лог)",
           title="Точка безубыточности: сколько токенов за вызов выдерживает кольцо")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9.5)
    fig.text(0.01, -0.13, "FLOP кольца: стадия 2 растёт как R³, стадии 1 и 3 как R² (сверено с "
                          "замерами R = 8 и 16). Dense читает вес из HBM на полной ПСП.",
             fontsize=8.5, color=INK2)
    save(fig, "fig3_break_even.png")


# ------------------------------------------------------------------ serving (A100 node)
GiB = 2 ** 30
# Measured on the inference-ops node (qwen3.6-27b-2-A100: Qwen3.8-27B BF16, SGLang, TP=1).
W_TARGET = 51.05 * GiB          # boot log: "Load weight end ... mem usage=51.05 GB"
BUDGET = 0.94 * 79.27 * GiB     # --mem-fraction-static 0.94 of 79.27 GiB
POOL_TOKENS = 182_528           # KV pool, boot log
GDN_SLOTS = 43                  # --max-mamba-cache-size
GDN_SLOT = 27.1e9 / 344         # host tier holds 27.1 GB for 344 GDN states
BW_EFF = 1.086e12               # 54 GB x 20.12 forwards/s at c=4: 56% of HBM peak
W_STEP_SPEC = 57e9              # weights read per DFlash2 step, target + drafter
ACCEPT = 3.2                    # DFlash2 accept length at ~55K context
MEAS_AGG = {1: 61.1, 4: 184.9}  # aggregate tok/s at ~55K context, applied config
MEAS_PER = {1: 65.2, 4: 55.2}   # median per-request tok/s, same runs
CTX = 55_000
VOCAB, HIDDEN = 248_044, 5_120
EMBED = LM_HEAD = VOCAB * HIDDEN * 2
KV_TOKEN = 16 * 4 * 256 * 2 * 2  # 16 full-attention layers x 4 KV heads x 256 x (K, V) x BF16
# ASSUMPTIONs.
OTHER_W = 1.3 * GiB             # vision tower, norms, conv: not factorized
LINEAR = W_TARGET - EMBED - LM_HEAD - OTHER_W   # the weights that become rings
ETAS = (0.3, 0.6)               # ring kernels at 30-60% of A100 Tensor Core peak
LAT = 384 * 5e-6                # ~6 factorized linears x 64 layers, ~5 us latency floor each
SLOTS_PER_SESSION = CTX / 4096  # one GDN checkpoint per 4096-token prefill chunk, as now
# The model's matrices are assumed to carry the test operator's overheads:
# FLOP x ring_flop(R) / DENSE_FLOP, memory / (DENSE_BYTES / ring_bytes(R)).

W_READ_DENSE = W_TARGET - EMBED - OTHER_W
KV_POOL = POOL_TOKENS * KV_TOKEN
OTHER_MEM = BUDGET - W_TARGET - KV_POOL - GDN_SLOTS * GDN_SLOT  # drafter, draft KV, graphs
RESIDENT = CTX * KV_TOKEN + SLOTS_PER_SESSION * GDN_SLOT          # memory per session
TRAFFIC = CTX * KV_TOKEN + 2 * GDN_SLOT                           # bytes per session per step


def ring_weights(R):
    return W_TARGET - LINEAR + LINEAR * ring_bytes(R) / DENSE_BYTES


def sessions_dense():
    return (KV_POOL + GDN_SLOTS * GDN_SLOT) / RESIDENT


def sessions_ring(R):
    return (BUDGET - ring_weights(R) - OTHER_MEM) / RESIDENT


def step_dense(n):
    """One DFlash2 step for n sessions: bandwidth-bound, returns seconds."""
    return (W_STEP_SPEC + n * TRAFFIC) / BW_EFF


def step_ring(R, n, eta):
    """One plain decode step for n sessions with ring linears, returns seconds."""
    flop = LINEAR * ring_flop(R) / DENSE_FLOP        # 2 FLOP per 2-byte weight
    fixed = (LM_HEAD + LINEAR * ring_bytes(R) / DENSE_BYTES) / BW_EFF
    return LAT + fixed + n * (flop / (eta * A100["peak"]) + TRAFFIC / BW_EFF)


def fig4_memory():
    rows = [("dense BF16\n(сейчас)", W_TARGET, KV_POOL + GDN_SLOTS * GDN_SLOT, sessions_dense())]
    for R in (8, 16):
        w = ring_weights(R)
        rows.append((f"кольцо R = {R}", w, BUDGET - w - OTHER_MEM, sessions_ring(R)))
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    print(f"\nfig4: budget {BUDGET / GiB:.1f} GiB, linear {LINEAR / GiB:.1f} GiB, other {OTHER_MEM / GiB:.1f} GiB,"
          f" per session {RESIDENT / 1e9:.2f} GB at {CTX} tokens")
    for i, (name, w, pool, n) in enumerate(rows):
        y = len(rows) - 1 - i
        parts = [(w, INK2, "веса"), (pool, C_VIOLET, "KV + состояния GDN"),
                 (OTHER_MEM, LIGHT, "драфтер, графы, резерв")]
        left = 0.0
        for size, color, lab in parts:
            ax.barh(y, size / GiB, left=left / GiB, color=color, height=0.55,
                    edgecolor=SURFACE, lw=2, label=lab if i == 0 else None)
            if size / GiB > 4:
                ax.text((left + size / 2) / GiB, y, f"{size / GiB:.1f}", ha="center", va="center",
                        color=SURFACE if color != LIGHT else INK, fontsize=10, fontweight="bold")
            left += size
        ax.text(left / GiB + 1, y, f"≈ {n:.1f} сессий\nпо {CTX // 1000}K", va="center", fontsize=10, color=INK)
        print(f"  {name.splitlines()[0]:14s} weights {w / GiB:5.1f} GiB  pool {pool / GiB:5.1f} GiB  sessions {n:.1f}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set(xlim=(0, 92), xlabel="GiB на одной A100 80GB (бюджет 0.94 × 79.3)",
           title="Qwen3.8-27B на A100: куда уходит память")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3, fontsize=9)
    fig.text(0.01, -0.2, "Оценка. Сжимаются все линейные слои (≈ 45 GiB), эмбеддинги и lm_head "
                         "остаются dense. Сжатие памяти как у тестового оператора: ÷124 (R = 8), ÷31 (R = 16).",
             fontsize=8.5, color=INK2)
    save(fig, "fig4_memory.png")


def fig5_serving():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    nd = np.arange(1, 5)
    agg_d = nd * ACCEPT / step_dense(nd)
    ax1.plot(nd, agg_d, "o-", color=C_DENSE, ms=7, label="dense + DFlash2 (модель)")
    ax2.plot(nd, ACCEPT / step_dense(nd), "o-", color=C_DENSE, ms=7, label="dense + DFlash2 (модель)")
    for ax, meas in ((ax1, MEAS_AGG), (ax2, MEAS_PER)):
        ax.plot(list(meas), list(meas.values()), "x", color=INK, ms=10, mew=2.5,
                label="замер на вашем узле, ~55K")
    print("\nfig5: tok/s   aggregate / per-session")
    for n in (1, 4):
        print(f"  dense+DFlash2 n={n}: {n * ACCEPT / step_dense(n):6.1f} / {ACCEPT / step_dense(n):5.1f}"
              f"   (measured {MEAS_AGG[n]} / {MEAS_PER[n]})")
    for R in (8, 16):
        color = RANK_COLOR[R]
        nmax = int(sessions_ring(R))
        n = np.arange(1, nmax + 1)
        lo, hi = (n / step_ring(R, n, eta) for eta in ETAS)
        ax1.fill_between(n, lo, hi, color=color, alpha=0.25, lw=0)
        ax1.plot(n, lo, color=color, lw=1.6, label=f"кольцо R = {R}, 30–60% пика TC")
        ax1.plot(n, hi, color=color, lw=1.6)
        ax1.plot(n[-1], hi[-1], "|", color=color, ms=16, mew=2.5)
        ax2.fill_between(n, lo / n, hi / n, color=color, alpha=0.25, lw=0)
        ax2.plot(n, lo / n, color=color, lw=1.6, label=f"кольцо R = {R}, 30–60% пика TC")
        ax2.plot(n, hi / n, color=color, lw=1.6)
        for k in (1, 4, nmax):
            print(f"  ring R={R} n={k:2d}: {lo[k - 1]:6.1f}-{hi[k - 1]:6.1f} / "
                  f"{lo[k - 1] / k:5.1f}-{hi[k - 1] / k:5.1f}")
    ax1.axvline(sessions_dense(), color=C_DENSE, ls=":", lw=1.2)
    ax1.text(sessions_dense() + 0.15, 20, "dense: память\nкончилась", fontsize=9, color=INK2)
    ax1.text(sessions_ring(8) - 0.2, 12, "кольцо: память\nкончилась ▸", fontsize=9, color=INK2, ha="right")
    ax1.set(xlabel="сессий по 55K одновременно на GPU", ylabel="токенов/с на GPU, сумма",
            title="Пропускная способность одной A100", xlim=(0.5, 14.5), ylim=(0, 260))
    ax2.set(xlabel="сессий по 55K одновременно на GPU", ylabel="токенов/с на одну сессию",
            title="Скорость, которую видит пользователь", xlim=(0.5, 14.5), ylim=(0, 130))
    ax1.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    fig.suptitle("Qwen3.8-27B на A100, decode при контексте 55K: модель, откалиброванная по вашему узлу",
                 fontsize=14, fontweight="bold")
    fig.text(0.01, -0.04, "Оценка, не замер. Dense: байты за шаг ÷ 1.09 ТБ/с (измеренная ПСП узла), "
                          "принято 3.2 токена за шаг. Кольцо: без спекуляции, накладные FLOP как у тестового "
                          "оператора (×3.6 / ×25),\nпол задержки 5 µs × 384 вызова на шаг. Prefill не показан: "
                          "там кольцо платит те же накладные FLOP на каждый токен промпта.",
             fontsize=8.5, color=INK2)
    save(fig, "fig5_serving.png")


if __name__ == "__main__":
    fig1_roofline()
    fig2_floors()
    fig3_break_even()
    fig4_memory()
    fig5_serving()
