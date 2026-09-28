#!/usr/bin/env python3
"""Three talk figures on how the V3G ring kernel works and what it gave (2026-09-28, H100).

  kernel_flow[_en].png     block diagram of one kernel call (MLP 5120 -> 17408, R = 8, T = 1)
  kernel_memory[_en].png   where the data lives on the H100 and how it moves (steps 1-6)
  kernel_results[_en].png  step time, step time vs tokens, where one call's time goes, quality

Sizes are those of mlp_gate_up at R = 8, T = 1 with its tuned tiling (KC 4, QC 9, TT 1, NT 512):
128 blocks, shared memory 70 KiB per block (Nsight Compute: 71.68 KB), FP32 workspace 68 KiB.
Numbers on the results figure are read from results/h100/qwen/*.json.

    python tools/talk_kernel_figs.py          # Russian labels
    python tools/talk_kernel_figs.py --en     # English labels
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

EN = "--en" in sys.argv
ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "results" / "h100" / "qwen"
OUT = ROOT / "kernel-design" / "physics"
SUF = "_en" if EN else ""
DPI = 200


def L(ru, en):
    return en if EN else ru


# ramps: fill, edge, title, subtitle
TEAL = ("#E1F5EE", "#0F6E56", "#085041", "#0F6E56")
PURPLE = ("#EEEDFE", "#534AB7", "#3C3489", "#534AB7")
CORAL = ("#FAECE7", "#993C1D", "#712B13", "#993C1D")
GRAY = ("#F1EFE8", "#5F5E5A", "#2C2C2A", "#5F5E5A")
INK = "#5F5E5A"


class Canvas:
    """Pixel canvas 680 wide, y down, so the layout reads like an SVG."""

    def __init__(self, h):
        self.fig = plt.figure(figsize=(6.8, h / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 680)
        self.ax.set_ylim(h, 0)
        self.ax.axis("off")

    def box(self, x, y, w, h, c, title, sub=None, dashed=False, tsize=10, ssize=8.4):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4",
                                         fc=c[0], ec=c[1], lw=0.8, ls="--" if dashed else "-"))
        cx, cy = x + w / 2, y + h / 2
        if sub is None:
            self.ax.text(cx, cy, title, ha="center", va="center", fontsize=tsize, color=c[2])
        else:
            self.ax.text(cx, cy - 8, title, ha="center", va="center", fontsize=tsize, color=c[2],
                         fontweight="bold")
            self.ax.text(cx, cy + 9, sub, ha="center", va="center", fontsize=ssize, color=c[3])

    def frame(self, x, y, w, h):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                         fc="none", ec="#888780", lw=0.8, ls=(0, (5, 4))))

    def text(self, x, y, s, size=8.6, color=INK, ha="left", weight="normal"):
        self.ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, fontweight=weight)

    def line(self, pts, color=INK, lw=1.1):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round")

    def arrow(self, pts, color=INK, lw=1.1, both=False):
        if len(pts) > 2:
            self.line(pts[:-1], color, lw)
        self.ax.annotate("", xy=pts[-1], xytext=pts[-2],
                         arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=color, lw=lw,
                                         shrinkA=0, shrinkB=0, mutation_scale=9))

    def num(self, x, y, n, color):
        self.ax.add_patch(Circle((x, y), 9, fc=color, ec="none"))
        self.ax.text(x, y, str(n), ha="center", va="center", fontsize=8.4, color="white",
                     fontweight="bold")

    def save(self, name):
        self.fig.savefig(OUT / f"{name}{SUF}.png", dpi=DPI, facecolor="white")
        plt.close(self.fig)
        print("wrote", OUT / f"{name}{SUF}.png")


# ---- 1. block diagram -------------------------------------------------------------------------
c = Canvas(980)
c.box(60, 40, 250, 56, GRAY, L("Вход x", "Input x"), L("T токенов × 5120, BF16", "T tokens × 5120, BF16"))
c.box(370, 40, 250, 56, GRAY, L("Ядра кольца A, B, C", "Ring cores A, B, C"),
      L("88 тыс. параметров вместо 89 млн", "88 k parameters instead of 89 M"))
c.arrow([(185, 96), (270, 128)])
c.arrow([(495, 96), (410, 128)])
c.box(190, 130, 300, 56, GRAY, L("Выбор тайлинга", "Tiling choice"),
      L("по таблице замеров, один раз", "from the measured table, once"))
c.arrow([(340, 186), (340, 218)])
c.box(190, 220, 300, 56, GRAY, L("Сетка из 128 блоков", "Grid of 128 blocks"),
      L("звено a × кусок k × кусок q", "ring link a × k chunk × q chunk"))
c.frame(110, 312, 500, 356)
c.text(360, 330, L("Один блок = один SM, 512 потоков", "One block = one SM, 512 threads"))
c.arrow([(340, 276), (340, 346)])
steps = [
    (TEAL, L("1. Загрузка в общую память", "1. Load into shared memory"),
     L("cp.async: срезы A, B, C и x", "cp.async: slices of A, B, C and x")),
    (PURPLE, L("2. Этап 1: x · A", "2. Stage 1: x · A"),
     L("CUDA-ядра, FP32 → S1 в smem", "CUDA cores, FP32 → S1 in smem")),
    (PURPLE, L("3. Этап 2: S1 · B", "3. Stage 2: S1 · B"),
     L("тензорные ядра, mma.sync", "Tensor Cores, mma.sync")),
    (PURPLE, L("4. Этап 3: · C, копим Y", "4. Stage 3: · C, accumulate Y"),
     L("тензорные ядра, Y в регистрах", "Tensor Cores, Y stays in registers")),
    (TEAL, L("5. Сложить в FP32-буфер", "5. Add into the FP32 buffer"),
     L("атомики red.add в L2, 32 на выход", "red.add atomics in L2, 32 per output")),
]
for n, (col, t, s) in enumerate(steps):
    y = 348 + 64 * n
    c.box(170, y, 340, 48, col, t, s)
    if n < 4:
        c.arrow([(340, y + 48), (340, y + 62)])
c.arrow([(510, 564), (530, 564), (530, 500), (512, 500)])
c.text(538, 532, L("цикл по k", "loop over k"))
c.arrow([(340, 652), (340, 698)])
c.box(170, 700, 340, 48, GRAY, L("6. Как отдать результат?", "6. How to hand over the result?"),
      L("режим и счётчик куска", "mode and chunk counter"))
c.line([(340, 748), (340, 768)])
c.line([(135, 768), (545, 768)])
for x in (135, 340, 545):
    c.arrow([(x, 768), (x, 788)])
c.box(40, 790, 190, 48, TEAL, L("Режим 3: слияние", "Mode 3: fused"), L("без хвоста, FP32", "no tail, FP32 out"))
c.box(245, 790, 190, 48, GRAY, L("Не последний", "Not the last"), L("блок просто выходит", "the block just exits"))
c.box(450, 790, 190, 48, TEAL, L("Последний блок", "The last block"),
      L("FP32 → BF16, обнулить", "FP32 → BF16, clear"))
c.arrow([(135, 838), (135, 876)])
c.arrow([(545, 838), (545, 876)])
c.box(40, 878, 190, 48, GRAY, L("Следующее ядро", "Next kernel"),
      L("SiLU·up или +остаток", "SiLU·up or + residual"))
c.box(450, 878, 190, 48, GRAY, L("Выход y, BF16", "Output y, BF16"), L("запись в HBM", "written to HBM"))
for x, col, s in ((190, TEAL, L("движение данных", "data movement")), (370, PURPLE, L("вычисления", "compute"))):
    c.ax.add_patch(FancyBboxPatch((x, 950), 12, 12, boxstyle="round,pad=0,rounding_size=2",
                                  fc=col[0], ec=col[1], lw=0.8))
    c.text(x + 18, 956, s)
c.save("kernel_flow")

# ---- 2. memory map ----------------------------------------------------------------------------
c = Canvas(500)
c.box(40, 20, 600, 76, GRAY, "", None)
c.text(56, 42, L("HBM — видеопамять, 80 ГБ", "HBM — device memory, 80 GB"), 10, GRAY[2], weight="bold")
c.text(56, 66, L("3.35 ТБ/с: большая, но далёкая", "3.35 TB/s: large but far away"), 8.4, GRAY[3])
c.box(262, 56, 140, 30, TEAL, L("ядра слоя ~0.3 МБ", "layer cores ~0.3 MB"), tsize=8.4)
c.box(460, 56, 164, 30, TEAL, L("x 10 КБ · y 34 КБ", "x 10 KB · y 34 KB"), tsize=8.4)
c.box(40, 130, 600, 76, GRAY, "", None)
c.text(56, 152, L("L2-кэш, 50 МБ", "L2 cache, 50 MB"), 10, GRAY[2], weight="bold")
c.text(56, 176, L("общий для всех 132 SM", "shared by all 132 SMs"), 8.4, GRAY[3])
c.box(350, 156, 100, 30, CORAL, L("счётчики", "counters"), tsize=8.4)
c.box(460, 156, 164, 30, CORAL, L("FP32-буфер 68 КБ", "FP32 buffer 68 KB"), tsize=8.4)
c.frame(40, 240, 600, 234)
smem = [(56, 70, TEAL, "x 5 " + L("КБ", "KB")), (132, 70, TEAL, "A 5 " + L("КБ", "KB")),
        (208, 100, PURPLE, "S1 17 " + L("КБ", "KB")), (314, 130, TEAL, L("срез B 38 КБ", "B slice 38 KB")),
        (450, 70, TEAL, "C 5 " + L("КБ", "KB"))]
for x, w, col, t in smem:
    c.box(x, 252, w, 40, col, t, tsize=8.6)
c.box(526, 252, 98, 40, GRAY, L("свободно", "free"), "158 " + L("КБ", "KB"), dashed=True, tsize=8.4, ssize=8.4)
c.box(56, 350, 146, 50, PURPLE, L("CUDA-ядра", "CUDA cores"), L("этап 1, FP32", "stage 1, FP32"))
c.box(208, 350, 292, 50, PURPLE, L("Тензорные ядра", "Tensor Cores"), L("этапы 2 и 3: mma.sync", "stages 2 and 3: mma.sync"))
c.box(520, 350, 104, 50, CORAL, L("Регистры", "Registers"), L("acc и Y, FP32", "acc and Y, FP32"))
c.text(56, 440, L("SM — один из 132: блок 512 потоков, 64 регистра на поток",
                  "SM — one of 132: a block of 512 threads, 64 registers per thread"), 9.4, GRAY[2], weight="bold")
c.text(56, 460, L("общая память (smem) блока: занято 70 из 228 КБ", "block shared memory (smem): 70 of 228 KB used"),
       8.4, GRAY[3])
tl, pu, co = "#1D9E75", "#534AB7", "#D85A30"
c.line([(332, 86), (332, 224)], tl, 1.6)                     # 1: HBM -> L2 -> smem
c.line([(91, 224), (485, 224)], tl, 1.6)
for x in (91, 167, 379, 485):
    c.arrow([(x, 224), (x, 250)], tl, 1.6)
c.num(348, 113, 1, tl)
for x in (91, 167):                                          # 2: x, A -> CUDA cores -> S1
    c.arrow([(x, 292), (x, 348)], pu, 1.6)
c.arrow([(190, 350), (240, 294)], pu, 1.6)
c.num(129, 322, 2, pu)
for x in (258, 379):                                         # 3: S1, B -> Tensor Cores
    c.arrow([(x, 292), (x, 348)], pu, 1.6)
c.num(320, 322, 3, pu)
c.arrow([(485, 292), (485, 348)], pu, 1.6)                   # 4: C -> Tensor Cores <-> registers
c.arrow([(500, 375), (520, 375)], pu, 1.6, both=True)
c.num(505, 322, 4, pu)
c.arrow([(624, 375), (634, 375), (634, 232), (542, 232), (542, 188)], co, 1.6)  # 5: red.add
c.num(652, 300, 5, co)
c.arrow([(542, 156), (542, 88)], co, 1.6)                   # 6: tail FP32 -> BF16 y
c.num(560, 113, 6, co)
c.save("kernel_memory")

# ---- 3. results -------------------------------------------------------------------------------
chain = {(r["variant"], r["tokens"]): r for r in json.loads((Q / "model_chain_lowbit.json").read_text())["rows"]}
fused = {(r["variant"], r["tokens"]): r for r in json.loads((Q / "model_chain_fp32out.json").read_text())["rows"]}
ablate = {(r["shape"], r["rank"], r["tokens"]): r for r in json.loads((Q / "v3g_ablate.json").read_text())}
act = json.loads((Q / "activation.json").read_text())["rows"]
BLUE, BLUE2, GR = "#2a6fdb", "#8fb3ee", "#a8a7a0"

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
ax = axes[0, 0]
bars = [(L("dense BF16", "dense BF16"), chain[("dense", 1)]["graph_ms"], GR),
        ("dense FP8", chain[("fp8", 1)]["graph_ms"], GR),
        (L("INT4 (веса)", "INT4 (weights)"), chain[("int4", 1)]["graph_ms"], GR),
        (L("кольцо R=16", "ring R=16"), chain[("ring16", 1)]["graph_ms"], BLUE2),
        (L("кольцо R=16, слияние", "ring R=16, fused"), fused[("ring16f", 1)]["graph_ms"], BLUE2),
        (L("кольцо R=8", "ring R=8"), chain[("ring8", 1)]["graph_ms"], BLUE),
        (L("кольцо R=8, слияние", "ring R=8, fused"), fused[("ring8f", 1)]["graph_ms"], BLUE)]
for i, (name, v, col) in enumerate(bars):
    ax.barh(i, v, color=col, height=0.62)
    ax.text(v + 0.25, i, f"{v:.2f}", va="center", fontsize=9)
ax.set_yticks(range(len(bars)))
ax.set_yticklabels([b[0] for b in bars], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 23)
ax.set_xlabel(L("мс на шаг (меньше — лучше)", "ms per step (lower is better)"))
ax.set_title(L("Шаг модели, 1 токен", "Model step, 1 token"), fontsize=11)
ax.grid(alpha=0.3, axis="x")

ax = axes[0, 1]
Ts = [1, 2, 4, 8, 16, 32]
for v, name, col, ls in (("dense", "dense BF16", "#7f7f7f", (0, (8, 3))), ("fp8", "dense FP8", "#eb6834", (0, (5, 3))),
                         ("int4", "INT4", "#1baf7a", (0, (1.5, 2.5))), ("ring8", L("кольцо R=8", "ring R=8"), BLUE, "-")):
    ax.plot(Ts, [chain[(v, T)]["graph_ms"] for T in Ts], color=col, ls=ls, lw=2, marker="o", ms=4, label=name)
Tf = [1, 8, 32]
ax.plot(Tf, [fused[("ring8f", T)]["graph_ms"] for T in Tf], color=BLUE, ls="none", marker="D", ms=6,
        mfc="white", mew=1.6, label=L("кольцо R=8, слияние", "ring R=8, fused"))
ax.set_xscale("log", base=2)
ax.set_xticks(Ts)
ax.set_xticklabels([str(T) for T in Ts])
ax.set_ylim(0, 60)
ax.set_xlabel(L("токенов за шаг", "tokens per step"))
ax.set_ylabel(L("мс на шаг", "ms per step"))
ax.set_title(L("Где кольцо перестаёт выигрывать", "Where the ring stops winning"), fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, loc="upper left")

ax = axes[1, 0]
cases = [("mlp_gate_up", 8, 1), ("mlp_gate_up", 8, 8), ("mlp_gate_up", 16, 1), ("mlp_gate_up", 16, 8)]
parts = ((L("этапы 1–3", "stages 1–3"), BLUE), (L("атомики red.add", "red.add atomics"), "#eb6834"),
         (L("хвост FP32 → BF16", "FP32 → BF16 tail"), "#eda100"))
for i, key in enumerate(cases):
    r = ablate[key]
    seg = (r["compute_only_us"], r["full_us"] - r["no_red_us"], r["no_red_us"] - r["compute_only_us"])
    left = 0
    for (lab, col), s in zip(parts, seg):
        frac = s / r["full_us"]
        ax.barh(i, frac, left=left, color=col, height=0.6, label=lab if i == 0 else None,
                edgecolor="white", lw=1.5)
        ax.text(left + frac / 2, i, f"{frac * 100:.0f}%", ha="center", va="center", fontsize=8.5,
                color="white" if col == BLUE else "#2C2C2A")
        left += frac
    ax.text(1.01, i, f"{r['full_us']:.1f} µs", va="center", fontsize=8.5)
ax.set_yticks(range(len(cases)))
ax.set_yticklabels([f"R={k[1]}, {k[2]} " + L("ток.", "tok.") for k in cases], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1.13)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_title(L("Куда уходит время вызова (слой 5120 → 17408)", "Where one call's time goes (layer 5120 → 17408)"),
             fontsize=11)
ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)

ax = axes[1, 1]
meth = (("out_err_ring", L("кольцо", "ring"), "#b8cdf3"), ("out_err_ring_act", L("кольцо с учётом активаций", "ring, activation-aware"), BLUE),
        ("out_err_svd", L("SVD того же размера", "SVD, same size"), "#d3d1c7"),
        ("out_err_svd_act", L("SVD с учётом активаций", "SVD, activation-aware"), "#6f6e68"))
w = 0.2
for mi, (k, lab, col) in enumerate(meth):
    xs = [i + (mi - 1.5) * w for i in range(len(act))]
    ax.bar(xs, [r[k] for r in act], width=w, color=col, label=lab)
    for x, r in zip(xs, act):
        ax.text(x, r[k] + 0.01, f"{r[k]:.2f}", ha="center", va="bottom", fontsize=6.8, rotation=90)
ax.set_xticks(range(len(act)))
ax.set_xticklabels([f"R={r['R']}\n×{r['compression']:.0f}" for r in act], fontsize=9)
ax.set_ylim(0, 1.25)
ax.set_ylabel(L("ошибка выхода (меньше — лучше)", "output error (lower is better)"))
ax.set_title(L("Качество: слой 0, настоящий текст, один бюджет", "Quality: layer 0, real text, same budget"),
             fontsize=11)
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=8, loc="upper right", ncol=2)
fig.suptitle(L("Qwen3.8-27B на H100, BF16, замеры 28.09: скорость, узкие места, качество",
               "Qwen3.8-27B on H100, BF16, measured 28 Sep: speed, bottlenecks, quality"), fontsize=12)
fig.tight_layout()
fig.savefig(OUT / f"kernel_results{SUF}.png", dpi=150)
print("wrote", OUT / f"kernel_results{SUF}.png")
