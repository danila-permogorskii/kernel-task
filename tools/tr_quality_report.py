#!/usr/bin/env python3
"""QUALITY.md from tools/tr_quality.py (+ tools/tr_quality_visual.py) results: tables only, every
number read from the JSON. Figures: tools/tr_quality_charts.py.

    python tools/tr_quality_report.py results/h100/qwen/quality.json > results/h100/qwen/QUALITY.md
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def energy(err):
    return 100 * (1 - err * err)


def main():
    d = json.loads(Path(sys.argv[1]).read_text())
    rows = d["rows"]
    real = [r for r in rows if r["layer"] >= 0]
    bal = [r for r in real if r["tensorization"] == "balanced"]
    ranks = sorted({r["R"] for r in bal})
    layers = sorted({r["layer"] for r in real})
    p = print
    p("# Tensor ring on real Qwen3.8-27B weights: how much of each matrix survives\n")
    p(f"Source: `{sys.argv[1]}` ({d['device']}), {len(layers)} layers ({', '.join(map(str, layers))}), "
      f"TR-SVD init + up to {d['sweeps']} TR-ALS sweeps per fit, FP32. "
      "**err** = ||W − W_ring|| / ||W|| (relative Frobenius error; 0 = exact, 1 = nothing kept). "
      "**kept** = 1 − err² = share of the matrix's energy the approximation keeps. "
      "**SVD same size** = the best possible low-rank matrix with the same number of parameters "
      "as the ring (Eckart–Young, exact).\n")

    p("## Check of the method (controls, mlp_gate_up shape 5120→17408)\n")
    p("| matrix | R | ring err | SVD same size err |")
    p("|---|---|---|---|")
    for r in rows:
        if r["layer"] < 0:
            p(f"| {r['matrix']} | {r['R']} | {r['err_tr']:.4f} | {r['err_svd_eq']:.4f} |")
    p("\nA matrix that *is* a rank-8 ring is recovered (err ≈ 0 at R ≥ 8): the fitter works. "
      "A random Gaussian matrix (no structure) is the other extreme.\n")
    if "kernel_check" in d:
        k = d["kernel_check"]
        p("## Our kernel on fitted cores\n")
        p(f"Layer {k['layer']} {k['matrix']}, R = {k['R']}, {k['tokens']} tokens, BF16 kernel: "
          f"kernel vs the ring it was given **{k['kernel_vs_ring_rel_l2']:.1e}** (numerics are fine); "
          f"kernel vs the original layer **{k['kernel_vs_original_layer_rel_l2']:.4f}** "
          f"(= ring vs original {k['ring_vs_original_layer_rel_l2']:.4f}: the loss is the "
          "approximation, not the kernel).\n")

    p("## Q1 — does R = 8 / 16 keep a real matrix? (balanced tensorization, median over layers)\n")
    kinds = sorted({r["matrix"] for r in bal}, key=lambda m: ["mlp_gate", "mlp_up", "mlp_down",
                   "attn_q_gate", "attn_k", "attn_v", "o_proj", "gdn_qkvz", "gdn_out"].index(m))
    hdr = "| matrix | shape | " + " | ".join(f"R={R} err (kept) / SVD same size" for R in ranks) + \
          " | rank for 50% / 90% energy (of min side) |"
    p(hdr)
    p("|" + "---|" * (hdr.count("|") - 1))
    by = defaultdict(list)
    for r in bal:
        by[(r["matrix"], r["R"])].append(r)
    for m in kinds:
        cells = []
        for R in ranks:
            g = by[(m, R)]
            e = statistics.median(x["err_tr"] for x in g)
            s = statistics.median(x["err_svd_eq"] for x in g)
            cells.append(f"{e:.4f} ({energy(e):.1f}%) / {s:.4f} ({energy(s):.1f}%)")
        spec = [x for x in by[(m, ranks[0])] if "rank_for_50pct_energy" in x]
        r50 = statistics.median(x["rank_for_50pct_energy"] for x in spec)
        r90 = statistics.median(x["rank_for_90pct_energy"] for x in spec)
        g0 = by[(m, ranks[0])][0]
        n_min = min(prod(g0["ins"]), prod(g0["outs"]))
        p(f"| {m} | {prod(g0['ins'])}→{prod(g0['outs'])} | " + " | ".join(cells)
          + f" | {r50:.0f} / {r90:.0f} (of {n_min}) |")
    p("\nCompression (dense params / ring params) at these ranks: "
      + ", ".join(f"R={R}: ×{statistics.median(x['compression'] for x in bal if x['R'] == R):.0f}"
                  for R in ranks) + " (median over matrices).\n")

    p("### Spread over depth (mlp_up, R = 8 and 16)\n")
    p("| layer | " + " | ".join(f"R={R} err" for R in ranks) + " |")
    p("|" + "---|" * (len(ranks) + 1))
    for L in layers:
        cells = []
        for R in ranks:
            g = [x for x in bal if x["layer"] == L and x["matrix"] == "mlp_up" and x["R"] == R]
            cells.append(f"{g[0]['err_tr']:.4f}" if g else "–")
        p(f"| {L} | " + " | ".join(cells) + " |")

    q2 = [r for r in real if r["layer"] in {x["layer"] for x in real if x["tensorization"] != "balanced"}]
    if any(r["tensorization"] != "balanced" for r in q2):
        p("\n## Q2 — does the tensorization matter? (layers with all variants, R = 8 / 16)\n")
        p("Variants: **balanced** (our modes), **out reversed** (same digits, other grouping of the "
          "output index), **skewed** (FLOP-optimal digits, e.g. 5120 = 128·10·4), **random perm** "
          "(balanced after shuffling rows and columns: destroys any structure of the index order).\n")
        variants = ["balanced", "balanced_out_reversed", "skewed", "balanced_random_perm"]
        p("| layer | matrix | R | " + " | ".join(variants) + " | SVD same size (balanced) |")
        p("|" + "---|" * (len(variants) + 4))
        for L in sorted({r["layer"] for r in q2}):
            for m in kinds:
                for R in (8, 16):
                    g = {r["tensorization"]: r for r in q2 if r["layer"] == L and r["matrix"] == m
                         and r["R"] == R}
                    if len(g) < 2:
                        continue
                    cells = [f"{g[v]['err_tr']:.4f}" + (f" {tuple(g[v]['ins'])}→{tuple(g[v]['outs'])}"
                             if v == "skewed" else "") if v in g else "–" for v in variants]
                    p(f"| {L} | {m} | {R} | " + " | ".join(cells)
                      + f" | {g['balanced']['err_svd_eq']:.4f} |")


def prod(t):
    out = 1
    for v in t:
        out *= v
    return out


if __name__ == "__main__":
    main()
