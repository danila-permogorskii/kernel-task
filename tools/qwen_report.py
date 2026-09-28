#!/usr/bin/env python3
"""Markdown summary of tools/qwen_bench.py results.

    python tools/qwen_report.py results/a100/qwen/bf16.json \\
        [--tuned results/a100/qwen/bf16_tuned.json] > results/a100/qwen/summary.md

Per (shape, R, T): CUDA-event stream µs of dense, the torch reference and our kernel (chooser
tiling, and the sweep's tiling if --tuned is given), our speed relative to dense, dense's
achieved bandwidth, our useful FLOP rate, relative L2 error, and weight bytes. Then one decode
step (T = 1) of all linear layers of the model: sum over shapes of (layers x stream time).
Isolated repeated calls: the ring cores stay in L2 between calls, dense weights do not fit;
see the caveats printed under the tables.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# datasheet peaks (dense, no sparsity): HBM GB/s, BF16/FP16 Tensor Core TFLOP/s
PEAKS = {"A100-SXM4-80GB": (2039, 312), "A100-SXM4-40GB": (1555, 312),
         "A100 80GB PCIe": (1935, 312), "A100-PCIE-40GB": (1555, 312),
         "H100 80GB HBM3": (3350, 989), "H100 PCIe": (2000, 756)}


def load(path):
    data = json.loads(Path(path).read_text())
    cases = {}
    for c in data["cases"]:
        if "failed" in c:
            continue
        rank = None if c["method"] == "dense" else c["rank"]
        cases[(c["shape"], rank, c["tokens"], c["method"])] = c
    return data, cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--tuned")
    a = ap.parse_args()
    data, cases = load(a.bench)
    tuned = load(a.tuned)[1] if a.tuned else {}
    any_case = next(iter(cases.values()))
    dev = any_case["environment"]["device_name"]
    peak = next((v for k, v in PEAKS.items() if k in dev), None)
    shapes = data["shapes"]
    ranks = sorted({k[1] for k in cases if k[1] is not None})
    tokens = sorted({k[2] for k in cases})

    print(f"# Tensor-ring kernel on Qwen3.8-27B layer shapes — {dev}, "
          f"{any_case['environment']['dtype']}\n")
    print(f"Source: `{a.bench}`" + (f", tuned tilings `{a.tuned}`" if a.tuned else "")
          + f". Stream = CUDA-event µs per call (harness protocol). "
          + (f"Peaks used: {peak[0]} GB/s HBM, {peak[1]} TFLOP/s BF16 Tensor Core.\n"
             if peak else "No datasheet peaks known for this device.\n"))
    hdr = ("| shape | R | T | dense µs | ref µs | ours µs | tiling | "
           + ("tuned µs | tuned tiling | " if tuned else "")
           + "dense / ours | dense GB/s | ours TFLOP/s | relL2 ours | relL2 dense |")
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    step = defaultdict(float)  # (who, R) -> µs of one decode step, all linear layers
    for s, meta in shapes.items():
        for R in ranks:
            for T in tokens:
                d = cases.get((s, None, T, "dense"))
                r = cases.get((s, R, T, "factorized_reference"))
                o = cases.get((s, R, T, "factorized_optimized"))
                t = tuned.get((s, R, T, "factorized_optimized"))
                if not (d and o):
                    continue
                us = lambda c: c["cuda_event_stream_median_ms"] * 1e3 if c else float("nan")  # noqa: E731
                best = min(us(o), us(t)) if t else us(o)
                gbs = d["memory"]["representation_logical_bytes"] / (us(d) * 1e-6) / 1e9
                tfl = o["ring_flop"] / (best * 1e-6) / 1e12
                k, kt = o["kernel"] or {}, (t or {}).get("kernel") or {}
                fb = " FALLBACK" if k.get("fell_back_to_reference") else ""
                ok = "" if o["correctness"]["passed"] else " FAIL"
                row = (f"| {s} | {R} | {T} | {us(d):.1f} | {us(r):.1f} | {us(o):.1f}{fb}{ok} | "
                       f"{tuple(k.get('tiling_kc_tt_qc') or ())} | ")
                if tuned:
                    row += (f"{us(t):.1f} | {tuple(kt.get('tiling_kc_tt_qc') or ())} | " if t
                            else "– | – | ")
                pct = f" ({100 * gbs / peak[0]:.0f}%)" if peak else ""
                row += (f"{us(d) / best:.2f}x | {gbs:.0f}{pct} | {tfl:.1f} | "
                        f"{o['correctness']['relative_l2_error']:.1e} | "
                        f"{d['correctness']['relative_l2_error']:.1e} |")
                print(row)
                if T == 1:
                    n = meta["layers_per_forward"]
                    step[("dense", R)] += n * us(d)
                    step[("ours", R)] += n * best
    print("\n## Memory per layer (R-dependent rows are ours)\n")
    print("| shape | dense weight MB | R | cores KB | packed operands KB | workspace+output KB (T=1) |")
    print("|---|---|---|---|---|---|")
    for s in shapes:
        d = cases.get((s, None, tokens[0], "dense"))
        for R in ranks:
            o = cases.get((s, R, tokens[0], "factorized_optimized"))
            if not (d and o):
                continue
            print(f"| {s} | {d['memory']['representation_logical_bytes'] / 2**20:.1f} | {R} | "
                  f"{o['memory']['representation_logical_bytes'] / 2**10:.0f} | "
                  f"{(o['kernel'] or {}).get('packed_operand_bytes', 0) / 2**10:.0f} | "
                  f"{o['memory']['incremental_workspace_and_output_bytes'] / 2**10:.0f} |")
    if step:
        print("\n## One decode step (T = 1), all factorised linear layers of the model\n")
        print("Sum over shapes of layers-per-forward x isolated stream time "
              "(no CUDA graphs, no fusion; lm_head, embeddings, attention and GDN in_proj_ba excluded).\n")
        print("| R | dense ms | ours ms | dense / ours |")
        print("|---|---|---|---|")
        for R in ranks:
            dd, oo = step[("dense", R)], step[("ours", R)]
            print(f"| {R} | {dd / 1e3:.2f} | {oo / 1e3:.2f} | {dd / oo:.2f}x |")
    print("\nCaveats: weights are random (speed and numerics only, not model quality); each case "
          "repeats one layer, so the ring's few-hundred-KB cores stay in L2 between calls while "
          "the dense weight (tens to hundreds of MB) streams from HBM every call — a real "
          "forward pass touches ~400 distinct layers (see tools/graph_bench.py --part chain for "
          "the chained measurement on the assignment shape).")


if __name__ == "__main__":
    main()
