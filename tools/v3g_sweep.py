#!/usr/bin/env python3
"""Every compiled V3G variant: correctness against the FP64 oracle, then stream µs per call.

Correctness: token counts that exercise full and partial token tiles (and T = 1), every call
twice (design B must leave ws / counters clean), |err| <= 0.02 + 0.02 |oracle|.
Timing: CUDA events around 20 calls (tools/measure_kernels.stream_us), per T in --tokens, next
to dense (F.linear, same weights materialised) and the generic kernel's best from --generic
(tools/qwen_bench.py JSON with --tilings, e.g. results/h100/qwen/bf16_tuned.json).
The assignment's FP16 variants are compared with V3T (TR_V3T default) instead.

    python tools/v3g_sweep.py --out results/h100/qwen/v3g_sweep.json \\
        --generic results/h100/qwen/bf16_tuned.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_kernels import stream_us  # noqa: E402
from qwen_shapes import SHAPES  # noqa: E402

from factorized_inference import (  # noqa: E402
    TRSpec, dense_forward, make_cores, materialize_dense_weight,
)
from factorized_inference.tr_kernel import PreparedTRKernel, load_v3g, pack_cores  # noqa: E402

TOL = 2e-2
NAMES = {(ins, outs): n for n, (ins, outs, _) in SHAPES.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,8,32,128")
    ap.add_argument("--generic", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    ext = load_v3g()
    tokens = [int(v) for v in a.tokens.split(",")]
    generic = {}
    if a.generic:
        for c in json.loads(a.generic.read_text())["cases"]:
            if c.get("method") == "factorized_optimized":
                generic[(c["shape"], c["rank"], c["tokens"])] = c["cuda_event_stream_median_ms"] * 1e3
    groups = {}
    for v in ext.variants():
        bf, *modes, R, kc, qc, tt, mg, nt, ks, cl = v
        groups.setdefault((bf, tuple(modes), R), []).append((kc, qc, tt, mg, nt, ks, cl))
    rows, out = [], {"device": torch.cuda.get_device_name(0), "rows": None}
    out["rows"] = rows
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ok_all = True
    for (bf, modes, R), tilings in groups.items():
        ins, outs = tuple(modes[:3]), tuple(modes[3:])
        spec = TRSpec(ins, outs, R)
        dt = torch.bfloat16 if bf else torch.float16
        name = NAMES.get((ins, outs), "assign")
        cores = make_cores(spec, device="cuda", dtype=dt, seed=1)
        A1, B2, C3 = pack_cores(cores, spec)
        w64 = materialize_dense_weight(tuple(c.double() for c in cores), spec)
        W = materialize_dense_weight(cores, spec)
        m7 = [*ins, *outs, R]
        ws = torch.zeros(max(tokens + [37]) * spec.out_features, device="cuda")
        td = torch.zeros(4096, dtype=torch.int32, device="cuda")
        g = torch.Generator(device="cuda").manual_seed(3)
        xs = {T: torch.randn(T, spec.in_features, generator=g, device="cuda", dtype=dt)
              for T in sorted(set(tokens + [1, 3, 5, 37]))}
        refs = {T: dense_forward(x.double(), w64) for T, x in xs.items()}
        base = {T: stream_us(lambda T=T: dense_forward(xs[T], W)) for T in tokens}
        v3t = {}
        if name == "assign":
            run = PreparedTRKernel(cores, spec)
            v3t = {T: stream_us(lambda T=T: run(xs[T])) for T in tokens}
        for kc, qc, tt, mg, nt, ks, cl in tilings:
            call = lambda T: ext.forward(xs[T], A1, B2, C3, m7, kc, qc, tt, mg, nt, ks, cl, ws, td)  # noqa: E731
            worst, ok = 0.0, True
            for T in (1, 3, 5, 37):
                for _ in range(2):
                    y = call(T)
                    d = (y.double() - refs[T]).abs()
                    bad = (d > TOL + TOL * refs[T].abs()).sum().item()
                    worst = max(worst, d.max().item())
                    if bad or y.dtype != dt or not torch.isfinite(y).all():
                        ok = False
            ok_all &= ok
            for T in tokens:
                us = stream_us(lambda T=T: call(T))
                rows.append({"shape": name, "rank": R, "tokens": T, "tiling": (kc, qc, tt, mg, nt, ks, cl),
                             "dtype": "bf16" if bf else "fp16", "correct": ok,
                             "max_abs_err": worst, "v3g_us": us, "dense_us": base[T],
                             "generic_best_us": generic.get((name, R, T)),
                             "v3t_us": v3t.get(T)})
            print(f"{'ok  ' if ok else 'FAIL'} {name:12s} R={R:2d} kc,qc,tt,mg,nt,ks,cl={kc},{qc},{tt},{mg},{nt},{ks},{cl}  "
                  f"max|err|={worst:.2e}  " + "  ".join(
                      f"T{r['tokens']}:{r['v3g_us']:.1f}" for r in rows[-len(tokens):]), flush=True)
            a.out.write_text(json.dumps(out, indent=2) + "\n")
        best = {}
        for r in rows:
            if (r["shape"], r["rank"]) == (name, R) and r["correct"]:
                k = r["tokens"]
                if k not in best or r["v3g_us"] < best[k]["v3g_us"]:
                    best[k] = r
        for T, r in sorted(best.items()):
            gb = r["generic_best_us"]
            print(f"  BEST {name:12s} R={R:2d} T={T:3d}  v3g {r['v3g_us']:8.1f}  dense "
                  f"{r['dense_us']:8.1f}" + (f"  generic {gb:8.1f}" if gb else "")
                  + (f"  v3t {r['v3t_us']:8.1f}" if r["v3t_us"] else "") + f"  {r['tiling']}",
                  flush=True)
        del cores, W, w64, ws
        torch.cuda.empty_cache()
    print("ALL OK" if ok_all else "SOME FAILED")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
