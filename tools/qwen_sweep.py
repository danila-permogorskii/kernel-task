#!/usr/bin/env python3
"""(kc, tt, qc) tiling sweep of the generic kernel on Qwen3.8-27B shapes, next to dense.

choose_tiling() in tr_kernel.py was written for the assignment's small modes; on the real
shapes it often lands on qc = 1 (stage 1 recomputed for every q) because the Y tile and S1/S2
must fit. This measures every tiling that fits, so the rule can be checked against the card.

Per (shape, R, T): dense, the chooser's pick, then every fitting tiling in order of predicted
work, until --budget-s is spent. Stream µs per call (CUDA events, like the harness), design B.
Each tiling's output is checked against the dense BF16 output (relative L2 < 1e-2).

    python tools/qwen_sweep.py --out results/a100/qwen/sweep.json
    python tools/qwen_sweep.py --shapes mlp_gate_up --ranks 8 --tokens 1,32 --budget-s 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_kernels import stream_us  # noqa: E402
from qwen_shapes import SHAPES  # noqa: E402

from factorized_inference import TRSpec, dense_forward, make_cores, materialize_dense_weight  # noqa: E402
from factorized_inference.tr_kernel import PreparedTRKernel, load_extension  # noqa: E402

ENV = ("TR_KC", "TR_TT", "TR_QC")


def r16(v):
    return -(-v // 16) * 16


def chunk_sizes(n):
    """Balanced chunk sizes ceil(n / c), largest first."""
    return sorted({-(-n // c) for c in range(1, n + 1)}, reverse=True)


def predicted_work(spec, T, kc, tt, qc):
    """Padded work of one call (MAC), stage 1 weighted x8 (CUDA cores, not Tensor Cores)."""
    (ni, nj, nk), (P, Q, Rr), R = spec.input_modes, spec.output_modes, spec.rank
    tiles, nqc = -(-T // tt), -(-Q // qc)
    Mp = r16(tt * P)
    per_k = 8 * tt * P * nj * R * ni + Mp * r16(nj * R) * qc * r16(R) + Mp * qc * r16(R) * r16(Rr)
    return R * nk * tiles * nqc * per_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="mlp_gate_up,mlp_down")
    ap.add_argument("--ranks", default="8,16")
    ap.add_argument("--tokens", default="1,8,32,128")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--budget-s", type=float, default=45.0, help="per (shape, R, T)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dtype = getattr(torch, a.dtype)
    ext = load_extension()
    props = torch.cuda.get_device_properties(0)
    smem_limit = props.shared_memory_per_block_optin - 1024
    os.environ["TR_DESIGN"] = "B"
    rows = []
    out = {"device": torch.cuda.get_device_name(0), "dtype": a.dtype, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for shape in a.shapes.split(","):
        ins, outs, _ = SHAPES[shape]
        for R in (int(v) for v in a.ranks.split(",")):
            spec = TRSpec(ins, outs, R)
            modes = [*ins, *outs, R]
            cores = make_cores(spec, device="cuda", dtype=dtype, seed=0)
            W = materialize_dense_weight(cores, spec)
            for T in (int(v) for v in a.tokens.split(",")):
                x = torch.randn(T, spec.in_features, device="cuda", dtype=dtype)
                ref = dense_forward(x, W).float()
                base = dict(shape=shape, rank=R, tokens=T)
                d_us = stream_us(lambda: dense_forward(x, W))
                rows.append({**base, "variant": "dense", "stream_us": d_us})
                for k in ENV:
                    os.environ.pop(k, None)
                run = PreparedTRKernel(cores, spec)
                pick = run.tiling(T)
                p_us = stream_us(lambda: run(x))
                rows.append({**base, "variant": "chooser", "tiling": pick, "stream_us": p_us})
                print(f"{shape:12s} R={R:2d} T={T:3d} dense {d_us:9.1f}  chooser {pick} "
                      f"{p_us:9.1f} us", flush=True)
                cands = [(kc, tt, qc) for kc in chunk_sizes(ins[2]) for tt in (1, 2, 4)
                         for qc in chunk_sizes(outs[1])
                         if tt <= T and ext.smem_bytes(modes, kc, tt, qc) <= smem_limit
                         and ext.y_tiles(modes, tt, qc) <= ext.max_y_tiles()]
                cands.sort(key=lambda c: predicted_work(spec, T, *c))
                best, t_end, tried = (p_us, pick), time.time() + a.budget_s, 0
                for kc, tt, qc in cands:
                    if time.time() > t_end:
                        break
                    os.environ.update(TR_KC=str(kc), TR_TT=str(tt), TR_QC=str(qc))
                    run = PreparedTRKernel(cores, spec)
                    y = run(x).float()
                    err = (torch.linalg.vector_norm(y - ref) / torch.linalg.vector_norm(ref)).item()
                    us = stream_us(lambda: run(x), calls=10, reps=3)
                    tried += 1
                    rows.append({**base, "variant": "forced", "tiling": (kc, tt, qc),
                                 "stream_us": us, "rel_l2_vs_dense": err})
                    if err > 1e-2:
                        print(f"   BAD OUTPUT tiling {(kc, tt, qc)} relL2 {err:.2e}", flush=True)
                    elif us < best[0]:
                        best = (us, (kc, tt, qc))
                print(f"{'':12s} {tried}/{len(cands)} tilings tried, best {best[1]} {best[0]:9.1f} us "
                      f"(dense {d_us:.1f})", flush=True)
                for k in ENV:
                    os.environ.pop(k, None)
                a.out.write_text(json.dumps(out, indent=2) + "\n")
            del W, cores
            torch.cuda.empty_cache()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
