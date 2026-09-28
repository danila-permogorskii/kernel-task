#!/usr/bin/env python3
"""Per layer, fine token sweep: our ring (V3G, R = 8 / 16) against dense BF16, FP8 and INT4.

For every Qwen3.8-27B shape and T in --tokens: CUDA-event stream µs per call (measure_kernels.
stream_us, 20 calls x 5), relative error of each method against the FP32 dense output of the
same weight (ring: against its own dense W; low-bit: quantisation error), and the crossover =
the smallest measured T at which the ring is no faster than that dense variant.

    python tools/lowbit_bench.py --out results/h100/qwen/lowbit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowbit import FP8Linear, INT4Linear  # noqa: E402
from measure_kernels import stream_us  # noqa: E402
from qwen_shapes import SHAPES  # noqa: E402

from factorized_inference import TRSpec, dense_forward, make_cores, materialize_dense_weight  # noqa: E402
from factorized_inference.tr_kernel import PreparedTRKernel  # noqa: E402


def rel(y, ref):
    return (torch.linalg.vector_norm(y.float() - ref) / torch.linalg.vector_norm(ref)).item()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,2,3,4,6,8,10,12,16,20,24,32,48,64")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tokens = [int(v) for v in a.tokens.split(",")]
    rows, cross = [], []
    out = {"device": torch.cuda.get_device_name(0), "rows": rows, "crossover": cross}
    for shape, (ins, outs, n_layers) in SHAPES.items():
        dense = {}
        for R in (8, 16):
            spec = TRSpec(ins, outs, R)
            cores = make_cores(spec, device="cuda", dtype=torch.bfloat16, seed=0)
            W = materialize_dense_weight(cores, spec)          # dense of THIS ring: same function
            ring = PreparedTRKernel(cores, spec)
            meth = {f"ring R={R}": ring}
            if R == 8:  # dense variants of one weight are enough for the timing
                dense = {"dense bf16": lambda x, W=W: dense_forward(x, W),
                         "dense fp8": FP8Linear(W), "dense int4": INT4Linear(W)}
                meth.update(dense)
            Wf = W.float()
            for T in tokens:
                x = torch.randn(T, spec.in_features, device="cuda", dtype=torch.bfloat16)
                ref = x.float() @ Wf.T
                for name, fn in meth.items():
                    us = stream_us(lambda fn=fn: fn(x))
                    rows.append({"shape": shape, "tokens": T, "method": name, "us": us,
                                 "rel_err": rel(fn(x), ref)})
            del W, Wf, cores, ring
            torch.cuda.empty_cache()
        by = {(r["method"], r["tokens"]): r["us"] for r in rows if r["shape"] == shape}
        line = [f"{shape:12s}"]
        for R in (8, 16):
            for d in ("dense bf16", "dense fp8", "dense int4"):
                t = next((T for T in tokens if by[(f"ring R={R}", T)] >= by[(d, T)]), None)
                cross.append({"shape": shape, "rank": R, "vs": d, "first_T_ring_not_faster": t,
                              "t1_speedup": by[(d, 1)] / by[(f"ring R={R}", 1)]})
                line.append(f"R{R} vs {d.split()[1]}: T*={t} (T=1 x{by[(d, 1)] / by[(f'ring R={R}', 1)]:.2f})")
        print("  ".join(line), flush=True)
        for T in tokens:
            print(f"   T={T:3d} " + "  ".join(f"{m} {by[(m, T)]:7.1f}" for m in
                  ("dense bf16", "dense fp8", "dense int4", "ring R=8", "ring R=16")), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
