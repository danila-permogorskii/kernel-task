#!/usr/bin/env python3
"""Correctness of the BF16 kernel path against the FP64 dense oracle.

Covers: odd small modes (padding, the scalar stage-1 path), the assignment's modes, every
Qwen3.8-27B shape of tools/qwen_shapes.py at R = 8 and 16, designs A and B, changing token
counts on one prepared object, and every call twice (design B must leave its workspace clean).
Acceptance: |y - oracle| <= 0.02 + 0.02 |oracle| (the harness's FP16 rule). For scale, the
relative L2 error of the dense BF16 matmul on the same inputs is printed next to ours.

    python tools/check_bf16.py            # everything
    python tools/check_bf16.py --quick    # small modes + mlp_gate_up only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from factorized_inference import TRSpec, dense_forward, make_cores, materialize_dense_weight  # noqa: E402
from factorized_inference.tr_kernel import PreparedTRKernel, load_extension  # noqa: E402
from qwen_shapes import SHAPES  # noqa: E402

TOL = 2e-2
DT = torch.bfloat16


def rel_l2(a, b):
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def run_case(spec, design, tokens_seq, seed=0):
    os.environ["TR_DESIGN"] = design
    cores = make_cores(spec, device="cuda", dtype=DT, seed=seed)
    run = PreparedTRKernel(cores, spec)
    w64 = materialize_dense_weight(tuple(c.double() for c in cores), spec)
    w16 = materialize_dense_weight(cores, spec)  # the dense BF16 baseline, for scale only
    g = torch.Generator(device="cuda").manual_seed(seed + 5)
    worst, ours_l2, dense_l2 = 0.0, 0.0, 0.0
    for T in tokens_seq:
        if run.tiling(T) is None:
            return None, f"T={T}: no tiling fits this GPU"
        x = torch.randn(T, spec.in_features, generator=g, device="cuda", dtype=DT)
        exp = dense_forward(x.double(), w64)
        dense_l2 = max(dense_l2, rel_l2(dense_forward(x, w16).double(), exp))
        for _ in range(2):
            y = run(x)
            torch.cuda.synchronize()
            d = (y.double() - exp).abs()
            bad = (d > TOL + TOL * exp.abs()).sum().item()
            worst = max(worst, d.max().item())
            ours_l2 = max(ours_l2, rel_l2(y.double(), exp))
            if bad or y.shape != exp.shape or y.dtype != DT or not torch.isfinite(y).all():
                return False, f"T={T}: {bad} of {exp.numel()} outside tolerance, max|err|={worst:.3g}"
    return True, (f"max|err|={worst:.2e}  relL2 ours={ours_l2:.2e} dense-bf16={dense_l2:.2e}  "
                  f"tilings={[run.tiling(T) for T in tokens_seq]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    load_extension()
    cases = [
        (TRSpec((2, 3, 4), (5, 6, 7), 3), (1, 5, 2)),        # scalar stage 1, heavy padding
        (TRSpec((4, 4, 4), (4, 4, 4), 8), (1, 3)),
        (TRSpec((8, 12, 20), (12, 10, 24), 8), (1, 8, 32)),  # assignment modes, generic kernel
        (TRSpec((8, 12, 20), (12, 10, 24), 16), (1, 32)),
    ]
    for name, (ins, outs, _) in SHAPES.items():
        if a.quick and name != "mlp_gate_up":
            continue
        for R in (8, 16):
            cases.append((TRSpec(ins, outs, R), (1, 8, 3, 32)))
    ok = True
    for spec, toks in cases:
        for design in ("A", "B"):
            res, msg = run_case(spec, design, toks)
            tag = {True: "ok  ", False: "FAIL", None: "skip"}[res]
            ok &= res is not False
            print(f"{tag} {design} {spec.input_modes}->{spec.output_modes} R={spec.rank:2d} "
                  f"T={toks}  {msg}", flush=True)
            torch.cuda.empty_cache()
    print("ALL OK" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
