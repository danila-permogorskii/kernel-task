"""Correctness sweep for the fused kernel, designs A and B, against the FP64 dense oracle.

Covers: odd small modes (padding paths), the real R = 8 / 16 workloads, many tilings
(kc, tt), changing token counts on one prepared object, and repeated calls (design B must
leave its workspace clean for the next call).

    python tools/check_kernel.py            # full sweep
    python tools/check_kernel.py --quick    # real workloads only
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from factorized_inference import TRSpec, make_cores, materialize_dense_weight, dense_forward
from factorized_inference.tr_kernel import PreparedTRKernel, load_extension

TOL = 2e-2  # FP16 acceptance in IMPLEMENTATION.md: atol = rtol = 0.02


def oracle(x, cores, spec):
    return dense_forward(x.double(), materialize_dense_weight(tuple(c.double() for c in cores), spec))


def run_case(spec, design, tokens_seq, kc=None, tt=None, seed=0):
    os.environ["TR_DESIGN"] = design
    for name, val in (("TR_KC", kc), ("TR_TT", tt)):
        if val is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(val)
    cores = make_cores(spec, device="cuda", dtype=torch.float16, seed=seed)
    run = PreparedTRKernel(cores, spec)
    g = torch.Generator(device="cuda").manual_seed(seed + 5)
    worst = 0.0
    if any(run.tiling(T) is None for T in tokens_seq):
        return None, 0.0  # does not fit this GPU's shared memory: the kernel is not used
    for T in tokens_seq:
        x = torch.randn(T, spec.in_features, generator=g, device="cuda", dtype=torch.float16)
        exp = oracle(x, cores, spec)
        for _ in range(2):  # twice: B must have cleaned up after the first call
            y = run(x)
            torch.cuda.synchronize()
            bad = ((y.double() - exp).abs() > TOL + TOL * exp.abs()).sum().item()
            err = (y.double() - exp).abs().max().item()
            worst = max(worst, err)
            if bad or y.shape != exp.shape or y.dtype != torch.float16:
                print(f"FAIL {design} {spec} T={T} kc={kc} tt={tt}: {bad} bad, max|err|={err:.3g}")
                return False, worst
    return True, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    load_extension(verbose=False)
    print("device:", torch.cuda.get_device_name(0))

    cases = [  # (spec, token sequence, kc, tt)
        (TRSpec(rank=8), (1, 8, 32, 5), None, None),
        (TRSpec(rank=16), (1, 32, 3), None, None),
    ]
    if not args.quick:
        cases += [
            (TRSpec((2, 3, 4), (5, 6, 7), 3), (1, 5, 2), None, None),
            (TRSpec((2, 3, 2), (2, 4, 3), 3), (3, 1), None, None),
            (TRSpec((4, 4, 4), (4, 4, 4), 16), (1, 4, 9), None, None),
            (TRSpec((8, 6, 20), (12, 5, 24), 16), (1, 32, 7), None, None),  # R=16 paths, smaller B
            (TRSpec((8, 6, 20), (12, 5, 24), 16), (32, 5), 3, 4),
            (TRSpec(rank=8), (32, 7), 4, 4),
            (TRSpec(rank=8), (32, 13), 3, 5),   # ragged k chunks and token tiles
            (TRSpec(rank=16), (32, 9), 2, 2),
            (TRSpec(rank=16), (1,), 20, 1),     # one block per a
        ]
    ok_all = True
    for design in ("A", "B"):
        for spec, toks, kc, tt in cases:
            ok, worst = run_case(spec, design, toks, kc, tt)
            if ok is None:
                print(f"skip design {design}  modes={spec.input_modes}->{spec.output_modes} "
                      f"R={spec.rank:2d}: does not fit this GPU's shared memory")
                continue
            ok_all &= ok
            print(f"{'ok  ' if ok else 'FAIL'} design {design}  modes={spec.input_modes}->"
                  f"{spec.output_modes} R={spec.rank:2d} tokens={toks} kc={kc} tt={tt}  "
                  f"max|err|={worst:.2e}")
    print("ALL OK" if ok_all else "SOME CASES FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
