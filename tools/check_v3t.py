"""Correctness of the V3T path (t > 1, real modes) against the FP64 dense oracle: every compiled
tiling that fits this GPU, both designs, ragged token counts, several token counts on one
prepared object (design B must leave its workspace and counters clean), and agreement with the
WMMA kernel (TR_V3T=0).

    python tools/check_v3t.py
"""
from __future__ import annotations

import os

import torch

from factorized_inference import TRSpec, make_cores, materialize_dense_weight, dense_forward
from factorized_inference.tr_kernel import PreparedTRKernel

TOL = 2e-2  # FP16 acceptance in IMPLEMENTATION.md
# must match TR_V3T_TILINGS in csrc/tr_ring.cu: (R, kc, qc, tt, mg)
TILINGS = [(8, 2, 5, 4, 1), (8, 2, 5, 4, 3), (8, 4, 5, 4, 3), (8, 4, 5, 8, 1), (8, 5, 5, 4, 1),
           (8, 5, 5, 4, 3), (8, 10, 5, 4, 3), (8, 4, 10, 4, 3),
           (16, 2, 10, 4, 1), (16, 2, 10, 4, 3), (16, 4, 10, 4, 3), (16, 4, 10, 4, 1),
           (16, 5, 10, 4, 1), (16, 5, 10, 4, 3)]
TOKENS = (2, 3, 8, 5, 12, 9, 32, 17, 33, 8)  # ragged and full tiles, sizes going up and down


def main():
    ok, ran = True, 0
    for R in (8, 16):
        spec = TRSpec(rank=R)
        cores = make_cores(spec, device="cuda", dtype=torch.float16, seed=5)
        W = materialize_dense_weight(tuple(c.double() for c in cores), spec)
        for (r, kc, qc, tt, mg) in TILINGS:
            if r != R:
                continue
            for design in ("A", "B"):
                os.environ.update(TR_DESIGN=design, TR_V3T_TILING=f"{kc},{qc},{tt},{mg}")
                run = PreparedTRKernel(cores, spec)
                if run.v3t_tiling(8) is None:
                    print(f"skip V3T R={R:2d} kc={kc} qc={qc} tt={tt} mg={mg}: does not fit this GPU")
                    break
                g = torch.Generator(device="cuda").manual_seed(13)
                worst, bad_total = 0.0, 0
                for T in TOKENS:
                    x = torch.randn(T, spec.in_features, generator=g, device="cuda",
                                    dtype=torch.float16)
                    exp = dense_forward(x.double(), W)
                    y = run(x)
                    torch.cuda.synchronize()
                    d = (y.double() - exp).abs()
                    bad_total += (d > TOL + TOL * exp.abs()).sum().item()
                    worst = max(worst, d.max().item())
                    ok &= y.shape == exp.shape
                good = bad_total == 0
                ok &= good
                ran += 1
                # the last call through the WMMA kernel must agree
                os.environ["TR_V3T"] = "0"
                y2 = PreparedTRKernel(cores, spec)(x)
                os.environ.pop("TR_V3T")
                diff = (y2.float() - y.float()).abs().max().item()
                ok &= diff < TOL
                print(f"{'ok  ' if good and diff < TOL else 'FAIL'} V3T R={R:2d} kc={kc} qc={qc} tt={tt} mg={mg}"
                      f" design {design}  max|err| {worst:.2e}  vs WMMA {diff:.2e}", flush=True)
    for k in ("TR_DESIGN", "TR_V3T_TILING"):
        os.environ.pop(k, None)
    print(f"{ran} configurations")
    print("ALL OK" if ok and ran else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok and ran else 1)


if __name__ == "__main__":
    main()
