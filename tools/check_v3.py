"""Correctness of the V3 path (t = 1, real modes) against the FP64 dense oracle, both designs,
every compiled tiling, repeated calls (design B must leave its workspace clean).

    python tools/check_v3.py
"""
from __future__ import annotations

import os

import torch

from factorized_inference import TRSpec, make_cores, materialize_dense_weight, dense_forward
from factorized_inference.tr_kernel import PreparedTRKernel

TOL = 2e-2  # FP16 acceptance in IMPLEMENTATION.md


def main():
    ok = True
    for R, tilings in ((8, ((2, 4), (2, 5))), (16, ((4, 4), (4, 5)))):
        spec = TRSpec(rank=R)
        cores = make_cores(spec, device="cuda", dtype=torch.float16, seed=3)
        W = materialize_dense_weight(tuple(c.double() for c in cores), spec)
        for kc, qc in tilings:
            for design, tail in (("A", 0), ("B", 0), ("B", 1), ("B", 2)):
                os.environ.update(TR_DESIGN=design, TR_KC=str(kc), TR_TT="1", TR_QC=str(qc),
                                  TR_V3_TAIL=str(tail))
                run = PreparedTRKernel(cores, spec)
                assert run.ext.v3_supported(R, kc, qc)
                g = torch.Generator(device="cuda").manual_seed(11)
                worst = 0.0
                for _ in range(3):
                    x = torch.randn(1, spec.in_features, generator=g, device="cuda",
                                    dtype=torch.float16)
                    exp = dense_forward(x.double(), W)
                    y = run(x)
                    torch.cuda.synchronize()
                    bad = ((y.double() - exp).abs() > TOL + TOL * exp.abs()).sum().item()
                    worst = max(worst, (y.double() - exp).abs().max().item())
                    ok &= bad == 0 and y.shape == exp.shape
                print(f"{'ok  ' if worst < TOL else 'FAIL'} V3 R={R:2d} kc={kc} qc={qc} design {design}"
                      f" tail {tail}  max|err| = {worst:.2e}")
                # the same call through the WMMA kernel must agree
                os.environ["TR_V3"] = "0"
                y2 = PreparedTRKernel(cores, spec)(x)
                os.environ.pop("TR_V3")
                diff = (y2.float() - y.float()).abs().max().item()
                print(f"     WMMA vs V3 max|diff| = {diff:.2e}")
                ok &= diff < TOL
    for k in ("TR_KC", "TR_TT", "TR_QC", "TR_DESIGN", "TR_V3_TAIL"):
        os.environ.pop(k, None)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
