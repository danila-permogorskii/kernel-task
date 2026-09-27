"""V3T (t > 1) tiling sweep on the real modes: every compiled (kc, qc, tt), design B, next to
the WMMA kernel (TR_V3T=0) and dense, same process. Stream µs per call (like the harness) and
kernel-only µs (profiler).

    python tools/v3t_sweep.py --out results/h100/v3t/sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_kernels import kernel_us, stream_us  # noqa: E402
from check_v3t import TILINGS  # noqa: E402

from factorized_inference import TRSpec, dense_forward, make_cores, materialize_dense_weight  # noqa: E402
from factorized_inference.tr_kernel import PreparedTRKernel  # noqa: E402

CASES = [(8, 8), (8, 32), (16, 32), (8, 2), (8, 4), (8, 16), (16, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    rows = []
    for R, T in CASES:
        spec = TRSpec(rank=R)
        cores = make_cores(spec, device="cuda", dtype=torch.float16, seed=0)
        x = torch.randn(T, spec.in_features, device="cuda", dtype=torch.float16)
        W = materialize_dense_weight(cores, spec)
        variants = [("dense", None), ("wmma", None)] + [
            (f"v3t {kc},{qc},{tt},{mg}", (kc, qc, tt, mg)) for (r, kc, qc, tt, mg) in TILINGS if r == R]
        for name, tiling in variants:
            os.environ["TR_DESIGN"] = "B"
            os.environ.pop("TR_V3T_TILING", None)
            os.environ.pop("TR_V3T", None)
            if name == "dense":
                fn = lambda: dense_forward(x, W)  # noqa: E731
            else:
                if name == "wmma":
                    os.environ["TR_V3T"] = "0"
                else:
                    os.environ["TR_V3T_TILING"] = ",".join(map(str, tiling))
                run = PreparedTRKernel(cores, spec)
                if tiling is not None and run.v3t_tiling(T) is None:
                    continue
                fn = lambda: run(x)  # noqa: E731
            s = stream_us(fn)
            k, _ = kernel_us(fn)
            rows.append({"rank": R, "tokens": T, "variant": name, "stream_us": s, "kernel_us": k})
            print(f"R={R:2d} T={T:2d} {name:16s} stream {s:8.2f}  kernel {k:8.2f} µs", flush=True)
    for k in ("TR_DESIGN", "TR_V3T_TILING", "TR_V3T"):
        os.environ.pop(k, None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "rows": rows}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
