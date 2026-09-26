"""Hopper floors part 2 (floors2.cu): mma chains, lane moves, __syncthreads, dense stack.

    python kernel_work/hopper_floors/floors2.py --out results/h100/floors2.json
    python kernel_work/hopper_floors/floors2.py --dense-only --out results/h100/floors2_dense.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
from torch.utils import cpp_extension

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LAYER_BYTES = 2880 * 1920 * 2
DENSE_NAMES = {0: "ld.global.v4", 1: "ld.global.nc (__ldg)", 2: "ld.global.cs (__ldcs)",
               3: "ld.global.nc.L1::no_allocate", 11: "prefetch next layer to L2",
               12: "prefetch 2 layers ahead", 14: "prefetch 4 layers ahead"}


def load():
    shim = REPO / ".cuda_home"
    if "CUDA_HOME" not in os.environ and shim.exists():
        os.environ["CUDA_HOME"] = str(shim)
    if cpp_extension.CUDA_HOME is None and "CUDA_HOME" in os.environ:
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]
    build = REPO / "build" / "hopper_floors2"
    build.mkdir(parents=True, exist_ok=True)
    return cpp_extension.load(name="hopper_floors2", sources=[str(HERE / "floors2.cu")],
                              build_directory=str(build), extra_cuda_cflags=["-O3"])


def stream_us(fn, calls=10, reps=7):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(calls):
            fn()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(out)


def micro_floors(ext, res):
    out = torch.zeros(32, device="cuda")
    print("== cycles per instruction / step (one warp, clock64)")
    n = 20000
    for v, nm in ((0, "mma.sync m16n8k16, one dependent chain"),
                  (1, "mma.sync, two independent chains"),
                  (2, "mma.sync, four independent chains"),
                  (3, "mma.sync chain + FMUL on the accumulator")):
        cyc = ext.micro(0, v, n, out)[0] / (4 * n)
        res["micro"].append({"what": nm, "cycles": cyc})
        print(f"  {nm:44s} {cyc:6.2f}")
    for v, nm in ((0, "__shfl_down_sync, dependent"), (1, "shared-memory hop, dependent")):
        cyc = ext.micro(1, v, n, out)[0] / n
        res["micro"].append({"what": nm, "cycles": cyc})
        print(f"  {nm:44s} {cyc:6.2f}")
    cyc = ext.micro(2, 0, n, out)[0] / n
    res["micro"].append({"what": "__syncthreads, 256 threads", "cycles": cyc})
    print(f"  {'__syncthreads, 256 threads':44s} {cyc:6.2f}")


def dense_floors(ext, res, variants, grids):
    dev = "cuda"
    print("== dense stack in one persistent kernel (Kog-style baseline), batch 1")
    Lmax = 128
    torch.manual_seed(0)
    Wup = (torch.randn(Lmax // 2, 2880, 1920, device=dev) / 1920 ** 0.5).half()
    Wdn = (torch.randn(Lmax // 2, 1920, 2880, device=dev) / 2880 ** 0.5).half()
    buf = torch.zeros(2 * 2880, device=dev)
    sync = torch.zeros(4, dtype=torch.int32, device=dev)
    base = [0]
    x = torch.randn(1, 1920, device=dev, dtype=torch.float16)

    def call(L, variant, grid):
        g = grid if grid > 0 else ext.dense_grid(variant)
        need = g * (L + 1)
        if base[0] + need >= 2 ** 31 - 1:
            sync.zero_()
            base[0] = 0
        y = ext.dense_stack(x, Wup, Wdn, L, buf, sync, base[0], variant, grid)
        base[0] += need
        return y

    ref = x.float()
    for l in range(4):
        ref = torch.nn.functional.linear(ref, (Wup[l // 2] if l % 2 == 0 else Wdn[l // 2]).float())
    for variant in variants:
        y = call(4, variant, 264)
        torch.cuda.synchronize()
        rel = (torch.linalg.vector_norm(y.float() - ref) / torch.linalg.vector_norm(ref)).item()
        print(f"  correctness {DENSE_NAMES[variant]:30s} L=4: rel L2 {rel:.2e}")
        for grid in grids:
            pts = [(L, stream_us(lambda: call(L, variant, grid))) for L in (8, 32, 128)]
            mx = sum(p[0] for p in pts) / 3
            my = sum(p[1] for p in pts) / 3
            slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
            gbs = LAYER_BYTES / (slope * 1e-6) / 1e9
            res["dense"].append({"variant": variant, "name": DENSE_NAMES[variant], "grid": grid,
                                 "rel_l2": rel, "points": pts, "us_per_layer": slope, "gbs": gbs})
            print(f"  {DENSE_NAMES[variant]:30s} grid {grid:4d}: {slope:6.2f} µs/layer  "
                  f"{gbs:6.0f} GB/s ({100 * gbs / 3350:4.1f}% of HBM peak)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dense-only", action="store_true")
    args = ap.parse_args()
    ext = load()
    res = {"device": torch.cuda.get_device_name(0), "micro": [], "dense": []}
    if not args.dense_only:
        micro_floors(ext, res)
        dense_floors(ext, res, (0, 1, 2, 3), (132, 264, 528))
    else:
        dense_floors(ext, res, (0, 11, 12, 14), (264, 528))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
