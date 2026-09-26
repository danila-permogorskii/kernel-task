"""Run the Hopper floors (floors.cu) and print µs per operation.

    python kernel_work/hopper_floors/floors.py --out results/h100/floors.json
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


def load():
    shim = REPO / ".cuda_home"
    if "CUDA_HOME" not in os.environ and shim.exists():
        os.environ["CUDA_HOME"] = str(shim)
    if cpp_extension.CUDA_HOME is None and "CUDA_HOME" in os.environ:
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]
    build = REPO / "build" / "hopper_floors"
    build.mkdir(parents=True, exist_ok=True)
    return cpp_extension.load(name="hopper_floors", sources=[str(HERE / "floors.cu")],
                              build_directory=str(build),
                              extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])


def per_op_us(fn, iters, reps=7):
    """fn(iters) launches one kernel doing `iters` operations; µs per operation from the
    difference between iters and iters // 4 (cancels the launch and the prologue)."""
    def t(n):
        fn(n)
        torch.cuda.synchronize()
        out = []
        for _ in range(reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn(n)
            e.record()
            e.synchronize()
            out.append(s.elapsed_time(e) * 1000)
        return statistics.median(out)
    lo, hi = iters // 4, iters
    return (t(hi) - t(lo)) / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ext = load()
    dev = "cuda"
    res = {"device": torch.cuda.get_device_name(0), "barrier": [], "copy": [], "atomic": []}

    sync = torch.zeros(4, dtype=torch.int32, device=dev)
    names = {0: "ours (all threads fence)", 1: "cg grid.sync", 2: "ours + nanosleep(64)",
             3: "ours, fence in thread 0 only", 4: "monotonic red.release / ld.acquire",
             5: "cluster barrier + monotonic"}
    print("== grid barrier, µs per barrier")
    for grid in (132, 264):
        for v in (0, 1, 2, 3, 4):
            us = per_op_us(lambda n: ext.barrier_bench(v, grid, 1, n, sync), 4000)
            res["barrier"].append({"variant": v, "name": names[v], "grid": grid, "cluster": 1, "us": us})
            print(f"  grid {grid:3d}  {names[v]:36s} {us:6.3f}", flush=True)
        for cl in (2, 4, 8, 16):
            if grid % cl:
                continue
            try:
                cap = ext.max_clusters(cl)
            except RuntimeError as e:
                print(f"  cluster {cl}: {str(e).splitlines()[0][:70]}")
                continue
            if cap * cl < grid:
                print(f"  grid {grid:3d}  cluster {cl:2d}: only {cap} clusters resident, skipped")
                continue
            us = per_op_us(lambda n: ext.barrier_bench(5, grid, cl, n, sync), 4000)
            res["barrier"].append({"variant": 5, "name": names[5], "grid": grid, "cluster": cl, "us": us})
            print(f"  grid {grid:3d}  {names[5]:28s} x{cl:<2d}      {us:6.3f}", flush=True)

    print("== L2 -> shared copy of one chunk, µs per copy (every block, 264 blocks)")
    sink = torch.zeros(1, device=dev)
    for nbytes in (8192, 16384, 24576, 49152):
        src = torch.randn(8 * nbytes // 2, device=dev, dtype=torch.float16)
        for v, nm in ((0, "256 threads, int4 loads"), (1, "cp.async.bulk, 1 thread"),
                      (2, "cp.async.bulk, 4 threads x 1/4")):
            us = per_op_us(lambda n: ext.copy_bench(v, 264, nbytes, n, src, sink), 400)
            res["copy"].append({"variant": v, "name": nm, "bytes": nbytes, "us": us})
            print(f"  {nbytes // 1024:3d} KB  {nm:32s} {us:6.3f}", flush=True)

    print("== 264 blocks x 1152 float adds into 2880 floats, µs per round (incl. fence)")
    out = torch.zeros(2880, device=dev)
    for v, nm in ((0, "atomicAdd (scalar)"), (1, "red.global.add.v4.f32")):
        us = per_op_us(lambda n: ext.atomic_bench(v, 264, n, out), 400)
        res["atomic"].append({"variant": v, "name": nm, "us": us})
        print(f"  {nm:28s} {us:6.3f}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
