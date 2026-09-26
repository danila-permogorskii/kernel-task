"""Kernel-level measurements that the harness does not give: floors, kernel-only time,
% of the hardware ceilings, and a (kc, tt) tiling sweep.

    python tools/measure_kernels.py --out results/h100/kernels.json
    python tools/measure_kernels.py --sweep --out results/h100/sweep.json

Numbers printed:
  floors      host cost of an empty extension call; GPU time of an empty kernel
  per case    stream µs/call (like the harness), kernel-only µs/call (profiler),
              achieved TFLOP/s and % of FP16 Tensor Core peak, bytes/time vs HBM peak
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from torch.autograd import DeviceType

from factorized_inference import (
    TRSpec, dense_forward, make_cores, materialize_dense_weight, tr_forward_reference,
)
from factorized_inference.tr_kernel import PreparedTRKernel, load_extension

# H100 SXM5 datasheet peaks (dense, no sparsity). Other GPUs: pass --peak-tflops / --peak-gbs.
H100_SXM_FP16_TC_TFLOPS = 989.4
H100_SXM_HBM_GBS = 3350.0

CASES = [(8, 1), (8, 8), (8, 32), (16, 1), (16, 32)]  # the five required (rank, tokens)


def flops_per_call(spec: TRSpec, T: int) -> dict:
    ni, nj, nk = spec.input_modes
    P, Q, Rr = spec.output_modes
    R = spec.rank
    pieces = R * nk
    s1 = pieces * T * nj * P * R * ni
    s2 = pieces * T * P * Q * R * nj * R
    s3 = pieces * T * P * Q * Rr * R
    return {"stage1": 2 * s1, "stage2": 2 * s2, "stage3": 2 * s3, "total": 2 * (s1 + s2 + s3)}


def min_bytes_per_call(spec: TRSpec, T: int) -> int:
    """Compulsory traffic: cores + x + y, FP16. Everything else should stay on-chip / in L2."""
    return 2 * (spec.factor_parameters + T * spec.in_features + T * spec.out_features)


def stream_us(fn, calls=20, reps=5) -> float:
    """Like the harness: CUDA events around `calls` back-to-back calls, median of reps."""
    for _ in range(3):
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


def kernel_us(fn, calls=20) -> tuple[float, dict]:
    """GPU kernel time per call from the profiler (sum over every kernel the call launches)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    per_name: dict[str, float] = {}
    for ev in prof.events():
        if ev.device_type == DeviceType.CUDA:
            per_name[ev.name] = per_name.get(ev.name, 0.0) + ev.time_range.elapsed_us() / calls
    return sum(per_name.values()), per_name


def host_us(fn, n=2000) -> float:
    for _ in range(50):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    torch.cuda.synchronize()
    return statistics.median(ts) / 1000


def floors(ext) -> dict:
    return {
        "host_empty_extension_call_us": host_us(ext.empty_call),
        "host_empty_launch_call_us": host_us(ext.empty_launch),
        "stream_empty_kernel_us": stream_us(ext.empty_launch),
        "kernel_empty_us": kernel_us(ext.empty_launch)[0],
    }


def measure_case(R, T, design, peak_tflops, peak_gbs, kc=None, tt=None):
    spec = TRSpec(rank=R)
    os.environ["TR_DESIGN"] = design
    for name, val in (("TR_KC", kc), ("TR_TT", tt)):
        os.environ.pop(name, None) if val is None else os.environ.__setitem__(name, str(val))
    cores = make_cores(spec, device="cuda", dtype=torch.float16, seed=0)
    x = torch.randn(T, spec.in_features, device="cuda", dtype=torch.float16)
    if design == "ref":
        fn = lambda: tr_forward_reference(x, cores, spec)  # noqa: E731
        tiling = None
    elif design == "dense":  # baseline only: the dense W exists solely in this branch
        W = materialize_dense_weight(cores, spec)
        fn = lambda: dense_forward(x, W)  # noqa: E731
        tiling = None
    elif design == "compile":  # bonus: torch.compile of the reference, default mode (no CUDA graphs)
        compiled = torch.compile(lambda x_: tr_forward_reference(x_, cores, spec))
        fn = lambda: compiled(x)  # noqa: E731
        tiling = None
    else:
        run = PreparedTRKernel(cores, spec)
        fn = lambda: run(x)  # noqa: E731
        tiling = run.tiling(T)
        if tiling is None:
            raise RuntimeError("does not fit in shared memory on this GPU")
    s_us = stream_us(fn)
    k_us, names = kernel_us(fn)
    fl = flops_per_call(spec, T)
    by = min_bytes_per_call(spec, T)
    main_us = sum(v for n, v in names.items() if "tr_ring_fused" in n) or k_us
    tflops = fl["total"] / (main_us * 1e-6) / 1e12
    gbs = by / (main_us * 1e-6) / 1e9
    return {
        "rank": R, "tokens": T, "design": design, "kc_tt": tiling,
        "stream_us_per_call": s_us, "kernel_us_per_call": k_us, "kernels": names,
        "fused_kernel_us": main_us, "flops": fl, "min_bytes": by,
        "achieved_tflops": tflops, "pct_fp16_tc_peak": 100 * tflops / peak_tflops,
        "achieved_gbs": gbs, "pct_hbm_peak": 100 * gbs / peak_gbs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sweep", action="store_true", help="(kc, tt) sweep for the T = 32 cases")
    ap.add_argument("--with-compile", action="store_true", help="add torch.compile rows (bonus)")
    ap.add_argument("--peak-tflops", type=float, default=H100_SXM_FP16_TC_TFLOPS)
    ap.add_argument("--peak-gbs", type=float, default=H100_SXM_HBM_GBS)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # as the harness
    ext = load_extension()
    result = {"device": torch.cuda.get_device_name(0),
              "peaks": {"fp16_tc_tflops": args.peak_tflops, "hbm_gbs": args.peak_gbs}}

    if args.sweep:
        rows = []
        for R, T in ((8, 32), (16, 32), (16, 1), (8, 1)):
            for kc in (1, 2, 4, 5, 10):
                for tt in (1, 2, 4, 8, 16, 32):
                    if tt > T:
                        continue
                    try:
                        r = measure_case(R, T, "A", args.peak_tflops, args.peak_gbs, kc, tt)
                    except RuntimeError as err:  # does not fit in shared memory
                        print(f"R={R} T={T} kc={kc} tt={tt}: skipped ({err})")
                        continue
                    rows.append(r)
                    print(f"R={R:2d} T={T:2d} kc={kc:2d} tt={tt:2d} -> used {r['kc_tt']}  "
                          f"fused {r['fused_kernel_us']:8.2f} µs  stream {r['stream_us_per_call']:8.2f} µs  "
                          f"{r['pct_fp16_tc_peak']:5.1f}% TC peak")
        result["sweep"] = rows
    else:
        result["floors"] = floors(ext)
        print("floors:", json.dumps({k: round(v, 2) for k, v in result["floors"].items()}))
        rows = []
        for R, T in CASES:
            for design in ("ref", "dense", "A", "B") + (("compile",) if args.with_compile else ()):
                try:
                    r = measure_case(R, T, design, args.peak_tflops, args.peak_gbs)
                except RuntimeError as err:
                    print(f"R={R:2d} T={T:2d} {design:3s} skipped: {err}")
                    continue
                rows.append(r)
                print(f"R={R:2d} T={T:2d} {design:3s} stream {r['stream_us_per_call']:8.2f} µs/call   "
                      f"kernels {r['kernel_us_per_call']:8.2f} µs ({len(r['kernels'])} kinds)   "
                      f"fused {r['fused_kernel_us']:8.2f} µs  {r['pct_fp16_tc_peak']:5.1f}% TC  "
                      f"{r['pct_hbm_peak']:5.2f}% HBM")
        result["cases"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
