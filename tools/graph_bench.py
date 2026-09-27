"""Optional measurements beyond the README's required (uncaptured) runs: CUDA graphs, many
distinct layers, a cold L2, and where the baselines' resident memory comes from.

    python tools/graph_bench.py --part cublas --out results/h100/graphs/cublas.json   # run FIRST, fresh process
    python tools/graph_bench.py --part single --out results/h100/graphs/single.json
    python tools/graph_bench.py --part chain  --out results/h100/graphs/chain.json
    python tools/graph_bench.py --part cold   --out results/h100/graphs/cold.json

Parts:
  single  every required case, every method, under the same two rules:
            eager  G back-to-back calls from Python, CUDA events (like the harness)
            graph  the same G calls captured in ONE CUDA graph and replayed (no host path)
          Output of the captured calls is checked against the FP64 dense oracle.
  chain   L distinct instances of the assignment's operator (different weights each), applied
          one after another in one stream, eager and captured in one CUDA graph. Per layer µs.
          With L >= 8 the dense weights (11 MB each) no longer fit in the 50 MB L2, as in a
          real model; the ring's cores (89-348 KB each) still do. Layers all read the same x:
          graph nodes captured from one stream run strictly in order, so the time per layer
          is the same as for a data-dependent chain.
  cold    one call with the L2 flushed before it (a 256 MB read-modify-write), GPU time of
          the method's own kernels only (profiler), next to the same without the flush.
  cublas  resident PyTorch allocations around the first cuBLAS call and after
          torch._C._cuda_clearCublasWorkspaces(): is the ~32 MiB of the dense and reference
          rows in results/h100 the cuBLAS workspace?

Methods: dense (F.linear on the materialised W, baseline only), ref (tr_forward_reference),
B and A (our kernel, prepare_optimized with TR_DESIGN=B / A).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
from torch.autograd import DeviceType

from factorized_inference import (
    TRSpec, dense_forward, make_cores, materialize_dense_weight, tr_forward_reference,
)
from factorized_inference.submission import prepare_optimized

CASES = [(8, 1), (8, 8), (8, 32), (16, 1), (16, 32)]
METHODS = ("dense", "ref", "B", "A")
H100_SXM_HBM_GBS = 3350.0
MIB = 1024 * 1024


def make_fn(method: str, cores, spec: TRSpec):
    """Return (fn(x) -> y, W or None). W exists only for the dense baseline."""
    if method == "dense":
        W = materialize_dense_weight(cores, spec)
        return (lambda x: dense_forward(x, W)), W
    if method == "ref":
        return (lambda x: tr_forward_reference(x, cores, spec)), None
    os.environ["TR_DESIGN"] = method
    return prepare_optimized(cores, spec), None


def oracle(x, cores, spec):
    W = materialize_dense_weight([c.double() for c in cores], spec)
    return x.double() @ W.T


def err(y, ref) -> dict:
    d = (y.double() - ref).abs()
    return {"max_abs": d.max().item(),
            "rel_l2": (d.norm() / ref.norm().clamp_min(1e-12)).item(),
            "within_tol": bool((d <= 0.02 + 0.02 * ref.abs()).all())}  # FP16 atol = rtol = 0.02


def events_us(run, n_inner: int, reps: int = 7) -> float:
    """CUDA events around `run()`, which performs n_inner calls; µs per call, median of reps."""
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        run()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e) * 1000 / n_inner)
    return statistics.median(out)


def capture(body):
    """Warm `body` up on a side stream (PyTorch's recipe), then capture it into one graph.
    Returns (graph, whatever the captured body returned: static outputs)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = body()
    torch.cuda.synchronize()
    return g, out


# ------------------------------------------------------------------------------ single
def part_single(G: int = 20) -> list:
    rows = []
    for R, T in CASES:
        spec = TRSpec(rank=R)
        cores = make_cores(spec, device="cuda", dtype=torch.float16, seed=0)
        x = torch.randn(T, spec.in_features, device="cuda", dtype=torch.float16)
        ref = oracle(x, cores, spec)
        for m in METHODS:
            fn, W = make_fn(m, cores, spec)

            def eager():
                for _ in range(G):
                    fn(x)

            def body():
                y = None
                for _ in range(G):
                    y = fn(x)
                return y

            eager_us = events_us(eager, G)
            g, y_static = capture(body)
            g.replay()
            torch.cuda.synchronize()
            e_graph = err(y_static, ref)
            graph_us = events_us(g.replay, G)
            ok = e_graph["within_tol"]
            rows.append({"rank": R, "tokens": T, "method": m, "calls_per_graph": G,
                         "eager_us_per_call": eager_us, "graph_us_per_call": graph_us,
                         "graph_output_error": e_graph, "graph_output_ok": ok})
            print(f"single R={R:2d} T={T:2d} {m:5s}  eager {eager_us:8.2f}  graph {graph_us:8.2f} µs/call"
                  f"   graph out max|err| {e_graph['max_abs']:.2e} {'ok' if ok else 'FAIL'}", flush=True)
            del g, y_static, fn, W
            torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------------------- chain
def part_chain(ranks=(8, 16), tokens=(1, 8), layers=(1, 8, 32, 128), methods=("dense", "B", "A", "ref"),
               peak_gbs: float = H100_SXM_HBM_GBS) -> list:
    rows = []
    for R in ranks:
        spec = TRSpec(rank=R)
        Lmax = max(layers)
        all_cores = [make_cores(spec, device="cuda", dtype=torch.float16, seed=1000 + l) for l in range(Lmax)]
        core_bytes = sum(c.numel() * 2 for c in all_cores[0])
        for m in methods:
            fns, Ws = [], []
            for cores in all_cores:
                fn, W = make_fn(m, cores, spec)
                fns.append(fn)
                Ws.append(W)
            for T in tokens:
                x = torch.randn(T, spec.in_features, device="cuda", dtype=torch.float16)
                for L in layers:
                    sub = fns[:L]

                    def eager():
                        for f in sub:
                            f(x)

                    def body():
                        return [f(x) for f in sub]

                    eager_us = events_us(eager, L, reps=5)
                    g, ys = capture(body)
                    g.replay()
                    torch.cuda.synchronize()
                    # check the first and last captured layer against the oracle
                    e0 = err(ys[0], oracle(x, all_cores[0], spec))
                    e1 = err(ys[-1], oracle(x, all_cores[L - 1], spec))
                    graph_us = events_us(g.replay, L, reps=5)
                    row = {"rank": R, "tokens": T, "layers": L, "method": m,
                           "eager_us_per_layer": eager_us, "graph_us_per_layer": graph_us,
                           "err_first": e0, "err_last": e1,
                           "weight_bytes_per_layer": (spec.out_features * spec.in_features * 2
                                                      if m == "dense" else core_bytes)}
                    if m == "dense":
                        row["hbm_floor_us_per_layer"] = row["weight_bytes_per_layer"] / (peak_gbs * 1e9) * 1e6
                    rows.append(row)
                    print(f"chain R={R:2d} T={T:2d} L={L:4d} {m:5s}  eager {eager_us:8.2f}  graph {graph_us:8.2f}"
                          f" µs/layer   max|err| {max(e0['max_abs'], e1['max_abs']):.2e}"
                          f" {'ok' if e0['within_tol'] and e1['within_tol'] else 'FAIL'}", flush=True)
                    del g, ys
                    torch.cuda.empty_cache()
            del fns, Ws
            torch.cuda.empty_cache()
    return rows


# -------------------------------------------------------------------------------- cold
def kernel_names_time(fn, calls: int, exclude: set[str]) -> tuple[float, dict]:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    per = {}
    for ev in prof.events():
        if ev.device_type == DeviceType.CUDA and ev.name not in exclude:
            per[ev.name] = per.get(ev.name, 0.0) + ev.time_range.elapsed_us() / calls
    return sum(per.values()), per


def part_cold(calls: int = 30, flush_mb: int = 256) -> list:
    flush = torch.ones(flush_mb * MIB // 4, device="cuda", dtype=torch.float32)
    do_flush = lambda: flush.mul_(1.0)  # noqa: E731  read + write 256 MB: evicts the 50 MB L2
    for _ in range(3):
        do_flush()
    torch.cuda.synchronize()
    _, flush_names = kernel_names_time(do_flush, 3, set())
    exclude = set(flush_names)
    print("cold: flush kernel(s) excluded:", sorted(exclude))
    rows = []
    for R, T in CASES:
        spec = TRSpec(rank=R)
        cores = make_cores(spec, device="cuda", dtype=torch.float16, seed=0)
        x = torch.randn(T, spec.in_features, device="cuda", dtype=torch.float16)
        for m in ("dense", "B", "ref"):
            fn, W = make_fn(m, cores, spec)
            for _ in range(3):
                fn(x)
            torch.cuda.synchronize()
            warm, warm_k = kernel_names_time(lambda: fn(x), calls, exclude)

            def cold_call():
                do_flush()
                fn(x)

            cold, cold_k = kernel_names_time(cold_call, calls, exclude)
            rows.append({"rank": R, "tokens": T, "method": m, "warm_kernel_us": warm, "cold_kernel_us": cold,
                         "warm_kernels": warm_k, "cold_kernels": cold_k})
            print(f"cold R={R:2d} T={T:2d} {m:5s}  kernels warm L2 {warm:8.2f}  cold L2 {cold:8.2f} µs/call", flush=True)
            del fn, W
            torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------------------ cublas
def part_cublas() -> dict:
    """Run in a fresh process: nothing may have called cuBLAS before."""
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    alloc = lambda: torch.cuda.memory_allocated() / MIB  # noqa: E731
    out = {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
           "context_baseline_mib": alloc()}
    spec = TRSpec(rank=8)
    cores = make_cores(spec, device="cuda", dtype=torch.float16, seed=0)
    x = torch.randn(1, spec.in_features, device="cuda", dtype=torch.float16)

    # our kernel first: it must not add a workspace
    run = prepare_optimized(cores, spec)
    before = alloc()
    y = run(x)
    torch.cuda.synchronize()
    out["ours_B_first_call_growth_mib"] = alloc() - before - y.numel() * 2 / MIB
    del y

    clear = torch._C._cuda_clearCublasWorkspaces
    W = materialize_dense_weight(cores, spec)  # itself calls cuBLAS: drop that workspace first
    torch.cuda.synchronize()
    clear()
    before = alloc()
    y = dense_forward(x, W)
    torch.cuda.synchronize()
    out["dense_first_call_growth_mib"] = alloc() - before - y.numel() * 2 / MIB
    del y
    before = alloc()
    clear()
    torch.cuda.synchronize()
    out["freed_by_clearCublasWorkspaces_mib"] = before - alloc()
    before = alloc()
    y = tr_forward_reference(x, cores, spec)
    torch.cuda.synchronize()
    out["reference_first_call_growth_mib"] = alloc() - before - y.numel() * 2 / MIB
    for k, v in out.items():
        print(f"cublas {k}: {v if not isinstance(v, float) else round(v, 3)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("single", "chain", "cold", "cublas"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quick", action="store_true", help="small sizes, for a laptop correctness run")
    args = ap.parse_args()
    global CASES
    torch.backends.cuda.matmul.allow_tf32 = False  # as the harness
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    result = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "part": args.part}
    with torch.inference_mode():
        if args.part == "single":
            if args.quick:
                CASES = [(8, 1), (8, 8)]
            result["rows"] = part_single()
        elif args.part == "chain":
            kw = dict(ranks=(8,), tokens=(1,), layers=(1, 4)) if args.quick else {}
            result["rows"] = part_chain(**kw)
        elif args.part == "cold":
            if args.quick:
                CASES = [(8, 1)]
            result["rows"] = part_cold()
        else:
            result["cublas"] = part_cublas()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
