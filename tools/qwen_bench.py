#!/usr/bin/env python3
"""The harness protocol on Qwen3.8-27B layer shapes, in BF16 (or FP16).

Shapes and modes: tools/qwen_shapes.py. Weights are random (make_cores), so this measures speed,
memory and numerics on real sizes, not the quality of a compressed model.

Same protocol as benchmarks/benchmark.py, whose helpers it imports: every method / shape / rank /
token case runs in a fresh child process; 20 warm-up calls, 100 host-synchronised calls, 5 CUDA
event blocks of 20 calls; FP32 dense oracle built after the measurement. Differences, all
deliberate:
  - --dtype bfloat16 is allowed; tolerance for FP16 and BF16 is the harness's FP16 one (0.02)
  - allow_bf16_reduced_precision_reduction = False (the harness sets the FP16 flag the same way)
  - correctness is recorded, not asserted: a failing case is flagged and still timed
  - the optimized method records its tiling (kc, tt, qc), packed-operand bytes and whether the
    kernel actually ran (it falls back to the reference, with a warning, if nothing fits)
  - dense does not depend on R, so it runs once per shape and token count (with R = the first rank)
  - the JSON is rewritten after every case, so a lost instance loses at most one case
  - --tilings SWEEP.json (tools/qwen_sweep.py) forces the fastest measured (kc, tt, qc) per
    (shape, R, T) through TR_KC / TR_TT / TR_QC; cases the sweep did not cover use the chooser

    python tools/qwen_bench.py --device cuda:0 --dtype bfloat16 --ranks 8,16 \\
        --tokens 1,8,32,128 --output results/a100/qwen/bf16.json --profile-dir traces/a100/qwen
    python tools/qwen_report.py results/a100/qwen/bf16.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tools"))
from factorized_inference import (  # noqa: E402
    TRSpec, dense_forward, make_cores, materialize_dense_weight,
    prepare_optimized, tr_forward_reference,
)
from benchmark import environment, errors, synchronize, timed_call  # noqa: E402
from qwen_shapes import SHAPES, check_against_config  # noqa: E402

METHODS = ("dense", "factorized_reference", "factorized_optimized")


def best_tiling(sweep: Path, shape: str, rank: int, tokens: int):
    """Fastest correct tiling that tools/qwen_sweep.py measured for this case, or None."""
    rows = [r for r in json.loads(sweep.read_text())["rows"]
            if (r["shape"], r["rank"], r["tokens"]) == (shape, rank, tokens)
            and r["variant"] in ("chooser", "forced") and r.get("tiling")
            and r.get("rel_l2_vs_dense", 0.0) < 1e-2]
    return tuple(min(rows, key=lambda r: r["stream_us"])["tiling"]) if rows else None


def ring_flop(spec: TRSpec) -> int:
    """Useful FLOP per token of the three-stage contraction (no padding)."""
    (i, j, k), (p, q, r), R = spec.input_modes, spec.output_modes, spec.rank
    return 2 * (i * j * k * p * R * R + j * k * p * q * R ** 3 + p * q * r * k * R * R)


@torch.inference_mode()
def worker(a):
    device = torch.device(a.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dtype = getattr(torch, a.dtype)
    ins, outs, _ = SHAPES[a.shape]
    spec = TRSpec(ins, outs, a.rank)
    tokens = a.tokens

    probe = torch.empty(1, device=device)
    del probe
    synchronize(device)
    torch.cuda.empty_cache()
    baseline_bytes = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    cores = make_cores(spec, device=device, dtype=dtype, seed=a.seed)
    generator = torch.Generator(device=device).manual_seed(a.seed + tokens)
    x = torch.randn(tokens, spec.in_features, generator=generator, device=device, dtype=dtype)
    kernel = None
    if a.worker == "dense":
        weight, preparation_ms = timed_call(lambda: materialize_dense_weight(cores, spec), device)
        run = lambda value: dense_forward(value, weight)
        representation_bytes = weight.numel() * weight.element_size()
        del cores
    elif a.worker == "factorized_reference":
        run = lambda value: tr_forward_reference(value, cores, spec)
        preparation_ms = 0.0
        representation_bytes = sum(c.numel() * c.element_size() for c in cores)
    else:
        forced = best_tiling(a.tilings, a.shape, a.rank, tokens) if a.tilings else None
        if forced:  # read by choose_tiling at the first call
            os.environ.update(TR_KC=str(forced[0]), TR_TT=str(forced[1]), TR_QC=str(forced[2]))
        run, preparation_ms = timed_call(lambda: prepare_optimized(cores, spec), device)
        representation_bytes = sum(c.numel() * c.element_size() for c in cores)
        packed = [getattr(run, n, None) for n in ("A1", "B2", "C3")]
        kernel = {"class": type(run).__name__,
                  "packed_operand_bytes": sum(t.numel() * t.element_size()
                                              for t in packed if t is not None),
                  "tiling_source": "sweep" if forced else "chooser"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first_output, first_call_ms = timed_call(lambda: run(x), device)
    del first_output
    if kernel is not None:
        tiling = run.tiling(tokens) if hasattr(run, "tiling") else None
        kernel["tiling_kc_tt_qc"] = tiling
        kernel["fell_back_to_reference"] = (tiling is None or any(
            "using the reference" in str(w.message) for w in caught))
    for _ in range(a.warmup):
        run(x)
    synchronize(device)
    gc.collect()
    input_bytes = x.numel() * x.element_size()
    setup_peak = torch.cuda.max_memory_allocated(device)
    resident = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    samples = []
    for _ in range(a.iterations):
        result, elapsed = timed_call(lambda: run(x), device)
        del result
        samples.append(elapsed)
    steady_peak = torch.cuda.max_memory_allocated(device)
    memory = {
        "context_baseline_allocated_bytes": baseline_bytes,
        "input_logical_bytes": input_bytes,
        "representation_logical_bytes": representation_bytes,
        "resident_after_warmup_allocated_bytes": resident,
        "extra_resident_allocated_bytes": max(0, resident - baseline_bytes - input_bytes
                                              - representation_bytes),
        "preparation_and_warmup_peak_allocated_bytes": setup_peak,
        "steady_peak_allocated_bytes": steady_peak,
        "incremental_workspace_and_output_bytes": steady_peak - resident,
        "steady_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }

    event_samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        synchronize(device)
        start.record()
        for _ in range(a.event_block):
            run(x)
        end.record()
        end.synchronize()
        event_samples.append(start.elapsed_time(end) / a.event_block)

    # validation after measurement, from pristine factors, FP32 dense oracle (as the harness)
    check_cores = make_cores(spec, device=device, dtype=dtype, seed=a.seed)
    oracle_weight = materialize_dense_weight(tuple(c.float() for c in check_cores), spec)
    expected = dense_forward(x.float(), oracle_weight)
    actual = run(x)
    tol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    diff = (actual.float() - expected).abs()
    bad = (diff > tol + tol * expected.abs()).sum().item()
    correctness = {"rtol": tol, "atol": tol, "passed": bad == 0 and actual.dtype == dtype
                   and actual.shape == expected.shape and bool(torch.isfinite(actual).all()),
                   "mismatched_elements": bad, "elements": expected.numel(),
                   **errors(actual, expected)}
    del actual, expected, oracle_weight, check_cores, diff

    trace_path = None
    if a.profile_dir and tokens == 1:
        a.profile_dir.mkdir(parents=True, exist_ok=True)
        trace_path = a.profile_dir / f"{a.shape}_{a.worker}_rank{a.rank}_tokens1.json"
        acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=acts, record_shapes=True) as prof:
            for _ in range(3):
                with torch.profiler.record_function(a.worker):
                    run(x)
            synchronize(device)
        prof.export_chrome_trace(str(trace_path))

    return {
        "shape": a.shape, "input_modes": spec.input_modes, "output_modes": spec.output_modes,
        "rank": a.rank, "method": a.worker, "tokens": tokens,
        "in_features": spec.in_features, "out_features": spec.out_features,
        "dense_flop": 2 * tokens * spec.in_features * spec.out_features,
        "ring_flop": tokens * ring_flop(spec),
        "factor_parameters": spec.factor_parameters, "dense_parameters": spec.dense_parameters,
        "environment": environment(device, a),
        "preparation_ms": preparation_ms, "first_call_ms": first_call_ms,
        "host_synchronized_median_ms": statistics.median(samples),
        "host_synchronized_samples_ms": samples,
        "cuda_event_stream_median_ms": statistics.median(event_samples),
        "cuda_event_stream_samples_ms": event_samples,
        "memory": memory, "correctness": correctness, "kernel": kernel,
        "profiler_trace": str(trace_path) if trace_path else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--ranks", default="8,16")
    ap.add_argument("--tokens", default="1,8,32,128")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--event-block", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=Path("results/qwen_bench.json"))
    ap.add_argument("--profile-dir", type=Path)
    ap.add_argument("--tilings", type=Path, help="qwen_sweep.py JSON: use its fastest tilings")
    # internal: one child-process case
    ap.add_argument("--worker", choices=METHODS, help=argparse.SUPPRESS)
    ap.add_argument("--shape", help=argparse.SUPPRESS)
    ap.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--case-tokens", dest="case_tokens", type=int, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker:
        a.tokens = a.case_tokens
        a.cpu_threads = 1  # environment() reads it
        a.output.write_text(json.dumps(worker(a), indent=2) + "\n")
        return

    check_against_config()
    shapes = [s for s in a.shapes.split(",") if s]
    unknown = set(shapes) - set(SHAPES)
    if unknown:
        ap.error(f"unknown shapes {sorted(unknown)}; known: {', '.join(SHAPES)}")
    ranks = [int(v) for v in a.ranks.split(",")]
    tokens = [int(v) for v in a.tokens.split(",")]
    methods = [m for m in a.methods.split(",") if m]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "schema": "qwen_bench/1",
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
                     if k not in ("worker", "shape", "rank", "case_tokens")},
        "shapes": {s: {"input_modes": SHAPES[s][0], "output_modes": SHAPES[s][1],
                       "layers_per_forward": SHAPES[s][2]} for s in shapes},
        "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="qwen-bench-") as tmp:
        for shape in shapes:
            for rank in ranks:
                for t in tokens:
                    for method in methods:
                        if method == "dense" and rank != ranks[0]:
                            continue  # dense time does not depend on R
                        path = Path(tmp) / f"{shape}-{rank}-{t}-{method}.json"
                        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", method,
                               "--shape", shape, "--rank", str(rank), "--case-tokens", str(t),
                               "--output", str(path)]
                        for flag in ("device", "dtype", "warmup", "iterations", "event_block",
                                     "seed"):
                            cmd += ["--" + flag.replace("_", "-"), str(getattr(a, flag))]
                        if a.profile_dir:
                            cmd += ["--profile-dir", str(a.profile_dir.resolve())]
                        if a.tilings:
                            cmd += ["--tilings", str(a.tilings.resolve())]
                        proc = subprocess.run(cmd)
                        if proc.returncode != 0:
                            print(f"{shape} R={rank} T={t} {method}: CHILD FAILED "
                                  f"(exit {proc.returncode})", flush=True)
                            out["cases"].append({"shape": shape, "rank": rank, "tokens": t,
                                                 "method": method, "failed": proc.returncode})
                        else:
                            r = json.loads(path.read_text())
                            out["cases"].append(r)
                            k = r["kernel"] or {}
                            print(f"{shape:12s} R={rank:2d} T={t:3d} {method:21s} "
                                  f"stream={r['cuda_event_stream_median_ms'] * 1e3:9.1f} us  "
                                  f"host={r['host_synchronized_median_ms'] * 1e3:9.1f} us  "
                                  f"relL2={r['correctness']['relative_l2_error']:.2e} "
                                  f"{'ok' if r['correctness']['passed'] else 'FAIL'}"
                                  + (f"  tiling={k.get('tiling_kc_tt_qc')} ({k.get('tiling_source')})"
                                     + (" FALLBACK" if k.get("fell_back_to_reference") else "")
                                     if k else ""), flush=True)
                        a.output.write_text(json.dumps(out, indent=2) + "\n")
    out["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()
