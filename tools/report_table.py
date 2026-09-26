"""Build the report's results table straight from the harness JSON.

    python tools/report_table.py > results/h100/report_table.md

Rows: all five required cases x (dense, reference, ours A, ours B). Dense and reference come
from the design-A run (the harness runs them in every invocation; the B run's copies agree
within noise).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "h100"
MiB = 1024 * 1024


def load(design, rank):
    return json.load(open(ROOT / design / f"rank{rank}.json"))["cases"]


def row(rank, tokens, label, m):
    mem = m["memory"]
    return (f"| {rank} | {tokens} | {label} | {m['host_synchronized_median_ms']:.4f} | "
            f"{m['cuda_event_stream_median_ms']:.4f} | "
            f"{mem['resident_after_warmup_allocated_bytes'] / MiB:.2f} | "
            f"{mem['steady_peak_allocated_bytes'] / MiB:.2f} | "
            f"{mem['incremental_workspace_and_output_bytes'] / MiB:.3f} | "
            f"{m['preparation_ms']:.1f} | {m['first_call_ms']:.2f} | "
            f"{m['correctness']['max_absolute_error']:.4f} |")


print("| Rank | Tokens | Method | Host median ms | CUDA stream median ms | Resident MiB | "
      "Steady allocated peak MiB | Incremental workspace/output MiB | Preparation ms | "
      "First call ms | Max abs error |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for rank in (8, 16):
    for ca, cb in zip(load("A", rank), load("B", rank)):
        t = ca["tokens"]
        print(row(rank, t, "dense", ca["methods"]["dense"]))
        print(row(rank, t, "factorized_reference", ca["methods"]["factorized_reference"]))
        print(row(rank, t, "**ours, design A**", ca["methods"]["factorized_optimized"]))
        print(row(rank, t, "ours, design B", cb["methods"]["factorized_optimized"]))
