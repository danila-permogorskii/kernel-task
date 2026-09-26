"""Diagnostic experiments on the fused kernel: where does its time go?

  --threads   rebuild with 128 / 256 / 512 / 1024 threads per block, time each
  --stages    rebuild with one part removed (load, stage 1, 2, 3, atomics); the results are
              wrong, only the time matters: time(full) - time(without X) ~ cost of X

Each variant is a separate extension build in build/experiments/<name> (~1 min each).
Design A, the harness's default tiling unless TR_KC / TR_TT are set.

    python tools/h100_experiments.py --threads --stages --out results/h100/experiments.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from torch.utils import cpp_extension

import factorized_inference.tr_kernel as tk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_kernels import CASES, measure_case  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = (REPO / "src/factorized_inference/csrc/tr_ring.cu").read_text()

# text markers in tr_ring.cu that bound each removable part
PARTS = {
    "load":     ("  // ---- load: global -> shared", "  const int M = d.tt * d.P;"),
    "stage1":   ("    // ---- stage 1:", "    // ---- stage 2:"),
    "stage2":   ("    // ---- stage 2:", "    // ---- stage 3:"),
    # stop before the loop's last __syncthreads so the k loop's closing brace survives
    "stage3":   ("    // ---- stage 3:", "    __syncthreads();\n  }\n\n  // ---- add this block's Y"),
    "atomics":  ("  // ---- add this block's Y", "  // ---- design B:"),
}


def without(part: str) -> str:
    start, end = PARTS[part]
    i, j = SRC.index(start), SRC.index(end)
    keep = "  const int RRr = 0; (void)RRr;\n" if part == "load" else ""
    return SRC[:i] + "#if 0\n" + SRC[i:j] + "#endif\n" + keep + "  __syncthreads();\n" + SRC[j:]


def build(name: str, source: str, flags=()):
    d = REPO / "build" / "experiments" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "tr_ring.cu").write_text(source)
    tk._ext = cpp_extension.load(name=f"tr_exp_{name}", sources=[str(d / "tr_ring.cu")],
                                 build_directory=str(d),
                                 extra_cuda_cflags=["-O3", "-lineinfo", *flags])


def time_all(label: str) -> dict:
    row = {}
    for R, T in CASES:
        try:
            measure_case(R, T, "A", 989.4, 3350)            # warm-up (clocks, lazy init)
            r = measure_case(R, T, "A", 989.4, 3350)
        except RuntimeError:  # does not fit this GPU (laptop smoke test)
            continue
        row[f"R{R}_T{T}"] = {"fused_us": r["fused_kernel_us"], "kc_tt": r["kc_tt"]}
    print(f"{label:16s}", "  ".join(f"{k}:{v['fused_us']:8.2f}" for k, v in row.items()),
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", action="store_true")
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tk.load_extension()  # sets CUDA_HOME for the builds below
    out = {}
    if args.threads:
        for th in (128, 256, 512, 1024):
            build(f"threads{th}", SRC, [f"-DTR_THREADS={th}"])
            out[f"threads_{th}"] = time_all(f"threads {th}")
    if args.stages:
        build("full", SRC)
        out["full"] = time_all("full")
        for part in PARTS:
            build(f"without_{part}", without(part))
            out[f"without_{part}"] = time_all(f"without {part}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
