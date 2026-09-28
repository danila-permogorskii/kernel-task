"""A tiny program for Nsight Compute on a Qwen3.8-27B shape: prepare once, call a few times.

    ncu --set full -k regex:tr_ring_fused -c 1 -o out \\
        python tools/qwen_ncu_target.py --shape mlp_gate_up --rank 8 --tokens 1
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_shapes import SHAPES  # noqa: E402
from qwen_bench import best_tiling  # noqa: E402

from factorized_inference import TRSpec, make_cores  # noqa: E402
from factorized_inference.tr_kernel import PreparedTRKernel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shape", default="mlp_gate_up", choices=list(SHAPES))
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--tokens", type=int, default=1)
ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
ap.add_argument("--tilings", type=Path, help="qwen_sweep.py JSON: profile its fastest tiling")
args = ap.parse_args()
forced = best_tiling(args.tilings, args.shape, args.rank, args.tokens) if args.tilings else None
if forced:
    os.environ.update(TR_KC=str(forced[0]), TR_TT=str(forced[1]), TR_QC=str(forced[2]))

ins, outs, _ = SHAPES[args.shape]
spec = TRSpec(ins, outs, args.rank)
dtype = getattr(torch, args.dtype)
cores = make_cores(spec, device="cuda", dtype=dtype, seed=0)
x = torch.randn(args.tokens, spec.in_features, device="cuda", dtype=dtype)
run = PreparedTRKernel(cores, spec)
for _ in range(3):
    run(x)
torch.cuda.synchronize()
print("tiling (kc, tt, qc):", run.tiling(args.tokens))
