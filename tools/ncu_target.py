"""A tiny program for Nsight Compute: prepare once, call the kernel a few times.

    ncu --set full -k regex:tr_ring_fused -c 1 -o out python tools/ncu_target.py --rank 16 --tokens 32
"""
import argparse

import torch

from factorized_inference import TRSpec, make_cores
from factorized_inference.tr_kernel import PreparedTRKernel

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--tokens", type=int, default=32)
args = ap.parse_args()

spec = TRSpec(rank=args.rank)
cores = make_cores(spec, device="cuda", dtype=torch.float16, seed=0)
x = torch.randn(args.tokens, spec.in_features, device="cuda", dtype=torch.float16)
run = PreparedTRKernel(cores, spec)
for _ in range(3):
    run(x)
torch.cuda.synchronize()
print("tiling (kc, tt):", run.tiling(args.tokens))
