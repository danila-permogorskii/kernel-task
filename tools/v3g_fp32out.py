#!/usr/bin/env python3
"""Does fusing the FP32 -> BF16 tail into the next op pay? One MLP block of Qwen3.8-27B, ring V3G:

  tail   gate, up, down with the design-B tail (BF16 out), then torch silu * mul and the residual add
  fused  gate, up, down with FP32 output (no tail), consume_silu_mul and consume_add read the FP32
         workspace, do the op and clear it (csrc/v3g/bind.cu)
Both checked against each other and against an FP32 reference of the same block.

    python tools/v3g_fp32out.py --out results/h100/qwen/v3g_fp32out.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_kernels import stream_us  # noqa: E402
from qwen_shapes import SHAPES  # noqa: E402

from factorized_inference import TRSpec, make_cores, materialize_dense_weight  # noqa: E402
from factorized_inference.tr_kernel import PreparedTRKernel, load_v3g  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
ext = load_v3g()
rows = []
for R in (8, 16):
    mk = lambda s, seed: (TRSpec(*SHAPES[s][:2], R), make_cores(  # noqa: E731
        TRSpec(*SHAPES[s][:2], R), device="cuda", dtype=torch.bfloat16, seed=seed))
    (sg, cg), (su, cu), (sd, cd) = mk("mlp_gate_up", 1), mk("mlp_gate_up", 2), mk("mlp_down", 3)
    gate, up, down = PreparedTRKernel(cg, sg), PreparedTRKernel(cu, su), PreparedTRKernel(cd, sd)
    Wg, Wu, Wd = (materialize_dense_weight(tuple(c.float() for c in cs), s)
                  for cs, s in ((cg, sg), (cu, su), (cd, sd)))
    for T in (1, 8, 32):
        h = torch.randn(T, 5120, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(T, 5120, device="cuda", dtype=torch.bfloat16)

        def tail():
            return h + down(F.silu(gate(x)) * up(x))

        def fused():
            act = ext.consume_silu_mul(gate.accumulate(x), up.accumulate(x))
            return ext.consume_add(h, down.accumulate(act))

        xf = x.float()
        ref = h.float() + (F.silu(xf @ Wg.T) * (xf @ Wu.T)) @ Wd.T
        y_t, y_f = tail().float(), fused().float()
        y_f2 = fused().float()  # a second call: the consumers must have cleared the workspace
        rel = lambda u: (torch.linalg.vector_norm(u - ref) / torch.linalg.vector_norm(ref)).item()  # noqa: E731
        r = {"rank": R, "tokens": T, "tail_us": stream_us(tail), "fused_us": stream_us(fused),
             "rel_err_tail": rel(y_t), "rel_err_fused": rel(y_f), "rel_err_fused_2nd": rel(y_f2)}
        rows.append(r)
        print(f"R={R:2d} T={T:2d}: MLP block tail {r['tail_us']:7.1f} us  fused {r['fused_us']:7.1f} us  "
              f"(x{r['tail_us'] / r['fused_us']:.2f})  err tail {r['rel_err_tail']:.1e} fused "
              f"{r['rel_err_fused']:.1e} / 2nd call {r['rel_err_fused_2nd']:.1e}", flush=True)
a.out.write_text(json.dumps(rows, indent=1) + "\n")
print("wrote", a.out)
