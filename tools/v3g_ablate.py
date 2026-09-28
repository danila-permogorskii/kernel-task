#!/usr/bin/env python3
"""Where does a V3G call spend its time? The tuned tiling, timed three ways (env TR_V3G_ABLATE):
full kernel; without the red.add of partial Y into the FP32 workspace; without red.add and the
design-B tail (compute only). Outputs of the ablated runs are wrong on purpose: timing only.

    python tools/v3g_ablate.py --out results/h100/qwen/v3g_ablate.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_kernels import stream_us  # noqa: E402
from qwen_shapes import SHAPES  # noqa: E402

from factorized_inference import TRSpec, make_cores  # noqa: E402
from factorized_inference.tr_kernel import PreparedTRKernel, load_v3g  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
load_v3g()
rows = []
for shape in ("mlp_gate_up", "mlp_down", "gdn_qkvz"):
    ins, outs, _ = SHAPES[shape]
    for R in (8, 16):
        spec = TRSpec(ins, outs, R)
        run = PreparedTRKernel(make_cores(spec, device="cuda", dtype=torch.bfloat16), spec)
        for T in (1, 8):
            x = torch.randn(T, spec.in_features, device="cuda", dtype=torch.bfloat16)
            r = {"shape": shape, "rank": R, "tokens": T, "tiling": run.v3g_tiling(T)}
            for mode, key in (("0", "full_us"), ("1", "no_red_us"), ("2", "compute_only_us")):
                os.environ["TR_V3G_ABLATE"] = mode
                r[key] = stream_us(lambda: run(x))
            os.environ["TR_V3G_ABLATE"] = "0"
            run(x)  # leaves the workspace consistent: ablated runs skipped the clean-up
            rows.append(r)
            print(f"{shape:12s} R={R:2d} T={T}: full {r['full_us']:6.1f}  no red.add "
                  f"{r['no_red_us']:6.1f}  compute only {r['compute_only_us']:6.1f} us  {r['tiling']}",
                  flush=True)
os.environ.pop("TR_V3G_ABLATE", None)
a.out.write_text(json.dumps(rows, indent=1) + "\n")
print("wrote", a.out)
