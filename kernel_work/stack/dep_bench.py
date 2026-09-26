"""Mode 9 (dependency counters, timing proxy) against the barriers, on the same layers.

    python kernel_work/stack/dep_bench.py --out results/h100/stack/dep_r8.json

Rows: mode 3 (grid.sync), mode 6 (monotonic barrier), mode 9 + kDepAll (the counters, but
every unit waits for all groups: must match a barrier, the control), mode 9 (each unit waits
for one group = 1/nqc of the previous layer). Mode 9's output is wrong by construction (see
tr_stack.cu, "mode 9"); only the time per layer counts. Two interleaved rounds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stack_bench import Stack, errors, make_layers, reference_chain, slope, stream_us  # noqa: E402

K_DEP_ALL = 128
ROWS = (("grid.sync (mode 3)", 3, 0), ("monotonic barrier (mode 6)", 6, 0),
        ("counters, wait all groups (control)", 9, K_DEP_ALL),
        ("counters, wait one group (proxy)", 9, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--tiling", default="2,4,2,6")
    ap.add_argument("--variant", default="v3a")
    ap.add_argument("--layers", default="8,32,128")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    R, tiling = a.rank, tuple(int(v) for v in a.tiling.split(","))
    Ls = tuple(int(v) for v in a.layers.split(","))
    x = torch.randn(1, 1920, device="cuda", dtype=torch.float16)
    layer_sets = {L: make_layers(L, R) for L in Ls}
    stacks = {L: Stack(layer_sets[L], R, tiling, variant=a.variant) for L in Ls}
    ref = reference_chain(x, layer_sets[Ls[0]])
    info = stacks[Ls[0]].info(1)
    print(f"R={R} tiling {tiling} {a.variant}: units up {info['units_up']} down "
          f"{info['units_down']}, resident blocks {info['resident_blocks']}", flush=True)
    res = {name: [] for name, _, _ in ROWS}
    errs = {}
    for rnd in (1, 2):
        for name, mode, flags in ROWS:
            errs[name] = errors(stacks[Ls[0]](x, mode, flags=flags), ref)["rel_l2"]
            pts = [(L, stream_us(lambda: stacks[L](x, mode, flags=flags))) for L in Ls]
            fit = slope([{"layers": L, "u": u} for L, u in pts], "u")
            res[name].append({"points": pts, **fit})
            print(f"  round {rnd}  {name:38s} {fit['us_per_layer']:6.2f} µs/layer   "
                  f"rel L2 {errs[name]:.1e}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "rank": R,
                                     "tiling": list(tiling), "variant": a.variant,
                                     "rows": res, "rel_l2": errs}, indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
