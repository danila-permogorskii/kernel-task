#!/usr/bin/env bash
# Round 11 (H100): mode 8 = cp.async prefetch issued before grid.sync; final A/B table.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/stack
B=kernel_work/stack/stack_bench.py
echo "== check v3a (modes 0,3,4,5,6)"
STACK_VARIANT=v3a python $B --check 2>&1 | grep -E "FAIL|ALL OK|SOME|rror|Traceback" | head -8
python - <<'PY'
import sys, torch
sys.path.insert(0, "kernel_work/stack")
from stack_bench import Stack, make_layers, reference_chain, errors
for R, til in ((8, (2, 4, 2, 6)), (16, (4, 4, 3, 6))):
    for L in (1, 2, 3, 8):
        layers = make_layers(L, R)
        st = Stack(layers, R, til, variant="v3a")
        x = torch.randn(1, 1920, device="cuda", dtype=torch.float16)
        e = errors(st(x, 8), reference_chain(x, layers))["rel_l2"]
        e2 = errors(st(x, 8), reference_chain(x, layers))["rel_l2"]
        print(f"mode 8 check R={R} L={L}: rel L2 {e:.1e} / {e2:.1e}  {'ok' if max(e, e2) < 1e-2 else 'FAIL'}")
PY
echo "== compare modes 3, 6, 8"
COMPARE_VARIANTS=v3a COMPARE_MODES=3,6,8 python $B --compare --rank 8  --tiling 2,4,2,6 --out "$OUT/compare11_r8.json"
COMPARE_VARIANTS=v3,v3a COMPARE_MODES=3,6,8 python $B --compare --rank 16 --tiling 4,4,3,6 --out "$OUT/compare11_r16.json"
echo done
