#!/usr/bin/env bash
# Path 1 of WEIGHT_STATIONARY.md §7: core prefetch during the barrier (mode 4).
# Run after tools/h100_setup.sh:   bash tools/remote.sh run bash tools/h100_stack_pf.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/stack
mkdir -p "$OUT"
B=kernel_work/stack/stack_bench.py
best() { python -c "import json,sys; print(','.join(map(str, json.load(open(sys.argv[1]))['sweeps'][0]['best'])))" "$1"; }

echo "== 1. correctness, modes 0 and 4"
python $B --check 2>&1 | tee "$OUT/check_pf.log" | grep -E "FAIL|ALL OK|SOME"

echo "== 2. tiling search for mode 4 at L = 32, T = 1"
python $B --sweep --sweep-mode 4 --rank 8  --tokens 1 --out "$OUT/sweep_pf_r8.json"
python $B --sweep --sweep-mode 4 --rank 16 --tokens 1 --out "$OUT/sweep_pf_r16.json"
T8=$(best "$OUT/sweep_pf_r8.json"); T16=$(best "$OUT/sweep_pf_r16.json")
echo "tilings (mode 4): R8 $T8   R16 $T16"

echo "== 3. L = 8, 32, 128: mode 0 / grid.sync / prefetch, same tiling"
python $B --rank 8  --layers 8,32,128 --tokens 1 --tiling "$T8"  --skip-baselines --out "$OUT/r8_pf.json"  2>&1 | tee "$OUT/r8_pf.log"
python $B --rank 16 --layers 8,32,128 --tokens 1 --tiling "$T16" --skip-baselines --out "$OUT/r16_pf.json" 2>&1 | tee "$OUT/r16_pf.log"
echo "done: $OUT"
