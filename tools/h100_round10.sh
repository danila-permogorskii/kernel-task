#!/usr/bin/env bash
# Round 10 (H100): V3 without S1 zeroing + B in registers; cp.async cores (v3a).
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/stack
B=kernel_work/stack/stack_bench.py
MHZ=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
for v in v3 v3a; do
  echo "== check $v"
  STACK_VARIANT=$v python $B --check 2>&1 | grep -E "FAIL|ALL OK|SOME|rror|Traceback" | head -8
done
echo "== compare"
COMPARE_VARIANTS=v3,v3a python $B --compare --rank 8  --tiling 2,4,2,6 --out "$OUT/compare10_r8.json"
COMPARE_VARIANTS=v3,v3a python $B --compare --rank 16 --tiling 4,4,3,6 --out "$OUT/compare10_r16.json"
echo "== v3a timeline"
STACK_TIMING_VARIANT=v3atiming python $B --round2 --rank 8 --tokens 1 --tiling "2,4,2,6" --sm-mhz "$MHZ" \
  --out "$OUT/round10_v3a_r8.json" 2>&1 | grep -E "mode [56] (up|down)"
echo done
