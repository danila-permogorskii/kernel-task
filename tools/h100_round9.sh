#!/usr/bin/env bash
# Round 9 (H100): compile-time tiling and V3 (stages 2 -> 3 in registers, PTX mma.sync).
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/stack
B=kernel_work/stack/stack_bench.py
MHZ=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
for v in default v3; do
  echo "== check $v"
  STACK_VARIANT=$v python $B --check 2>&1 | grep -E "FAIL|ALL OK|SOME|rror|Traceback" | head -8
done
echo "== compare"
python $B --compare --rank 8  --tiling 2,4,2,6 --out "$OUT/compare9_r8.json"
python $B --compare --rank 16 --tiling 4,4,3,6 --out "$OUT/compare9_r16.json"
echo "== V3 timeline"
STACK_TIMING_VARIANT=v3timing python $B --round2 --rank 8 --tokens 1 --tiling "2,4,2,6" --sm-mhz "$MHZ" \
  --out "$OUT/round9_v3_r8.json" 2>&1 | grep -E "mode [56] (up|down)"
echo done
