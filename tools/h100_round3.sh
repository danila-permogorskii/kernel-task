#!/usr/bin/env bash
# Round 3 (H100): two-phase loads, monotonic barrier, vector reductions.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/stack
B=kernel_work/stack/stack_bench.py
MHZ=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
python $B --check 2>&1 | grep -E "FAIL|ALL OK|SOME|Error|Traceback" 
python $B --round2 --rank 8 --tokens 1 --tiling "2,4,2,6;2,5,2,6" --sm-mhz "$MHZ" --out "$OUT/round3_r8.json"
python $B --ablate --rank 8 --tokens 1 --tiling 2,4,2,6 --out "$OUT/ablate3_r8.json"
echo done
