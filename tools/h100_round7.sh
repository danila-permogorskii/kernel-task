#!/usr/bin/env bash
# Round 7 (H100): in-block balance: split-K stage 2, vectorized stage 1 for down layers.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/stack
B=kernel_work/stack/stack_bench.py
MHZ=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
python $B --check 2>&1 | grep -E "FAIL|ALL OK|SOME|Error|Traceback"
python $B --ablate --rank 8 --tokens 1 --tiling 2,4,2,6 --out "$OUT/ablate7_r8.json"
python $B --round2 --rank 8 --tokens 1 --tiling "2,4,2,6" --sm-mhz "$MHZ" --out "$OUT/round7_r8.json" 2>&1 | grep -E "default|timeline|mode [56] (up|down)"
echo done
