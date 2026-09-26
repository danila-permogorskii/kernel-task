#!/usr/bin/env bash
# Weight-stationary stack experiment on the H100 (kernel-design/WEIGHT_STATIONARY.md).
# Run after tools/h100_setup.sh:   bash tools/remote.sh run bash tools/h100_stack.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/stack
mkdir -p "$OUT"
B=kernel_work/stack/stack_bench.py
best() { python -c "import json,sys; print(','.join(map(str, json.load(open(sys.argv[1]))['sweeps'][0]['best'])))" "$1"; }

echo "== 1. correctness (FP32 chain oracle)"
python $B --check 2>&1 | tee "$OUT/check.log"

echo "== 2. tiling search at L = 32, T = 1"
python $B --sweep --rank 8  --tokens 1 --out "$OUT/sweep_r8.json"
python $B --sweep --rank 16 --tokens 1 --out "$OUT/sweep_r16.json"
T8=$(best "$OUT/sweep_r8.json"); T16=$(best "$OUT/sweep_r16.json")
echo "tilings: R8 $T8   R16 $T16"

echo "== 3. stacks of L layers: stack vs v2-in-a-graph vs dense (eager, graph)"
python $B --rank 8  --layers 8,32,128 --tokens 1,4 --tiling "$T8"  --out "$OUT/r8.json"  2>&1 | tee "$OUT/r8.log"
python $B --rank 16 --layers 8,32,128 --tokens 1   --tiling "$T16" --out "$OUT/r16.json" 2>&1 | tee "$OUT/r16.log"

nvidia-smi > "$OUT/gpu.txt"
echo "done: $OUT"
