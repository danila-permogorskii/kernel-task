#!/usr/bin/env bash
# Optional measurements (tools/graph_bench.py): CUDA graphs for every method, a chain of L
# distinct layers, a cold L2, the cuBLAS workspace, and one repeat of the README harness
# (design B) for reproducibility on a different instance. ~30-40 min. Run after h100_setup.sh.
#
#   bash tools/remote.sh run bash tools/h100_graphs.sh
#
# Output: results/h100/graphs/{session.log, gpu.txt, cublas.json, single.json, cold.json,
#         chain.json, repeat_B/rank{8,16}.json}
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/graphs
mkdir -p "$OUT/repeat_B"
exec > >(tee -a "$OUT/session.log") 2>&1
echo "== start $(date -u)"
nvidia-smi > "$OUT/gpu.txt"

echo "== 1. correctness on this instance"
python tools/check_v3.py 2>&1 | grep -E "FAIL|ALL OK|SOME" | tail -3

echo "== 2. cuBLAS workspace (fresh process: nothing has called cuBLAS yet)"
python tools/graph_bench.py --part cublas --out "$OUT/cublas.json" 2>&1 | grep -E "^cublas|wrote"

echo "== 3. single layer, eager vs one CUDA graph, every method, five cases"
python tools/graph_bench.py --part single --out "$OUT/single.json" 2>&1 | grep -E "^single|wrote"

echo "== 4. cold L2: kernel-only time with the L2 flushed before each call"
python tools/graph_bench.py --part cold --out "$OUT/cold.json" 2>&1 | grep -E "^cold|wrote"

echo "== 5. chain of L distinct layers (R 8/16, T 1/8, L 1/8/32/128), eager and one graph"
python tools/graph_bench.py --part chain --out "$OUT/chain.json" 2>&1 | grep -E "^chain|wrote"

echo "== 6. README harness again, design B: same numbers on another machine?"
TR_DESIGN=B python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
  --rank 8 --tokens 1,8,32 --output "$OUT/repeat_B/rank8.json" 2>&1 | grep -E "tokens=" | tail -12
TR_DESIGN=B python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
  --rank 16 --tokens 1,32 --output "$OUT/repeat_B/rank16.json" 2>&1 | grep -E "tokens=" | tail -8

echo "== end $(date -u)"
