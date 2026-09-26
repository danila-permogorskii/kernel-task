#!/usr/bin/env bash
# Stack: dependency counters (mode 9, timing proxy) vs grid.sync / monotonic barrier.
#   bash tools/remote.sh run bash tools/h100_dep.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/stack
mkdir -p $OUT
exec > >(tee -a "$OUT/dep_session.log") 2>&1
echo "== start $(date -u)"
echo "== check: the real modes still correct after the edit (v3a)"
STACK_VARIANT=v3a timeout 900 python kernel_work/stack/stack_bench.py --check 2>&1 \
  | grep -E "FAIL|ALL OK|SOME|rror|Traceback" | head -8
echo "== R8"
timeout 900 python kernel_work/stack/dep_bench.py --rank 8 --tiling 2,4,2,6 --out $OUT/dep_r8.json
echo "== R16"
timeout 900 python kernel_work/stack/dep_bench.py --rank 16 --tiling 4,4,3,6 --out $OUT/dep_r16.json
echo "== end $(date -u)"
