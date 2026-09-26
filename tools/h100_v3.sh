#!/usr/bin/env bash
# V3 in the submission kernel (t = 1): correctness, the README's harness runs, and a same-machine
# comparison with the WMMA kernel (TR_V3=0). Run after tools/h100_setup.sh.
#
#   bash tools/remote.sh run bash tools/h100_v3.sh
#
# Output: results/h100/v3/{A,B}/rank{8,16}.json + traces/h100/v3/...   (V3 on, the new defaults)
#         results/h100/v3/wmma_{A,B}/rank{8,16}_t1.json                (V3 off, t = 1 only)
#         results/h100/v3/kernels.json, results/h100/v3/report_table.md
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/v3
mkdir -p "$OUT" traces/h100/v3
exec > >(tee -a "$OUT/session.log") 2>&1
echo "== start $(date -u)"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv > "$OUT/gpu.txt"

echo "== 1. correctness on the H100"
python tools/check_v3.py 2>&1 | grep -vE "arning" | tail -20
python tools/check_kernel.py --quick 2>&1 | grep -E "FAIL|ALL|SOME" | tail -3
python -m pytest -q 2>&1 | tail -1

echo "== 2. the README's harness runs, V3 on (default), designs A and B"
for D in A B; do
  mkdir -p $OUT/$D traces/h100/v3/$D
  TR_DESIGN=$D python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
    --rank 8 --tokens 1,8,32 --output $OUT/$D/rank8.json --profile-dir traces/h100/v3/$D/rank8
  TR_DESIGN=$D python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
    --rank 16 --tokens 1,32 --output $OUT/$D/rank16.json --profile-dir traces/h100/v3/$D/rank16
done

echo "== 3. same machine, t = 1, V3 off (the previous WMMA kernel)"
for D in A B; do
  mkdir -p $OUT/wmma_$D
  for R in 8 16; do
    TR_V3=0 TR_DESIGN=$D python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
      --rank $R --tokens 1 --output $OUT/wmma_$D/rank${R}_t1.json
  done
done

echo "== 4. kernel-only time and % of ceilings (V3 on)"
python tools/measure_kernels.py --out $OUT/kernels.json 2>&1 | grep -E "floors|T= 1"

echo "== 5. summary"
python - <<'PY' | tee "$OUT/summary.md"
import json
from pathlib import Path
root = Path("results/h100/v3")
def case(p, T):
    for c in json.load(open(p))["cases"]:
        if c["tokens"] == T:
            return c["methods"]
def us(m, k="factorized_optimized"):
    return 1000 * m[k]["cuda_event_stream_median_ms"]
print("| case | dense | reference | WMMA A | WMMA B | **V3 A** | **V3 B** | max abs err V3 A |")
print("|---|---|---|---|---|---|---|---|")
for R in (8, 16):
    va, vb = case(root / "A" / f"rank{R}.json", 1), case(root / "B" / f"rank{R}.json", 1)
    wa, wb = case(root / "wmma_A" / f"rank{R}_t1.json", 1), case(root / "wmma_B" / f"rank{R}_t1.json", 1)
    print(f"| R{R} t1 | {us(va, 'dense'):.1f} | {us(va, 'factorized_reference'):.1f} | "
          f"{us(wa):.1f} | {us(wb):.1f} | **{us(va):.1f}** | **{us(vb):.1f}** | "
          f"{va['factorized_optimized']['correctness']['max_absolute_error']:.4f} |")
PY
echo "== end $(date -u)"
