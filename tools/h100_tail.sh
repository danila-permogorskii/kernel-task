#!/usr/bin/env bash
# Design B tail of the V3 kernel (t = 1): TR_V3_TAIL 0 (one block converts all of y, as
# measured in results/h100/v3), 1 (same, float4 loads in flight together), 2 (one counter per
# q chunk). Correctness, then the README harness at t = 1 for each tail, twice, interleaved.
#
#   bash tools/remote.sh run bash tools/h100_tail.sh
#
# Output: results/h100/v3tail/{session.log, tail<N>_r<round>_rank<R>.json, kernels_tail<N>.json,
#         B/rank{8,16}.json (the default tail, full README token list), summary.md}
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/v3tail
mkdir -p "$OUT/B" traces/h100/v3tail/B
exec > >(tee -a "$OUT/session.log") 2>&1
echo "== start $(date -u)"

echo "== 1. correctness on the H100"
python tools/check_v3.py 2>&1 | grep -vE "arning" | tail -40
python tools/check_kernel.py --quick 2>&1 | grep -E "FAIL|ALL|SOME" | tail -3
python -m pytest -q 2>&1 | tail -1

echo "== 2. harness, design B, t = 1, tails 0 1 2, two interleaved rounds"
for round in 1 2; do
  for tail in 0 1 2; do
    for R in 8 16; do
      TR_V3_TAIL=$tail TR_DESIGN=B python benchmarks/benchmark.py --device cuda:0 \
        --dtype float16 --rank $R --tokens 1 --output $OUT/tail${tail}_r${round}_rank$R.json \
        2>&1 | grep -E "tokens=1 (dense|factorized_optimized)"
    done
  done
done

echo "== 3. kernel-only time per tail"
for tail in 0 1 2; do
  echo "-- tail $tail"
  TR_V3_TAIL=$tail python tools/measure_kernels.py --out $OUT/kernels_tail$tail.json 2>&1 \
    | grep -E "T= 1 (B|A|dense)"
done

echo "== 4. the README's harness runs, design B, default tail"
TR_DESIGN=B python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
  --rank 8 --tokens 1,8,32 --output $OUT/B/rank8.json --profile-dir traces/h100/v3tail/B/rank8
TR_DESIGN=B python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
  --rank 16 --tokens 1,32 --output $OUT/B/rank16.json --profile-dir traces/h100/v3tail/B/rank16

echo "== 5. summary"
python - <<'PY' | tee "$OUT/summary.md"
import json
from pathlib import Path
root = Path("results/h100/v3tail")
def t1(p):
    for c in json.load(open(p))["cases"]:
        if c["tokens"] == 1:
            return c["methods"]
def us(m, k="factorized_optimized"):
    return 1000 * m[k]["cuda_event_stream_median_ms"]
def kern(tail, R):
    for c in json.load(open(root / f"kernels_tail{tail}.json"))["cases"]:
        if c["rank"] == R and c["tokens"] == 1 and c["design"] == "B":
            return c["kernel_us_per_call"]
    return None
print("| case | tail | dense r1 / r2 | V3 B r1 / r2 | V3 B kernel |")
print("|---|---|---|---|---|")
for R in (8, 16):
    for tail in (0, 1, 2):
        m = [t1(root / f"tail{tail}_r{r}_rank{R}.json") for r in (1, 2)]
        k = kern(tail, R)
        ks = f"{k:.2f}" if isinstance(k, (int, float)) else "see log"
        print(f"| R{R} t1 | {tail} | {us(m[0], 'dense'):.1f} / {us(m[1], 'dense'):.1f} | "
              f"{us(m[0]):.1f} / {us(m[1]):.1f} | {ks} |")
PY
echo "== end $(date -u)"
