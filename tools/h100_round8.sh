#!/usr/bin/env bash
# Round 8 (H100): confirm the best configuration; per-source-line stall profile of the unit.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/stack
B=kernel_work/stack/stack_bench.py
python $B --check 2>&1 | grep -E "FAIL|ALL OK|SOME|Error|Traceback"
python $B --rank 8 --layers 8,32,128 --tokens 1 --tiling 2,4,2,6 --skip-baselines --out "$OUT/r8_round8.json" 2>&1 | grep -E "µs/layer"
python $B --rank 16 --layers 8,32,128 --tokens 1 --tiling 4,4,3,6 --skip-baselines --out "$OUT/r16_round8.json" 2>&1 | grep -E "µs/layer"
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1)
$NCU --profile-from-start off -k regex:tr_stack_kernel -c 1 --import-source yes \
     --section WarpStateStats --section SourceCounters \
     -f -o profiles/h100/stack8_mode6 python /tmp/ncu_target.py 6 > /tmp/ncu8.log 2>&1 || true
[ -f /tmp/ncu_target.py ] || echo "ncu target missing"
$NCU -i profiles/h100/stack8_mode6.ncu-rep --page source --csv --print-source cuda 2>/dev/null > /tmp/src.csv
python - <<'PY' | tee "$OUT/ncu8_lines.txt"
import csv
rows = list(csv.reader(open("/tmp/src.csv")))
hi = next((i for i, r in enumerate(rows) if any(c.strip() in ("Source", "# Source") for c in r)), None)
if hi is None:
    print("no header; first rows:", rows[:3]); raise SystemExit
h = [c.strip() for c in rows[hi]]
def find(*names):
    for n in names:
        for i, c in enumerate(h):
            if c.lower() == n.lower(): return i
    for n in names:
        for i, c in enumerate(h):
            if n.lower() in c.lower(): return i
src = find("Source")
smp = find("Warp Stall Sampling (All Samples)", "Warp Stall Sampling")
ln = find("#", "Line")
print("columns used:", h[src], "|", h[smp])
data = []
for r in rows[hi + 1:]:
    try:
        data.append((float(r[smp] or 0), r[ln] if ln is not None else "", r[src].strip()[:100]))
    except Exception:
        pass
tot = sum(d[0] for d in data) or 1
for v, l, s in sorted(data, reverse=True)[:30]:
    print(f"{100 * v / tot:5.1f}%  line {l:>5}  {s}")
PY
echo done
