#!/usr/bin/env bash
# Per-source-line warp stall samples from an ncu report: bash tools/ncu_lines.sh <report> [n]
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1)
REP=$1; N=${2:-30}
$NCU -i "$REP" --page source --csv --print-source cuda,sass > /tmp/sass.csv 2>&1
echo "rows: $(wc -l < /tmp/sass.csv)"
head -3 /tmp/sass.csv | cut -c1-400
python - "$N" <<'PY'
import csv, sys, re, collections
n = int(sys.argv[1])
rows = list(csv.reader(open("/tmp/sass.csv")))
hi = next((i for i, r in enumerate(rows) if any("Warp Stall Sampling" in c for c in r)), None)
if hi is None:
    print("no stall-sampling column"); raise SystemExit
h = rows[hi]
smp = next(i for i, c in enumerate(h) if c.startswith("Warp Stall Sampling (All"))
src = next((i for i, c in enumerate(h) if c.strip() in ("Source", "# Source")), None)
addr = next((i for i, c in enumerate(h) if c.strip() in ("Address", "# Address")), None)
print("header:", [c for c in h][:10])
by_line = collections.Counter()
top_sass = []
cur_line = "?"
for r in rows[hi + 1:]:
    if len(r) <= smp:
        continue
    s = r[src] if src is not None else ""
    try:
        v = float(r[smp] or 0)
    except ValueError:
        continue
    top_sass.append((v, s[:90]))
tot = sum(v for v, _ in top_sass) or 1
for v, s in sorted(top_sass, reverse=True)[:n]:
    print(f"{100 * v / tot:5.1f}%  {s}")
PY
