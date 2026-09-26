#!/usr/bin/env bash
# Key metrics of Nsight Compute reports: bash tools/ncu_summary.sh profiles/h100/*.ncu-rep
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1)
KEYS='"(Duration|Registers Per Thread|Achieved Occupancy|Executed Instructions|Issue Slots Busy|Compute \(SM\) Throughput|Memory Throughput|Grid Size|Dynamic Shared Memory Per Block|Warp Cycles Per Issued Instruction|Kernel Name)"'
for r in "$@"; do
  echo "== $r"
  "$NCU" -i "$r" --page details --csv 2>/dev/null | grep -E "$KEYS" \
    | python -c 'import csv,sys
for row in csv.reader(sys.stdin): print(" | ".join(row[-3:]))'
done
