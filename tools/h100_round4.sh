#!/usr/bin/env bash
# Round 4 (H100): why does an empty unit cost ~3.5 µs? ptxas report + Nsight Compute.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/stack
mkdir -p "$OUT" profiles/h100
echo "== ptxas (registers, spills, stack frames) of the stack kernels"
python - <<'PY' 2>&1 | grep -E "Compiling entry|spill|Used|stack frame|Function properties" | sed -E 's/_Z[0-9A-Za-z]*(tr_stack_kernel|run_unit|run_prefetched|prefetch_cores)/\1/' | cut -c1-150 | tee "$OUT/ptxas_round4.txt" | grep -A2 -E "ILi8ELi[056]E|run_unit|run_prefetched|prefetch_cores" | head -60
import os, sys
sys.path.insert(0, "kernel_work/stack")
from torch.utils import cpp_extension
from pathlib import Path
os.environ.setdefault("CUDA_HOME", str(Path(".cuda_home").resolve()))
b = Path("build/tr_stack_ptxas"); b.mkdir(parents=True, exist_ok=True)
cpp_extension.load(name="tr_stack_ptxas", sources=["kernel_work/stack/tr_stack.cu"],
                   build_directory=str(b), extra_cuda_cflags=["-O3", "-Xptxas", "-v"], verbose=True)
PY
echo "== Nsight Compute: stack kernel, R = 8, L = 32, T = 1, mode 5 and 6"
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1 || true)
echo "ncu: ${NCU:-MISSING}"
cat > /tmp/ncu_target.py <<'PY'
import sys, torch
sys.path.insert(0, "kernel_work/stack")
from stack_bench import Stack, make_layers
mode = int(sys.argv[1])
st = Stack(make_layers(32, 8), 8, (2, 4, 2, 6))
x = torch.randn(1, 1920, device="cuda", dtype=torch.float16)
for _ in range(3):
    st(x, mode)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
st(x, mode)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
PY
if [ -n "${NCU:-}" ]; then
  for m in 5 6; do
    $NCU --profile-from-start off -k regex:tr_stack_kernel -c 1 \
         --section LaunchStats --section Occupancy --section WarpStateStats \
         --section MemoryWorkloadAnalysis --section SchedulerStats --section SourceCounters \
         --section ComputeWorkloadAnalysis \
         -f -o profiles/h100/stack_mode$m python /tmp/ncu_target.py $m > /tmp/ncu_$m.log 2>&1
    echo "-- mode $m (exit $?)"; tail -3 /tmp/ncu_$m.log
    $NCU -i profiles/h100/stack_mode$m.ncu-rep --page details --print-units base 2>/dev/null \
      | grep -iE "Duration|Registers Per|Stack Size|Local|Achieved Occupancy|Warp Cycles Per Issued|Issued Ipc|L1/TEX Hit|L2 Hit|Stall|Barrier|Long Scoreboard|Short Scoreboard|Membar|Branch|No Instruction|Wait|Sleeping|Dispatch|Mio|Lg Throttle|Tex Throttle|Selected|Not Selected|Math Pipe|Drain" \
      | head -60 | tee "$OUT/ncu_mode$m.txt"
    echo "-- top stall sources, mode $m"
    $NCU -i profiles/h100/stack_mode$m.ncu-rep --page source --csv --print-source sass 2>/dev/null \
      | python -c "
import csv, sys
rows = list(csv.reader(sys.stdin))
if not rows: sys.exit()
h = rows[0]
def col(name):
    for i, c in enumerate(h):
        if c.strip().lower() == name: return i
    return None
ci = col('warp stall sampling (all samples)') or col('warp stall sampling (all cycles)')
si = col('source')
if ci is None: print('columns:', h[:12]); sys.exit()
data = []
for r in rows[1:]:
    try: data.append((float(r[ci] or 0), r[si][:110]))
    except Exception: pass
tot = sum(d[0] for d in data) or 1
for v, s in sorted(data, reverse=True)[:25]:
    print(f'{100*v/tot:5.1f}%  {s}')
" | tee "$OUT/ncu_top_stalls_mode$m.txt"
  done
fi
echo done
