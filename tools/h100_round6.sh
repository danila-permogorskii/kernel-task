#!/usr/bin/env bash
# Round 6 (H100): dense with weight prefetch; ring spills after YT; ring stall sources.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/stack
echo "== dense stack with L2 weight prefetch"
python kernel_work/hopper_floors/floors2.py --dense-only --out results/h100/floors2_dense.json 2>&1 | grep -vE "arning|^\s*$"
echo "== ring: spills of the unit functions (R = 8)"
rm -rf build/tr_stack_ptxas
python - <<'PY' 2>&1 | grep -E "Function properties|spill|Used [0-9]+ reg" > /tmp/ptx.txt
import os
from torch.utils import cpp_extension
from pathlib import Path
os.environ.setdefault("CUDA_HOME", str(Path(".cuda_home").resolve()))
b = Path("build/tr_stack_ptxas"); b.mkdir(parents=True, exist_ok=True)
cpp_extension.load(name="tr_stack_ptxas", sources=["kernel_work/stack/tr_stack.cu"],
                   build_directory=str(b), extra_cuda_cflags=["-O3", "-Xptxas", "-v"], verbose=True)
PY
python - <<'PY' | tee "$OUT/ptxas_round6.txt"
import re
lines = open("/tmp/ptx.txt").read().splitlines()
for i, l in enumerate(lines):
    m = re.search(r"Function properties for (\S+)", l)
    if not m or i + 1 >= len(lines):
        continue
    name = m.group(1)
    if "ELi8EEE" not in name and "ILi8ELi" not in name:
        continue  # R = 8 only
    short = re.sub(r"_Z\d+", "", name)[:70]
    print(f"{short:72s} {lines[i+1].strip()}")
PY
echo "== ring: Nsight Compute stall sources, mode 6, R = 8, L = 32"
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1)
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
$NCU --profile-from-start off -k regex:tr_stack_kernel -c 1 --import-source yes \
     --section WarpStateStats --section SourceCounters --section MemoryWorkloadAnalysis \
     -f -o profiles/h100/stack6_mode6 python /tmp/ncu_target.py 6 > /tmp/ncu6.log 2>&1
tail -2 /tmp/ncu6.log
$NCU -i profiles/h100/stack6_mode6.ncu-rep --page details --print-units base 2>/dev/null \
  | sed -n '/Warp State Statistics/,$p' | grep -vE "^\s*$" | head -90 | tee "$OUT/ncu6_details.txt"
echo done
