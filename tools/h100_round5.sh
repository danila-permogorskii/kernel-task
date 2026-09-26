#!/usr/bin/env bash
# Round 5 (H100): Y tiles per warp as a compile-time bound -> spills?
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=results/h100/stack
python - <<'PY' 2>&1 | grep -E "Function properties|spill|Used" | sed -E 's/_Z[0-9A-Za-z]*(tr_stack_kernel|run_unit|run_prefetched|prefetch_cores)/\1/' | cut -c1-120 | paste - - | grep -E "ILi8EEEvRK|ILi8ELi[56]E" | grep -vE "ILi16E" | head -20 | tee "$OUT/ptxas_round5.txt"
import os
from torch.utils import cpp_extension
from pathlib import Path
os.environ.setdefault("CUDA_HOME", str(Path(".cuda_home").resolve()))
b = Path("build/tr_stack_ptxas"); b.mkdir(parents=True, exist_ok=True)
cpp_extension.load(name="tr_stack_ptxas", sources=["kernel_work/stack/tr_stack.cu"],
                   build_directory=str(b), extra_cuda_cflags=["-O3", "-Xptxas", "-v"], verbose=True)
PY
B=kernel_work/stack/stack_bench.py
python $B --check 2>&1 | grep -E "FAIL|ALL OK|SOME|Error|Traceback"
python $B --ablate --rank 8 --tokens 1 --tiling 2,4,2,6 --out "$OUT/ablate5_r8.json"
echo done
