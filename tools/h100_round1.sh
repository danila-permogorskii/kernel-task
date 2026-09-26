#!/usr/bin/env bash
# Round 1 of the stack investigation, all on the H100:
#   registers/spills of both builds, correctness, per-part ablation of a unit.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -d .cuda_home/bin ] && export PATH=$PWD/.cuda_home/bin:$PATH
OUT=results/h100/stack
mkdir -p "$OUT"
B=kernel_work/stack/stack_bench.py

echo "== 1. ptxas: registers and spills, default build vs split calls"
echo "nvcc: $(command -v nvcc || echo MISSING)"
INC=$(python -c 'from torch.utils.cpp_extension import include_paths; print(" ".join("-I"+p for p in include_paths("cuda")))')
PYI=$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
: > "$OUT/ptxas_round1.txt"
for def in "" "-DSTACK_SPLIT_CALLS"; do
  echo "-- build ${def:-default}" | tee -a "$OUT/ptxas_round1.txt"
  nvcc -arch=sm_90 -O3 -std=c++20 -Xptxas -v $def -c kernel_work/stack/tr_stack.cu -o /tmp/x.o \
       $INC -I"$PYI" -DTORCH_EXTENSION_NAME=x > /tmp/ptxas.log 2>&1
  echo "nvcc exit $?"
  grep -E "Compiling entry|Function properties|spill|Used" /tmp/ptxas.log \
    | grep -vE "EmptyKernel" | sed -E 's/_Z[0-9A-Za-z]*?(tr_stack_kernel|run_unit|run_prefetched|prefetch_cores|load_cores|compute_unit)/\1/' \
    | cut -c1-160 | tee -a "$OUT/ptxas_round1.txt" | grep -A2 -E "ILi8ELi[04]E|run_unit|run_prefetched|prefetch_cores" | head -60
done

echo "== 2. correctness"
python $B --check 2>&1 | grep -E "FAIL|ALL OK|SOME|Error"

echo "== 3. ablation, R = 8 and 16, T = 1"
python $B --ablate --rank 8  --tokens 1 --tiling 2,4,2,6 --out "$OUT/ablate_r8.json"
python $B --ablate --rank 16 --tokens 1 --tiling 4,4,3,6 --out "$OUT/ablate_r16.json"
echo "done"
