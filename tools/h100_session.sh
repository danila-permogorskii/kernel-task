#!/usr/bin/env bash
# Every measurement for the report, in one run (~20-30 min on an H100). Run after h100_setup.sh.
#
#   bash tools/h100_session.sh            # everything
#   SKIP_SWEEP=1 bash tools/h100_session.sh
#
# Output (pulled to the laptop with tools/remote.sh pull, then committed):
#   results/h100/{A,B}/rank{8,16}.json     the README's harness runs, per design
#   traces/h100/{A,B}/...                  token-1 profiler traces (open in https://ui.perfetto.dev)
#   results/h100/kernels.json              floors, kernel-only time, % of ceilings, torch.compile
#   results/h100/sweep.json                (kc, tt) tiling sweep
#   profiles/h100/*.ncu-rep                Nsight Compute reports, if ncu is installed
#   results/h100/environment.txt, gpu.txt  pip freeze, nvidia-smi
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"   # uv
source .venv/bin/activate
mkdir -p results/h100 traces/h100 profiles/h100
LOG=results/h100/session.log
exec > >(tee -a "$LOG") 2>&1
echo "== session start $(date -u)"

echo "== 0. environment"
nvidia-smi > results/h100/gpu.txt
nvidia-smi -q | grep -iE "product name|driver version|cuda version|max clocks|power limit" -A0 >> results/h100/gpu.txt || true
python -m pip freeze 2>/dev/null > results/h100/environment.txt || uv pip freeze > results/h100/environment.txt

echo "== 0b. correctness: V3 path (t = 1), both designs and every tail"
python tools/check_v3.py 2>&1 | grep -E "FAIL|ALL OK|SOME" | tail -3

echo "== 1. harness, the README's two commands, once per design"
# --device cuda:0 (not cuda): torch 2.14 rejects torch.cuda.set_device("cuda"); see report.
for D in A B; do
  mkdir -p results/h100/$D traces/h100/$D
  TR_DESIGN=$D python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
    --rank 8 --tokens 1,8,32 --output results/h100/$D/rank8.json --profile-dir traces/h100/$D/rank8
  TR_DESIGN=$D python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
    --rank 16 --tokens 1,32 --output results/h100/$D/rank16.json --profile-dir traces/h100/$D/rank16
done

echo "== 2. floors, kernel-only time, % of ceilings (+ torch.compile bonus)"
python tools/measure_kernels.py --with-compile --out results/h100/kernels.json

if [ -z "${SKIP_SWEEP:-}" ]; then
  echo "== 3. tiling sweep (kc, tt)"
  python tools/measure_kernels.py --sweep --out results/h100/sweep.json
fi

echo "== 4. Nsight Compute, design B (default): R=16 T=32 (WMMA kernel), R=8 T=1 (V3 kernel)"
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1 || true)
if [ -n "$NCU" ]; then
  for RT in "16 32" "8 1"; do
    set -- $RT
    TR_DESIGN=B "$NCU" --set full -k regex:tr_ring_fused -c 1 -f \
      -o profiles/h100/fused_R$1_T$2 \
      python tools/ncu_target.py --rank $1 --tokens $2 || echo "ncu failed (permissions?)"
  done
else
  echo "ncu not found: skipped (kernel times and % of peak are still in kernels.json)"
fi

echo "== session end $(date -u)"
echo
echo "Evidence is in results/h100 traces/h100 profiles/h100."
echo "From the laptop:  bash tools/remote.sh pull   then commit and push there."
echo "When the session is over: DELETE the instance in the Verda console."
