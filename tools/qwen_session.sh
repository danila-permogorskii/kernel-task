#!/usr/bin/env bash
# Qwen3.8-27B layer shapes in BF16 on a rented GPU (A100 first, H100 later). Run after
# tools/h100_setup.sh, which is GPU-independent despite its name. ~50-60 min on an A100.
#
#   bash tools/qwen_session.sh                     # tag from nvidia-smi: a100 / h100
#   GPU_TAG=a100 SKIP_SWEEP=1 bash tools/qwen_session.sh
#   switches: SKIP_SWEEP  SKIP_HARNESS  SKIP_NCU  TOKENS=1,8,32,128  SWEEP_BUDGET_S=25
#
# Output (tools/remote.sh pull brings results/<tag>, traces/<tag>, profiles/<tag> home):
#   results/<tag>/qwen/check_bf16.txt       BF16 correctness vs the FP64 oracle, every shape
#   results/<tag>/qwen/bf16.json            harness protocol, dense / reference / ours (chooser)
#   results/<tag>/qwen/sweep.json           (kc, tt, qc) sweep of our kernel, every shape
#   results/<tag>/qwen/bf16_tuned.json      ours again with the sweep's fastest tiling
#   results/<tag>/qwen/summary.md           tables: µs, dense/ours, GB/s, TFLOP/s, errors, memory
#   results/<tag>/harness_fp16/rank{8,16}.json   the README's two commands on this card (FP16)
#   traces/<tag>/qwen/*.json                token-1 profiler traces (https://ui.perfetto.dev)
#   profiles/<tag>/qwen_*.ncu-rep           Nsight Compute, if ncu is installed
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"   # uv
source .venv/bin/activate

NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$NAME" in
  *A100*) DEFAULT_TAG=a100 ;;
  *H100*) DEFAULT_TAG=h100 ;;
  *)      DEFAULT_TAG=gpu ;;
esac
TAG=${GPU_TAG:-$DEFAULT_TAG}
TOKENS=${TOKENS:-1,8,32,128}
OUT=results/$TAG/qwen
mkdir -p "$OUT" "traces/$TAG/qwen" "profiles/$TAG"
LOG=$OUT/session.log
exec > >(tee -a "$LOG") 2>&1
echo "== session start $(date -u)   GPU: $NAME   tag: $TAG"
ALL_SHAPES=mlp_gate_up,mlp_down,attn_q_gate,attn_kv,o_proj,gdn_qkvz

echo "== 0. environment"
# into $OUT, not results/$TAG: on the H100 those files are the submitted evidence
nvidia-smi > "$OUT/gpu.txt"
nvidia-smi -q | grep -iE "product name|driver version|cuda version|max clocks|power limit" \
  >> "$OUT/gpu.txt" || true
python -m pip freeze 2>/dev/null > "$OUT/environment.txt" \
  || uv pip freeze > "$OUT/environment.txt"
python tools/qwen_shapes.py

echo "== 1. correctness: FP16 on the assignment (all kernels), BF16 on every Qwen shape"
pytest -q 2>&1 | tail -2
python tools/check_kernel.py --quick 2>&1 | tail -1
python tools/check_v3.py 2>&1 | grep -E "FAIL|ALL OK|SOME" | tail -3
python tools/check_v3t.py 2>&1 | grep -E "FAIL|ALL OK|SOME" | tail -3
python tools/check_bf16.py > "$OUT/check_bf16.txt" 2>&1; tail -1 "$OUT/check_bf16.txt"
grep -E "^FAIL|^skip" "$OUT/check_bf16.txt" || true

echo "== 2. Qwen shapes, BF16, harness protocol, the chooser's tilings"
python tools/qwen_bench.py --device cuda:0 --dtype bfloat16 --ranks 8,16 --tokens "$TOKENS" \
  --output "$OUT/bf16.json" --profile-dir "traces/$TAG/qwen" 2>&1 | grep -v USDT

TUNED=()
if [ -z "${SKIP_SWEEP:-}" ]; then
  echo "== 3. tiling sweep of our kernel, then ours again with the fastest tilings"
  python tools/qwen_sweep.py --shapes "$ALL_SHAPES" --ranks 8,16 --tokens "$TOKENS" \
    --budget-s "${SWEEP_BUDGET_S:-25}" --out "$OUT/sweep.json"
  python tools/qwen_bench.py --device cuda:0 --dtype bfloat16 --ranks 8,16 --tokens "$TOKENS" \
    --methods factorized_optimized --tilings "$OUT/sweep.json" \
    --output "$OUT/bf16_tuned.json" 2>&1 | grep -v USDT
  TUNED=(--tuned "$OUT/bf16_tuned.json")
fi

echo "== 4. summary"
python tools/qwen_report.py "$OUT/bf16.json" "${TUNED[@]}" > "$OUT/summary.md"
cat "$OUT/summary.md"

if [ -z "${SKIP_HARNESS:-}" ]; then
  echo "== 5. the assignment on this card: README's two commands, FP16, design B (default)"
  # separate directory: never overwrite the submitted results/h100/{A,B}
  H=results/$TAG/harness_fp16
  mkdir -p "$H" "traces/$TAG/harness_fp16"
  python benchmarks/benchmark.py --device cuda:0 --dtype float16 --rank 8 --tokens 1,8,32 \
    --output "$H/rank8.json" --profile-dir "traces/$TAG/harness_fp16/rank8"
  python benchmarks/benchmark.py --device cuda:0 --dtype float16 --rank 16 --tokens 1,32 \
    --output "$H/rank16.json"
fi

if [ -z "${SKIP_NCU:-}" ]; then
  echo "== 6. Nsight Compute: mlp_gate_up (the biggest weight) R=8 T=1 and T=32, R=16 T=1"
  NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1 || true)
  if [ -n "$NCU" ]; then
    TIL=()
    [ -f "$OUT/sweep.json" ] && TIL=(--tilings "$OUT/sweep.json")
    for RT in "8 1" "8 32" "16 1"; do
      set -- $RT
      TR_DESIGN=B "$NCU" --set full -k regex:tr_ring_fused -c 1 -f \
        -o "profiles/$TAG/qwen_mlp_gate_up_R$1_T$2" \
        python tools/qwen_ncu_target.py --shape mlp_gate_up --rank "$1" --tokens "$2" "${TIL[@]}" \
        || echo "ncu failed (permissions?)"
    done
  else
    echo "ncu not found: skipped"
  fi
fi

echo "== session end $(date -u)"
echo
echo "From the laptop:  bash tools/remote.sh pull   (then commit and push there)."
echo "When the session is over: DELETE the instance in the Verda console."
