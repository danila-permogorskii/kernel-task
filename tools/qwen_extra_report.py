#!/usr/bin/env python3
"""EXTRA.md: low-bit baselines, crossovers, energy and real-activation errors, from the JSONs of
tools/lowbit_bench.py, tools/qwen_model_chain.py (low-bit run) and tools/tr_activation.py.

    python tools/qwen_extra_report.py results/h100/qwen > results/h100/qwen/EXTRA.md
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
low = json.loads((d / "lowbit.json").read_text())
chain = json.loads((d / "model_chain_lowbit.json").read_text())
act = json.loads((d / "activation.json").read_text())
p = print

p(f"# Ring vs dense BF16 / FP8 / INT4, energy, real activations — {low['device']}\n")
p("Dense FP8 = W8A8 per-tensor scales on cuBLASLt (`torch._scaled_mm`), activation as one cast "
  "(serving fuses its quantisation into RMSNorm). INT4 = weight-only, group 128, tinygemm "
  "(`torch._weight_int4pack_mm`), a GEMV-oriented kernel: at T > 4 Marlin-class kernels are faster. "
  "Ring = our V3G kernel, BF16.\n")
p("## 1. One layer: speed-up at T = 1 and first T where the ring is no longer faster\n")
p("| shape | R | vs BF16: ×(T=1) / T* | vs FP8 | vs INT4 |")
p("|---|---|---|---|---|")
by = {}
for c in low["crossover"]:
    by.setdefault((c["shape"], c["rank"]), {})[c["vs"]] = c
for (shape, R), v in by.items():
    cells = []
    for k in ("dense bf16", "dense fp8", "dense int4"):
        c = v[k]
        t = c["first_T_ring_not_faster"]
        cells.append(f"×{c['t1_speedup']:.2f} / {t if t else '>64'}")
    p(f"| {shape} | {R} | " + " | ".join(cells) + " |")

p("\n## 2. Whole decode step (all linear layers + lm_head, every layer its own weights, CUDA graph)\n")
p("| variant | weights GiB | T | ms / step | tokens/s | J / token | board W |")
p("|---|---|---|---|---|---|---|")
for r in chain["rows"]:
    p(f"| {r['variant']} | {r['weights_bytes'] / 2**30:.1f} | {r['tokens']} | {r['graph_ms']:.2f} | "
      f"{r['tokens'] / r['graph_ms'] * 1e3:.0f} | {r['joules_per_token']:.2f} | {r['avg_watts']:.0f} |")
p("\nlm_head stays BF16 in every variant (2.4 GiB, ~0.8 ms). Energy = median board power "
  "(nvidia-smi, 50 ms samples) × time; attention / DeltaNet recurrence / KV cache not included.\n")

p("## 3. Real inputs: layer 0 Gated DeltaNet input projection (5120 → 16384)\n")
s = act["activation_spectrum"]
p(f"Inputs = token embeddings after the layer's RMSNorm; calibration {act['tokens_cal']} tokens "
  f"(kernel-design/*.md), test {act['tokens_test']} tokens (kernel-guides/*.md). Input energy: "
  f"50% in {s['dims_for_50pct']} directions, 90% in {s['dims_for_90pct']}, 99% in "
  f"{s['dims_for_99pct']} (of 5120). Error = relative output error on the TEST tokens.\n")
p("| R (compression) | ring, matrix error | ring | SVD same size | SVD activation-aware | ring activation-aware (Adam) |")
p("|---|---|---|---|---|---|")
for r in act["rows"]:
    p(f"| {r['R']} (×{r['compression']:.0f}) | {r['matrix_err_ring']:.3f} | {r['out_err_ring']:.3f} | "
      f"{r['out_err_svd']:.3f} | {r['out_err_svd_act']:.3f} | {r['out_err_ring_act']:.3f} |")
p("\nLayer 0 only (its inputs are embeddings, likely lower-dimensional than deeper hidden states); "
  "the activation-aware ring fit is 400 Adam steps from the TR-ALS solution, not converged at R ≤ 16.")
