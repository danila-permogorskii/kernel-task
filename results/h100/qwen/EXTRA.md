# Ring vs dense BF16 / FP8 / INT4, energy, real activations — NVIDIA H100 80GB HBM3

Dense FP8 = W8A8 per-tensor scales on cuBLASLt (`torch._scaled_mm`), activation as one cast (serving fuses its quantisation into RMSNorm). INT4 = weight-only, group 128, tinygemm (`torch._weight_int4pack_mm`), a GEMV-oriented kernel: at T > 4 Marlin-class kernels are faster. Ring = our V3G kernel, BF16.

## 1. One layer: speed-up at T = 1 and first T where the ring is no longer faster

| shape | R | vs BF16: ×(T=1) / T* | vs FP8 | vs INT4 |
|---|---|---|---|---|
| mlp_gate_up | 8 | ×5.23 / 20 | ×3.14 / 10 | ×2.29 / >64 |
| mlp_gate_up | 16 | ×2.82 / 4 | ×1.69 / 2 | ×1.23 / 2 |
| mlp_down | 8 | ×5.16 / 10 | ×3.17 / 6 | ×2.42 / >64 |
| mlp_down | 16 | ×1.83 / 2 | ×1.12 / 2 | ×0.86 / 1 |
| attn_q_gate | 8 | ×4.58 / 16 | ×2.87 / 8 | ×1.95 / >64 |
| attn_q_gate | 16 | ×2.58 / 4 | ×1.61 / 2 | ×1.10 / 2 |
| attn_kv | 8 | ×1.70 / 10 | ×2.45 / 20 | ×0.80 / 1 |
| attn_kv | 16 | ×1.07 / 2 | ×1.55 / 6 | ×0.50 / 1 |
| o_proj | 8 | ×2.69 / 10 | ×1.96 / 6 | ×1.30 / 3 |
| o_proj | 16 | ×1.66 / 2 | ×1.21 / 2 | ×0.81 / 1 |
| gdn_qkvz | 8 | ×5.30 / 20 | ×3.15 / 10 | ×2.29 / >64 |
| gdn_qkvz | 16 | ×2.84 / 4 | ×1.69 / 2 | ×1.23 / 2 |

## 2. Whole decode step (all linear layers + lm_head, every layer its own weights, CUDA graph)

| variant | weights GiB | T | ms / step | tokens/s | J / token | board W |
|---|---|---|---|---|---|---|
| dense | 47.7 | 1 | 19.59 | 51 | 9.38 | 479 |
| dense | 47.7 | 2 | 20.20 | 99 | 4.78 | 473 |
| dense | 47.7 | 4 | 20.28 | 197 | 2.41 | 476 |
| dense | 47.7 | 8 | 20.41 | 392 | 1.23 | 483 |
| dense | 47.7 | 16 | 20.55 | 778 | 0.65 | 504 |
| dense | 47.7 | 32 | 20.82 | 1537 | 0.35 | 545 |
| fp8 | 25.1 | 1 | 12.60 | 79 | 5.48 | 435 |
| fp8 | 25.1 | 2 | 12.83 | 156 | 2.77 | 432 |
| fp8 | 25.1 | 4 | 12.86 | 311 | 1.40 | 434 |
| fp8 | 25.1 | 8 | 12.96 | 617 | 0.71 | 440 |
| fp8 | 25.1 | 16 | 12.88 | 1242 | 0.38 | 466 |
| fp8 | 25.1 | 32 | 13.14 | 2435 | 0.21 | 508 |
| int4 | 14.5 | 1 | 10.04 | 100 | 5.76 | 573 |
| int4 | 14.5 | 2 | 10.58 | 189 | 2.93 | 554 |
| int4 | 14.5 | 4 | 13.95 | 287 | 1.74 | 500 |
| int4 | 14.5 | 8 | 21.19 | 378 | 1.11 | 418 |
| int4 | 14.5 | 16 | 32.64 | 490 | 0.83 | 405 |
| int4 | 14.5 | 32 | 57.32 | 558 | 0.79 | 443 |
| ring8 | 2.6 | 1 | 5.82 | 172 | 1.75 | 300 |
| ring8 | 2.6 | 2 | 7.27 | 275 | 1.26 | 347 |
| ring8 | 2.6 | 4 | 10.81 | 370 | 1.05 | 386 |
| ring8 | 2.6 | 8 | 14.06 | 569 | 0.67 | 379 |
| ring8 | 2.6 | 16 | 23.05 | 694 | 0.60 | 417 |
| ring8 | 2.6 | 32 | 39.75 | 805 | 0.63 | 505 |
| ring16 | 2.8 | 1 | 9.69 | 103 | 3.59 | 370 |
| ring16 | 2.8 | 2 | 15.80 | 127 | 3.13 | 396 |
| ring16 | 2.8 | 4 | 27.65 | 145 | 2.92 | 422 |
| ring16 | 2.8 | 8 | 49.95 | 160 | 2.77 | 444 |
| ring16 | 2.8 | 16 | 94.65 | 169 | 2.71 | 458 |
| ring16 | 2.8 | 32 | 182.07 | 176 | 2.73 | 480 |

lm_head stays BF16 in every variant (2.4 GiB, ~0.8 ms). Energy = median board power (nvidia-smi, 50 ms samples) × time; attention / DeltaNet recurrence / KV cache not included.

## 3. Real inputs: layer 0 Gated DeltaNet input projection (5120 → 16384)

Inputs = token embeddings after the layer's RMSNorm; calibration 19191 tokens (kernel-design/*.md), test 22543 tokens (kernel-guides/*.md). Input energy: 50% in 20 directions, 90% in 517, 99% in 1651 (of 5120). Error = relative output error on the TEST tokens.

| R (compression) | ring, matrix error | ring | SVD same size | SVD activation-aware | ring activation-aware (Adam) |
|---|---|---|---|---|---|
| 8 (×975) | 0.999 | 0.974 | 0.530 | 0.383 | 0.924 |
| 16 (×244) | 0.995 | 0.957 | 0.421 | 0.300 | 0.797 |
| 32 (×61) | 0.979 | 0.930 | 0.325 | 0.221 | 0.346 |
| 64 (×15) | 0.918 | 0.806 | 0.244 | 0.154 | 0.232 |

Layer 0 only (its inputs are embeddings, likely lower-dimensional than deeper hidden states); the activation-aware ring fit is 400 Adam steps from the TR-ALS solution, not converged at R ≤ 16.
