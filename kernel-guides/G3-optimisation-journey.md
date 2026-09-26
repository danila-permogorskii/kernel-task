# G3: the optimisation journey, every step measured

All times: **fused kernel only**, design A, H100 SXM5, profiler kernel time per call
(`tools/measure_kernels.py`). Files: `results/h100/kernels_*.json`, `experiments_v1.json`,
`sweep_v2.json`, `profiles/h100/*.ncu-rep`.

```
   R16 t32 (the throughput case), µs

   v1 first run      ████████████████████████████████████████████  430
   + padded smem     ██████████████████                             176
   + v2              █████████████                                  125
   + tuned tiling    ████████████                                   121
   reference (all 8 kernels)  ████████████████████                  197
   dense (1 cuBLAS GEMM)      ▌                                       5
```

| Step | R8 t1 | R8 t8 | R8 t32 | R16 t1 | R16 t32 |
|---|---|---|---|---|---|
| v1, first H100 run | 11.7 | 53.3 | 176.6 | 27.3 | 430.0 |
| + padded shared memory, balanced k chunks | 9.8 | 41.0 | 109.5 | 23.4 | 176.5 |
| v2: Y in registers + fixed-shape variants | 8.2 | 21.5 | 50.1 | 12.9 | 125.4 |
| + tiling tuned by sweep (final) | **8.4** | **19.9** | **44.4** | **12.9** | **121.1** |
| reference: all its kernels | 20.5 | 30.8 | 59.5 | 26.2 | 197.4 |

---

## Step 0: the laptop is not a performance proxy

v0 (one piece per block, scalar stages 1 and 3) and the first v1 were tuned on the laptop's
RTX 3050 Ti. That was a mistake worth remembering:

```
                      laptop RTX 3050 Ti      H100 SXM5
   architecture       Ampere sm_86            Hopper sm_90
   SMs                20                      132
   smem per block     99 KB                   227 KB      ← real R = 16 did not even fit
   timing             WSL, noisy (47–340 µs)  stable
```

Lesson kept: **the laptop checks correctness; speed is measured only on the target GPU.**

---

## Step 1: where does v1's time go? (remove one part at a time)

`tools/h100_experiments.py --stages` rebuilds the kernel with one part cut out (wrong
results, right timing). R16 t32:

```
   full                      426.7 µs
   without stage 2           177.6   → stage 2 costs ~249 µs   ◄── the big one
   without stage 3           362.8   → ~64
   without stage 1           390.2   → ~37
   without the load          395.7   → ~31
   without the atomics       411.7   → ~15     (atomics were NOT the problem)
```

Stage 2 is 7.5 GFLOP on Tensor Cores; 249 µs means ~30 TFLOP/s, 3% of the H100's 989. So
the Tensor Cores were mostly waiting.

## Step 2: Nsight Compute says why

```
   occupancy        12.5 %   (1 block of 8 warps per SM: the block uses 213 KB of smem)
   issue rate       0.38 instructions / cycle / scheduler   (max 1)
   instructions     63.4 M,  of which Tensor Core: 2.15 M  (3.4 %)
   smem bank conflicts   6.4 M
```

Two findings:

1. **Bank conflicts.** Shared memory is 32 banks, 4 bytes wide. A Tensor Core tile reads 16
   rows. S1's rows were 384 bytes apart and Y's 128 bytes apart, both multiples of 128, so
   all 16 rows started in the **same bank** and were served one after another.

   ```
   stride 384 B:  row0 → bank 0,  row1 → bank 0,  row2 → bank 0 ...   16-way conflict
   stride 400 B:  row0 → bank 0,  row1 → bank 4,  row2 → bank 8 ...   spread out
   ```

   Fix: pad every row by 16 bytes (`K2s = K2p + 8` halves, and so on). **430 → 176 µs.**

2. **Unbalanced chunks.** The tiling rule could split 20 k values into 19 + 1, so some blocks
   did 19× the work of others. Fix: only balanced chunk sizes, `kc = ceil(nk / chunks)`.

## Step 3: v2, fewer instructions and less shared memory

Two changes, both aimed at the ncu numbers:

- **Y accumulators in registers** (`FragC yacc[8]`). v1 loaded and stored Y from shared
  memory for every k; now each warp keeps its Y tiles in registers for the whole k loop.
  Frees 69 KiB of shared memory per block.
- **Fixed-shape variants.** For the two real workloads all sizes are template constants,
  so divisions by runtime sizes (tens of instructions each) become shifts and multiplies.

```
   instructions   63.4 M → 40.3 M
   registers      61 → 127 per thread
   smem / block   213 → 144 KiB          R16 t32:  176 → 125 µs
```

## Step 4: a q split (no gain) and the tiling sweep

- **q split.** Stage 3 treats each q separately, so blocks can take different q chunks and
  load only their slice of B, at the cost of recomputing stage 1 per chunk. Two blocks then
  fit per SM. Measured: **no gain** (125 → 124 µs at R16 t32). More occupancy was not the
  limit any more.
- **Sweep** (`measure_kernels.py --sweep`): every (kc, tt, qc) for all five cases,
  `results/h100/sweep_v2.json`. The best settings are only 10–20% better than the rule;
  they are stored in `H100_TUNED` and used automatically on sm_90.

## Where it stands, and why

```
   best: R16 t32 at 121 µs = 7.4 % of FP16 Tensor Core peak
   Tensor Core instructions: 2.15 M of 40.3 M executed (5 %)
```

Most executed instructions are not Tensor Core work: stage 1's scalar FMAs, the FP32 → FP16
staging of S2 through shared memory, Tensor Core operand loads and address arithmetic, with
only 8 warps per SM to hide their latency. The next structural step (v3) is in
`kernel-design/V3_IDEAS.md`.

*Read later:* CUDA Best Practices Guide, "Shared Memory" (bank conflicts) and "Occupancy";
Nsight Compute "Warp State Statistics" (what each stall reason means).
