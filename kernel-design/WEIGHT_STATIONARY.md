# Weight-stationary batch-1 decode: where compression could become speed

**Status:** optional exploration beyond the assignment. V0 is implemented and checked on the
laptop (`kernel_work/stack/`); H100 numbers are pending. Predictions below were written
**before** the H100 run.

## 1. The question

On one isolated layer, dense beats our kernel (G4): dense reads 11 MB, which the H100 does in
~3–5 µs, while the ring does 3.6–25× more arithmetic in small, latency-bound pieces.

That comparison is flattering to dense in one way: in the repeated-call benchmark the 11 MB
weight stays in the 50 MB L2. A real model has many **different** layers. Their dense weights
cannot stay on-chip; ring cores can:

```
                         128 layers, FP16      fits in L2 (50 MB)?
   dense                 1.4 GB                no: every token re-reads every weight from HBM
   ring R = 8            11 MB                 yes, with room to spare
   ring R = 16           44 MB                 barely
```

And batch-1 decode is exactly the regime where the ring's weakness (arithmetic grows with the
number of tokens) does not matter. Kog AI's monokernel (one persistent kernel for the whole
decode, no launches, fast grid barriers) attacks the other cost, fixed overheads. Put
together:

> One persistent kernel runs a **stack** of ring layers. Cores never leave L2; there are no
> launches between layers, only a grid barrier. Does a ring layer then cost less than a dense
> layer that must stream its weight from HBM?

## 2. V0: what is implemented

```
   prologue   x (FP16) -> buf[0] (FP32), zero buf[1]                          barrier
   layer l    zero buf[(l+2)%3]; blocks take work units (a, k-chunk, q-chunk)
              of layer l: read buf[l%3] from L2, atomicAdd into buf[(l+1)%3]     barrier
   epilogue   buf[L%3] -> y (FP16)
```

- Layers alternate **up** 1920 → 2880 (the assignment's shape) and **down** 2880 → 1920
  (input and output modes swapped), so they chain like an MLP.
- One work unit = one block of the v2 kernel (same three stages, same shared-memory layout).
- Barrier: an arrival counter plus a generation number (`cuda::atomic_ref`, acquire/release);
  cooperative launch guarantees that every block is resident. `grid.sync()` is measured too.
- Activations stay FP32 between layers, and are read with `__ldcg` (L2, not a stale L1 line).
- 128 registers per thread (non-inlined work unit): 2 blocks per SM, like v2.
- Checked against an FP32 chain of `tr_forward_reference`: rel. L2 error 3.6e-4 for one layer,
  ~8.5e-4 for 8 layers, R = 8 and 16, T = 1, 2, 4.

Compared on the same chain of L different layers, batch 1:

| method | what it pays per layer |
|---|---|
| `dense_eager` | one `F.linear`, weight from HBM, a launch from Python |
| `dense_graph` | the same in one CUDA graph: **the fair baseline** |
| `dense_hbm_floor` | 11.06 MB / 3.35 TB/s = **3.30 µs**: no dense kernel can beat it |
| `v2_graph` | the v2 fused kernel (design B) per layer, in one CUDA graph |
| `stack` | V0: one launch in total, a grid barrier per layer |
| `barrier_ours`, `barrier_cg` | the stack with no work: the cost of the barrier alone |

## 3. Prediction (before measuring)

From the first H100 session (`results/h100/kernels_v2tuned.json`, R = 8, t = 1): the v2
kernel takes **8.4 µs** of GPU time per call with its cores already in L2; the dense GEMV
kernel takes **5.0 µs** with its weight in L2.

| per layer, R = 8, t = 1 | predicted | reasoning |
|---|---|---|
| dense_graph | 4–5.5 µs | 11 MB from HBM; the L2-resident GEMV already took 5.0 µs |
| barrier | 0.5–2 µs | unknown: the number this session exists to measure |
| stack | 9–11 µs | the v2 work unit (8.4 µs) + a barrier; down layers similar |
| v2_graph | 10–13 µs | the v2 kernel per layer + a graph node gap |

**Predicted verdict: V0 loses to dense by about 2×.** Removing launches and keeping weights
in L2 is not enough while one ring layer takes 8 µs of latency. If this holds, the
experiment still gives two useful numbers: the barrier cost (the floor of any persistent
design) and the per-layer latency the ring must reach to win.

```
   ring wins  ⇔   unit latency + barrier   <   dense per-layer time (≈ 4–5 µs, floor 3.3)
                  (V0: ~8.4 + ~1)
```

## 4. What V1 would change (only if the numbers justify it)

The 8.4 µs is latency, not work (0.04 µs of arithmetic at peak). Where it goes is the next
measurement. Candidates, cheapest first:

1. more, smaller units per layer so all 264 resident blocks work at once (the sweep checks
   part of this);
2. reduce the pieces inside a cluster of SMs (distributed shared memory) instead of atomics
   in L2;
3. overlap the next layer's core loads with the current layer's barrier (cores do not depend
   on activations);
4. ★ pin cores in shared memory and send the work to the data (B is needed by every piece).

## 5. Caveats to state with any result

- Only the linear layers are modelled; attention and the KV cache still read HBM.
- Model quality at R = 8 is not measured (random weights).
- The win region is small batch only; at larger batch the ring's arithmetic grows linearly.

## 6. Results (H100 SXM5, 2026-09-26, `results/h100/stack/`)

Tiling from the sweep at L = 32 (`sweep_r{8,16}.json`): R = 8 up (kc 2, qc 5) / down (2, 6);
R = 16 up (4, 5) / down (3, 6). Fewer, larger units won: every unit reloads its B slice.

**µs per layer**: least-squares slope over L = 8, 32, 128 different layers (`r8.json`,
`r16.json`, `r{8,16}_cg.json`):

| case | stack, our barrier | stack, grid.sync | v2 per layer in a CUDA graph | dense, CUDA graph | dense, eager | dense floor (HBM peak) | barrier alone: ours / grid.sync |
|---|---|---|---|---|---|---|---|
| R8, t1 | 12.3 | **11.9** | 20.6 | **7.7** | 11.4 | 3.3 | 2.9 / 1.8 |
| R8, t4 | 20.5 | — | 42.8 | 7.7 | 11.6 | 3.3 | 2.7 / 1.6 |
| R16, t1 | 19.1 | **18.4** | 31.3 | 7.7 | 11.5 | 3.3 | 2.4 / 1.4 |

Correctness: every stack matches the FP32 reference chain; rel. L2 error grows with depth
(8.1e-4 at L = 8, 3.5e-3 at L = 128 for R = 8; 5.4e-3 at L = 128 for R = 16).

### What it says

- **Prediction vs measurement.** Predicted a ~2× loss; measured 1.55× (R = 8). The stack itself
  was slower than predicted (11.9 vs 9–11 µs), but dense was slower too: batch-1 `F.linear`
  streams 11 MB in 7.7 µs, **43% of HBM peak**, not the 4–5.5 µs assumed.
- **The persistent kernel works as intended**: the same work units cost 20.6 µs per layer when
  launched layer by layer in a CUDA graph, 11.9 µs in one persistent kernel (1.7×).
- **The barrier is not the bottleneck.** Alone, our barrier costs 2.9 µs and `grid.sync` 1.8 µs;
  but swapping them inside the full stack saves only 0.3 µs per layer: the barrier wait overlaps
  with the uneven finish of the units. What remains, ~9–10 µs per layer, is the latency of one
  work unit (load the cores from L2, three dependent stages, atomics).
- **Weights in L2 are not enough.** The ring layer never touches HBM for weights and still
  loses, because its per-layer latency is higher than dense's per-layer streaming time.
- **More tokens hurt the ring, not dense** (t = 4: 20.5 vs 7.7 µs), as §3 of the discussion
  predicts.

### The number that decides it

```
   ring beats dense_graph   ⇔   work-unit latency  ≲ 6 µs   (today ~9–10 µs, R = 8)
   ring beats ideal dense   ⇔   work-unit latency  ≲ 1.5 µs (dense at 100% of HBM: 3.3 µs)
```

So V1 is a latency problem: fewer dependent steps per unit (v3's registers-only stage 2 → 3),
and prefetching the next layer's cores during the barrier. Until one unit costs ~2× less, the
weight-stationary design is a measured negative result, not a win.

## 7. Path 1: prefetch the next layer's cores during the barrier (mode 4)

The cores do not depend on activations. The barrier is split into *arrive* and *wait*; between
the two, each block loads the cores of its first unit of the next layer into a second
shared-memory slot. Only the x slice (2–4 KB) stays on the critical path of a layer.

- Shared memory per block: 62 KB (R = 8), ~111 KB (R = 16); two blocks per SM still fit on
  the H100 (228 KB). On the laptop (99 KB) mode 4 runs one block per SM, so laptop timings
  say nothing here.
- Correct on the laptop: modes 0 and 4 match the FP32 chain (R = 8, 16; T = 1, 2, 4).
- **Prediction (before measuring):** the stage-removal run (`experiments_v1.json`) puts the
  load at ~2.5 µs of the unit; hiding it gives **R = 8: ~12 → ~9.5 µs per layer**. If the
  gain is much smaller, the load was already overlapped by the warp scheduler and the unit
  is bound by its dependent stages; then path 2 (registers) is the only lever left.

**Measured** (second H100 instance): mode 4 hid ~2.4 of the ~3.2 µs core load, but the first
build also regressed mode 0 by ~2.5 µs (cores and x were now loaded one after the other).
Net: no gain until the loads were fixed (§8).

## 8. The investigation, round by round (H100 SXM5, second instance, all on one machine)

Every number below is µs per layer, R = 8, t = 1, tiling (2,4,2,6), least-squares slope over
stacks of 8, 32 and 128 different layers; every variant is checked against the FP32 reference
chain first (rel. L2 error 3.5e-4 for one layer, ~9e-4 for eight). Files:
`results/h100/stack/{ablate*,round*,compare*}_r8.json`, floors in `results/h100/floors*.json`.

| step | what changed | best µs/layer |
|---|---|---|
| V0 | first session's kernel, rebuilt as `tr_stack_v0.cu` for A/B | 12.3–12.6 |
| 1 | prefetch during a split barrier (regressed mode 0) | 12.1 |
| 3 | **two-phase loads** (all loads to registers, then all stores), monotonic barrier, `red.v4` | 10.7 |
| 5 | **Y tiles per warp as a template constant**: register spills 224 → 32 bytes | 10.3 |
| 7 | stage 1 of down layers vectorised over b (split-K of stage 2 tried: slower, reverted) | 10.3 |
| 9 | **tiling as compile-time constants** (no division by runtime values) | 9.3 |
| 9 | **V3: stages 2 → 3 in registers on PTX `mma.sync`** | 7.8 |
| 10 | V3 + **`cp.async`** cores, no S1 zeroing, B fragments hoisted | **7.35** |
| 11 | cp.async prefetch issued before `grid.sync` (mode 8) | 7.69 (no gain) |
| — | dense, cuBLAS in a CUDA graph (L different weights) | 7.7 |
| — | dense, our persistent kernel (same framework, 16-byte loads) | 8.1 |
| — | dense floor at 100% HBM | 3.3 |

R = 16: 17.2 → **13.6** (V3, monotonic barrier); dense stays 7.7.

**At batch 1, a stack of ring layers whose cores live in L2 now runs faster than the same
stack of dense layers through cuBLAS (7.35 vs 7.7 µs per layer, R = 8)**, and 10% faster than a
dense persistent kernel built in the same framework. It is not near the HBM floor (3.3 µs), and
neither is dense: both are latency-bound at this layer size.

### Hopper floors (microbenchmarks, `kernel_work/hopper_floors/`)

| mechanism | cost |
|---|---|
| grid barrier, 264 blocks: ours (counter + generation) / `grid.sync` / monotonic `red.release` + `ld.acquire` | 2.19 / 1.28 / 1.38 µs |
| cluster barrier + one global arrival per cluster (x2..x8) | 2.1–2.2 µs (worse) |
| 16–24 KB L2 → shared, 256 threads or `cp.async.bulk` | 0.47–0.59 µs (same latency) |
| 1152 float adds per block into 2880 floats, scalar vs `red.global.add.v4.f32` | 2.10 vs 1.90 µs |
| `mma.sync.m16n8k16` dependent chain / 2 chains / 4 chains / chain + FMUL on the accumulator | 24 / 12.3 / 6.5 / 32 cycles |
| `__shfl_down_sync` / shared-memory hop / `__syncthreads` (256 threads) | 30 / 33 / 29 cycles |
| dense GEMV, 16-byte loads: plain / `.nc` / `.cs` / `L1::no_allocate` | all 8.1 µs/layer (41% HBM) |
| dense + `cp.async.bulk.prefetch.L2` of the next 1 / 2 / 4 layers | 9.2 / 10.2 / 10.3 (worse) |

### What each finding means

- The unit was never limited by arithmetic or by memory bandwidth: it was a chain of small
  dependent steps. What paid off removed steps or instructions: one L2 round trip instead of
  six, no spills, no division by runtime values, and above all no S2 round trip through shared
  memory (V3).
- The accumulator → operand forwarding of `mma.sync` (the FlashAttention-2 register trick) is
  the NVIDIA counterpart of CDNA's DL-ops forwarding: the m16n8 accumulator layout equals the
  A-operand layout of m16n8k8, so the next matmul needs no shared memory and no shuffles.
- Cache hints and L2 prefetch did not move dense at batch 1; the published megakernels reach
  62–78% of HBM by paging shared memory and replacing grid barriers with dependency counters
  (sources below) — the next step for both the dense baseline and the ring.

### Sources

- Hazy Research, *Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B*
  (2025-05-27): counters instead of barriers, 13 × 16 KiB shared-memory pages, 78% HBM on H100.
  https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Mirage Persistent Kernel (MPK), *A Compiler and Runtime for Mega-Kernelizing Tensor Programs*:
  event counters per task, paged shared memory, cross-task pipelining.
  https://arxiv.org/html/2512.22219v1 , https://github.com/mirage-project/mirage
- Cohere, *North Mini Code Megakernel* (2026-09): global-memory counters, one block per SM,
  62% of HBM at batch 1 on H100 (vLLM: 39%). https://cohere.com/blog/megakernels
- Kog AI, *Real-time LLM inference on standard GPUs* (2026-05): monokernel on MI300X / H200.
  https://blog.kog.ai/real-time-llm-inference-on-standard-gpus-3-000-tokens-s-per-request
- PTX ISA, warp-level matrix fragments for `mma.m16n8k16` / `m16n8k8`, `cp.async`,
  `red.global.add.v4.f32`. https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- *The Evolution of Tensor Core Data Layouts* (fragment coordinates).
  https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_layout_generations/index.html
- Colfax, *FlashAttention-2 on Hopper with CUTLASS* (operand kept in registers between GEMMs).
  https://research.colfax-intl.com/wp-content/uploads/2023/12/colfax-flashattention.pdf
- danila-permogorskii/batch1-cdna: the same method (one mechanism per microbenchmark) on MI300X.
  https://github.com/danila-permogorskii/batch1-cdna
- No GPU inference kernels for tensor-ring / tensor-train layers were found; the closest work is
  general tensor-contraction code generation (J. Kim, *Optimizing Tensor Contractions on GPUs*,
  2019).

### Dependency counters: the ceiling, measured first (third instance, 2026-09-26)

In the current cut, each unit of layer l + 1 reads a k chunk of its input. That chunk is
an r chunk of layer l's output, and every unit of layer l writes all r. So every unit
depends on the whole previous layer: a counter there is just a barrier. For a real
dependency, layer l + 1 must be cut by j chunks that match layer l's q chunks. Then each
unit waits for only 1/nqc of the previous layer: 80 of 240 units (R8 up) or 96 of 192
(R8 down).

That re-partition takes hours, so the **ceiling** was measured first with mode 9
(`kernel_work/stack/dep_bench.py`, `tools/h100_dep.sh`, `results/h100/stack/dep_r*.json`).
Mode 9 keeps today's units but makes each unit wait for exactly that pattern: one group of
R · nkc units of the previous layer, counted by per-layer, per-group `red.release`
counters. The control is the same counters with every group awaited, which should equal
a barrier. Numbers are µs per layer (slope over L = 8, 32, 128), round 1 / round 2:

| | grid.sync (3) | monotonic (6) | counters, all groups | **counters, one group** |
|---|---|---|---|---|
| R8 (2,4,2,6)  | 8.00 / 7.93 | 8.38 / 8.36 | 8.37 / 8.37 | **8.00 / 8.00** |
| R16 (4,4,3,6) | 14.12 / 14.11 | 13.77 / 13.77 | 14.00 / 14.00 | **13.59 / 13.59** |

- The control matches the monotonic barrier, so the counter machinery costs nothing extra.
- Waiting for one group instead of all saves 0.37–0.41 µs per layer. Against the best
  barrier for each rank, the gain is **0** at R8 (grid.sync is as good) and **0.18 µs
  (1.3%)** at R16. This is the ceiling: the real re-partition would add costs on top
  (more units, partial sums over j, and a WAR guard on the activation buffers).
- **Why so little:** each group of 80–96 units is spread across all 132 SMs. Its slowest
  unit is almost as slow as the slowest unit of the whole layer. So the ~30% "waiting"
  is **work imbalance between units, not the barrier**. Waiting for fewer producers does
  not change who is slow. Counters would pay off only with *much* finer dependencies
  (a few producers per consumer), which the ring's all-to-all mixing across a layer
  does not allow.
- **Decision: not worth the re-partition.** The lever against the stragglers is balance
  (equal-cost units, e.g. qc = 5 instead of 4 + 4 + 2, or work stealing), not the
  synchronisation.
- On this instance the barriers are ~8% slower than on the second one: 8.0 vs 7.35 µs
  per layer, grid.sync, R8. Compare only within one run.

### Not tried yet (ordered by expected value)

1. ~~Dependency counters~~: measured above, ceiling ≤ 0.2 µs/layer, dropped. Look at the
   straggler instead. First find *which* units are slow, from the per-unit timeline
   (STACK_TIMING): is it a fixed unit type or random (L2 / SM placement)? Only then choose
   a fix. Note that the tiling sweep already preferred qc = 4 (chunks 4 + 4 + 2) over
   qc = 5, so simple chunk balance alone is not the answer.
2. The same V3 treatment for the dense baseline (paged shared memory, weight streaming), to keep
   the comparison honest against a 60–78%-of-HBM dense.
3. `ldmatrix` for the S1 fragments and a q split across more warps (half the warps idle at qc = 4).
4. ~~The V3 path in the single-layer submission kernel~~: done. R8 t1 is 8.0 µs per call
   against dense's 10.6 (`kernel-design/V3_IDEAS.md` §8).
