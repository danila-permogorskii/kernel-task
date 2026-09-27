# Performance report

Keep answers concise and link to supporting artifacts.

GPU: NVIDIA H100 80GB HBM3 (SXM5), driver 580.178.04, CUDA 13.0 runtime (Verda). All numbers
below are measured on it unless marked *(estimate)*. The results table, traces, kernel
measurements and profiles come from one final session on one instance
(`results/h100/session.log`, 2026-09-27). Comparisons between versions are only made within
one instance.

## 1. Findings

**Implemented:** a CUDA C++ implementation of the whole three-core tensor-ring operator, in
three kernels.

- **V3 kernel, t = 1** (`tr_ring_fused_v3_kernel`), on the two required shapes:
  - stages 2 and 3 run on PTX `mma.sync` Tensor Core instructions;
  - stage 2's FP32 result is packed to FP16 **in registers** and fed straight into stage 3's
    `mma.sync`, so it never goes through shared memory;
  - the cores are loaded with `cp.async`.
- **V3T kernel, t > 1** (`tr_ring_fused_v3t_kernel`), the same two shapes: V3 with the tokens
  stacked into the rows of the `mma` tiles (4 tokens × 12 = 48 rows = 3 full tiles), so the
  cores in shared memory serve 4 tokens at once.
- **v2 kernel** (`tr_ring_fused_kernel`), any other shape: stage 1 on CUDA cores, stages 2
  and 3 with WMMA.
- **The ring is cut the same way in all three.** There is one independent piece per (ring
  link `a`, chunk of input digit `k`, chunk of output digit `q`, tile of tokens). A block
  computes its piece on-chip and adds it into an FP32 workspace.
- **Two endings were built and measured:**
  - **B, the default:** one launch. The last block of each output tile converts FP32 to
    FP16 and clears the workspace. It keeps a persistent workspace between calls.
  - **A:** atomics, then a separate convert kernel. Three launches, stateless.

**What happened:**

- **Correct in all five required cases, both designs.** Max abs error ≤ 0.0023 against a
  tolerance of 0.02.
- **Faster than the factorized reference in every case:** 18.9× at R8 t1, down to 2.6× at
  R16 t32.
- **Against dense** (required, uncaptured measurement):
  - **R8 t1: 8.2 vs 9.8 µs, 1.2× faster.**
  - R16 t1: 10.7 vs 9.9 µs, 1.08× slower.
  - Slower with more tokens: 1.24× at R8 t8, 2.8× at R8 t32, 8.3× at R16 t32. The ring
    does 3.6× (R8) to 25× (R16) more arithmetic per token than dense.
- **~35–140× less resident memory than dense:** 0.30–1.23 MiB against 42.6 MiB. Of dense's
  42.6 MiB, 10.5 MiB is its weight and 32.0 MiB the cuBLAS workspace (measured, §3).
- **Where the gains came from**, each step measured on one instance:
  1. t = 1: V3 cut the R8 kernel from 8.4 to 4.5 µs, with 3.1× fewer executed instructions.
  2. t = 1: design B's finish issued its loads one after another (12 dependent L2 round
     trips per thread). Issuing them together cut the call from 10.1 to 7.9 µs.
  3. t > 1: V3T, then a grid of exactly one wave. R8 t8 went from 23.6 to 12.9 µs, R16 t32
     from 126 to 83 µs (`results/h100/v3t/`, `kernel-guides/G6-v3t-tokens-and-waves.md`).

**Measured, and it matters for the claim** (optional graph section, §3):

- **One layer under CUDA graphs, dense wins:** R8 t1 5.2 vs 6.8 µs. Our uncaptured win comes
  from a shorter host path: one extension call and one launch, against torch dispatch plus
  cuBLAS.
- **128 distinct layers under one CUDA graph** (weights no longer fit in L2, as in a model):
  R8 t1 **7.0 vs 6.7 µs per layer, a tie within 4%**, with 124× less weight memory. Without
  graphs: 8.1 vs 9.4, ours faster.
- **Cold L2, kernel against kernel, R8 t1:** dense 7.9 µs, ours 7.2. Dense's 5.0 µs in the
  repeated-call benchmark relies on its 11 MB weight staying in the 50 MB L2.

## 2. Kernel and evidence

- **Kernel source files/entry point, toolchain, and any upstream code adapted:**
  - Kernels, all in `src/factorized_inference/csrc/tr_ring.cu`:
    - `v3::tr_ring_fused_v3_kernel<design, tail, R, kc, qc>` for t = 1;
    - `v3::tr_ring_fused_v3t_kernel<design, R, kc, qc, tt, mg>` for t > 1;
    - `tr_ring_fused_kernel<design, shape>` for other shapes;
    - `tr_ring_convert_kernel` for design A.
  - Launchers: `tr_ring_forward_v3`, `tr_ring_forward_v3t`, `tr_ring_forward`.
  - Host side, `src/factorized_inference/tr_kernel.py`: packing, tiling, and
    `PreparedTRKernel`, which picks V3 for t = 1 and V3T for t > 1 when the tiling is
    compiled and fits.
  - Entry point: `prepare_optimized` in `submission.py`. It uses the kernel for CUDA FP16
    cores and the reference otherwise.
  - Switches: `TR_DESIGN=A|B` (default B), `TR_V3=0` (V3 and V3T off), `TR_V3T=0` (V3T off),
    `TR_V3_TAIL=0|1|2` (design B finish at t = 1, default 2), `TR_V3T_TILING=kc,qc,tt,mg`.
  - Toolchain: built with `torch.utils.cpp_extension.load` (JIT, cached), using `nvcc`
    13.0.88 from the pip `cuda-toolkit[nvcc,cccl]==13.0.3` packages, `-O3 -lineinfo`,
    sm_90. torch 2.14.0+cu130, Python 3.12.14.
  - **No upstream kernel code adapted.** Ideas used:
    - the fence-and-counter "last block" pattern (CUDA `threadFenceReduction` sample);
    - keeping a matrix-multiply result in registers as the next multiply's input
      (FlashAttention-2, Dao 2023);
    - the `mma.sync` fragment layouts from the PTX ISA.
    The code is our own.
  - Walkthroughs: `kernel-guides/G2-kernel-walkthrough.md` (v2),
    `G5-v3-and-beating-dense.md` (V3), `G6-v3t-tokens-and-waves.md` (V3T).
- **Main reference bottleneck and profiler evidence:**
  - Token-1 traces: `traces/h100/{A,B}/rank{8,16}/factorized_reference_*_tokens1.json`.
  - One reference call is **8 GPU ops: 3 cuBLAS GEMMs (`nvjet_sm90_*`) and 5 elementwise
    copy kernels** that re-lay out the intermediates.
  - At t = 1 the GPU work is ~20 µs, but a call takes ~155 µs on this instance (114 µs on an
    earlier one). It is **host-bound on three einsum dispatches**. Under a CUDA graph the
    same call takes 21 µs.
  - At R16 t32 it turns GPU-bound: 197 µs of kernels in a 213 µs call. There the copies move
    the large intermediates (S1/S2) through HBM.
- **Your kernel's calculation, layout and memory movement; where it appears in the trace:**
  - **V3 (t = 1).** One block (256 threads) per piece (a, 2–4 k's, 4 q's): 240 blocks at R8.
    - `cp.async` loads the block's slice of the packed cores into shared memory, while the
      threads convert x to FP32.
    - Stage 1 (`S1 = x·A`, CUDA-core FMAs) runs for all k of the block, then one
      `__syncthreads`.
    - Then one warp per q, per k:
      - stage 2 `S2 = S1 @ B` on `mma.sync.m16n8k16`;
      - the FP32 accumulator is packed to FP16 in registers, which is exactly the A-operand
        layout of the next `mma`;
      - stage 3 `Y += S2 @ C` on `mma.sync` (m16n8k8 at R8, m16n8k16 at R16).
    - Y stays in registers for all k and is added to the FP32 workspace with
      `red.global.add.v2.f32`.
  - **V3T (t > 1).** The same stages. A block owns (a, 5 k's, 5 q's at R8 / 10 at R16,
    4 tokens). The rows of S1, S2 and Y are (token, p): 48 rows, three full `mma` tiles
    instead of one tile with 4 padding rows. A warp's work item is (q, m tile); its Y tile
    stays in registers over the block's k loop. kc = 5 makes R8 t8 and R16 t8 grids of 128
    blocks, one wave on 132 SMs; kc = 4 (160 blocks) was 1.2–1.5× slower (G6 §5).
  - **Design B finish.** Each block fences, then increments a counter: one per q chunk (V3),
    one per (token tile, q chunk) (V3T), one per token tile (v2). The last block converts
    its outputs and clears the workspace, with all its loads issued before any store.
  - **Per-call HBM traffic:** x, the packed cores (mostly L2 hits), and the FP32 workspace
    and FP16 y.
  - The reference's 5 copy kernels and the S2 round trip through shared memory have no
    counterpart.
  - **In the trace:**
    - design B, t = 1: one `v3::tr_ring_fused_v3_kernel<true, 2, 8, 2, 4>` per call
      (R16: `<true, 2, 16, 4, 4>`);
    - design B, t > 1: one `v3::tr_ring_fused_v3t_kernel<true, 8, 5, 5, 4, 1>` per call
      (R16: `<true, 16, 5, 10, 4, 1>`). The harness saves traces for t = 1 only; the t > 1
      kernel is visible in the Nsight Compute reports `fused_R8_T8` and `fused_R16_T32`;
    - design A: a fill kernel, the fused kernel with `false`, then `tr_ring_convert_kernel`.
  - Diagrams: `kernel-guides/G1-overview.md`, `G5`, `G6`.
- **Preparation, persistent state, temporary buffers and remaining bottlenecks:**
  - **Preparation** (~96–101 ms per process): loading the compiled extension and packing
    the cores once. The packed cores are the same values plus zero padding to multiples of
    16; no dense W is formed.
  - The one-time extension compile (~1–2 min per machine) runs in `tools/h100_setup.sh`,
    before measurement.
  - **Persistent state:** the packed cores. Design B also keeps an FP32 workspace (T × 2880)
    and up to max(16, tiles) counters; the kernel leaves both zeroed.
  - **Temporary:** design A allocates a T × 2880 FP32 workspace per call.
  - **Remaining bottlenecks.** Nsight Compute: `profiles/h100/fused_R8_T1.ncu-rep` (V3),
    `fused_R8_T8.ncu-rep` and `fused_R16_T32.ncu-rep` (V3T); key metrics in
    `results/h100/ncu_summary_final.txt`.
    - V3, R8 t1: 0.92 M executed instructions (v2: 2.88 M), 40 registers, 23% achieved
      occupancy, ~19 cycles per issued instruction. **Latency-bound**: loads, the barrier
      after stage 1, and the finish, which is ~2.2 µs of the 6.7 µs kernel.
    - V3T, R8 t8: 2.6 M instructions, 60 registers, 128 blocks with 82 KB of shared memory
      each, 12.5% occupancy (one 8-warp block per SM), ~7 cycles per issued instruction.
      2.9% of Tensor Core peak: bound by per-block work that is not `mma` (loading the cores,
      stage 1 on CUDA cores, the barrier, the atomics).
    - V3T, R16 t32: 25.4 M instructions (v2: 40.3 M), 95 registers, 178 KB of shared memory,
      11% of Tensor Core peak.
    - A tested hypothesis that failed: several independent `mma` chains per warp (`mg` = 2, 3)
      did not help at t > 1 (G6 §4). The `mma` dependency chain is not the limit.
  - Optimisation history with measurements: `kernel-guides/G3-optimisation-journey.md`
    (v0 → v2), `G5` (V3, the finish), `G6` (V3T).

## 3. Correctness and results

Tests:

- `pytest -q`: 21 passed on the H100.
- `tools/check_kernel.py`: A and B against the FP64 dense oracle, 15 configurations each —
  odd small shapes (including an output size not divisible by 4), forced tilings with
  ragged k / token / q chunks, changing token counts on one prepared object, and repeated
  calls, which check that B leaves its workspace clean. All pass.
- `tools/check_v3.py`: the V3 path against the FP64 oracle and against the v2 kernel, for
  every compiled tiling × {A, B with each finish variant}, three calls each. 16/16 pass.
- `tools/check_v3t.py`: the V3T path, every compiled tiling × {A, B}, token counts
  2, 3, 5, 8, 9, 12, 17, 32, 33 on one prepared object (ragged token tiles, sizes going up and
  down), against the FP64 oracle and the WMMA kernel. 28/28 pass.
- Every output captured inside a CUDA graph (§ graphs below) was also checked against the
  FP64 oracle.
- A deliberately broken kernel (wrong C slice) fails the same checks.
- Tolerances: unchanged (FP16 atol = rtol = 0.02).
- Observed errors: max abs 0.0013–0.0023 in every case, both designs.
- Harness changes: none. The only difference from the README commands is `--device cuda:0`
  instead of `--device cuda`: torch 2.14 rejects `torch.cuda.set_device("cuda")` in the
  harness. The session runs each command twice, with `TR_DESIGN=B` (the default) and
  `TR_DESIGN=A`.

Generated from the raw JSON by `tools/report_table.py` (`results/h100/report_table.md`).
Resident / peak / incremental are PyTorch allocator counters as defined in
`BENCHMARK_NOTES.md`.

| Rank | Tokens | Method | Host median ms | CUDA stream median ms | Resident MiB | Steady allocated peak MiB | Incremental workspace/output MiB | Preparation ms | First call ms | Max abs error |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | dense | 0.0187 | 0.0098 | 42.55 | 42.56 | 0.006 | 86.1 | 1.53 | 0.0016 |
| 8 | 1 | factorized_reference | 0.1612 | 0.1545 | 32.09 | 33.10 | 1.011 | 0.0 | 90.41 | 0.0013 |
| 8 | 1 | **ours, design B (default)** | 0.0168 | 0.0082 | 0.30 | 0.30 | 0.006 | 96.8 | 3.84 | 0.0013 |
| 8 | 1 | ours, design A | 0.0198 | 0.0117 | 0.29 | 0.30 | 0.017 | 97.1 | 3.75 | 0.0013 |
| 8 | 8 | dense | 0.0185 | 0.0099 | 42.58 | 42.62 | 0.044 | 86.6 | 1.57 | 0.0017 |
| 8 | 8 | factorized_reference | 0.1638 | 0.1544 | 32.11 | 40.10 | 7.983 | 0.0 | 87.91 | 0.0018 |
| 8 | 8 | **ours, design B (default)** | 0.0214 | 0.0123 | 0.40 | 0.44 | 0.044 | 97.4 | 3.71 | 0.0018 |
| 8 | 8 | ours, design A | 0.0247 | 0.0143 | 0.31 | 0.44 | 0.132 | 97.6 | 3.86 | 0.0018 |
| 8 | 32 | dense | 0.0190 | 0.0099 | 42.66 | 42.84 | 0.176 | 86.8 | 1.56 | 0.0017 |
| 8 | 32 | factorized_reference | 0.1606 | 0.1527 | 32.20 | 65.59 | 33.390 | 0.0 | 88.26 | 0.0023 |
| 8 | 32 | **ours, design B (default)** | 0.0376 | 0.0279 | 0.75 | 0.93 | 0.176 | 100.3 | 3.85 | 0.0023 |
| 8 | 32 | ours, design A | 0.0392 | 0.0283 | 0.40 | 0.93 | 0.527 | 96.9 | 3.79 | 0.0023 |
| 16 | 1 | dense | 0.0186 | 0.0099 | 42.55 | 42.56 | 0.006 | 88.2 | 1.57 | 0.0016 |
| 16 | 1 | factorized_reference | 0.1656 | 0.1602 | 32.34 | 36.39 | 4.043 | 0.0 | 96.51 | 0.0017 |
| 16 | 1 | **ours, design B (default)** | 0.0192 | 0.0107 | 0.77 | 0.78 | 0.006 | 100.5 | 3.84 | 0.0017 |
| 16 | 1 | ours, design A | 0.0232 | 0.0133 | 0.76 | 0.78 | 0.017 | 96.4 | 3.73 | 0.0017 |
| 16 | 32 | dense | 0.0186 | 0.0099 | 42.66 | 42.84 | 0.176 | 85.8 | 1.55 | 0.0018 |
| 16 | 32 | factorized_reference | 0.2629 | 0.2134 | 32.46 | 162.52 | 130.059 | 0.0 | 89.01 | 0.0019 |
| 16 | 32 | **ours, design B (default)** | 0.0921 | 0.0824 | 1.23 | 1.40 | 0.176 | 101.1 | 4.02 | 0.0019 |
| 16 | 32 | ours, design A | 0.0922 | 0.0811 | 0.88 | 1.40 | 0.527 | 97.2 | 3.84 | 0.0019 |

Dense and reference rows are from the design-B run; the A run's copies agree within noise.

What changes across ranks and token counts:

- **vs reference.** The gain is largest at t = 1: 18.9× at R8, 15.0× at R16. There the
  reference pays three einsum dispatches and 8 kernels for very little arithmetic. The gain
  shrinks with work, to 2.6× at R16 t32: both become GPU-bound, and our kernel is at ~11%
  of Tensor Core peak.
- **vs dense.**
  - Dense costs the same ~9.9 µs at every size: one GEMM (~5 µs of GPU time) on an 11 MB
    weight that stays in the 50 MB L2 across the repeated calls, plus ~5 µs of torch/cuBLAS
    call path.
  - At t = 1 our whole call (8.2 µs at R8) is shorter than that. The kernel itself is
    6.7 µs, of which 4.5 µs is the ring's arithmetic and ~2.2 µs the finish.
  - With more tokens our cost grows with the ring's arithmetic, 3.6× (R8) to 25× (R16)
    dense's per token. So dense wins from t = 8 on: by 1.24× at R8 t8, up to 8.3× at R16 t32.
- **A vs B.** B is faster at small t (3.6 µs at R8 t1, where A's three launches cost more
  than its kernel) and equal within noise at t = 32 (R16 t32: A 81.1, B 82.4). Traces and
  discussion: `kernel-guides/G4-evidence-and-A-vs-B.md`.
- **Memory.**
  - Both baseline processes hold 32.0 MiB of cuBLAS workspace, which ours does not: our
    process never calls cuBLAS. Measured in a fresh process: the first `F.linear` and the
    first reference call each allocate 32.0 MiB, and
    `torch._C._cuda_clearCublasWorkspaces()` frees exactly that; our first call allocates
    0.012 MiB (`results/h100/graphs/cublas.json`).
  - Dense additionally holds its 10.5 MiB weight.
  - The reference's temporaries reach 130 MiB at R16 t32. Ours peak at 1.4 MiB.

Kernel-level measurements (`results/h100/kernels.json`, `tools/measure_kernels.py`):

- An empty kernel takes ~0.9 µs of GPU time and ~2.3 µs of stream time.
- **R8 t1:** V3 kernel alone 4.5 µs; design B kernel, including the finish, 6.7 µs; dense
  GEMM 5.0 µs.
- **R8 t8:** V3T kernel alone 8.5 µs; design B kernel 11.0 µs; dense 4.8 µs.
- **R16 t32:** our kernel 75–81 µs (11–12% of FP16 Tensor Core peak); dense 5.2 µs.
- *Optional, not a required result:* `torch.compile` of the reference (default mode, no
  CUDA graphs) runs at 112–241 µs per call.

### Optional: CUDA graphs, many distinct layers, cold L2

Not part of the required measurement. All methods under the same rules
(`tools/graph_bench.py`; write-up with predictions made before measuring:
`kernel-design/GRAPHS_AND_CHAIN.md`).

One layer, 20 calls captured in one CUDA graph, µs per call (final code,
`results/h100/graphs_final/single.json`):

| case | dense | ours B | ours A | reference |
|---|---|---|---|---|
| R8 t1   | **5.23** | 6.83 | 7.18 | 21.19 |
| R8 t8   | **5.08** | 11.20 | 11.29 | 31.56 |
| R8 t32  | **5.39** | 26.88 | 25.67 | 60.69 |
| R16 t1  | **5.41** | 9.78 | 10.79 | 26.93 |
| R16 t32 | **5.63** | 81.44 | 77.36 | 201.35 |

L distinct instances of the operator (different weights per layer) in one stream, µs per
layer at L = 128 (final code, `results/h100/graphs_final/chain.json`). 128 dense weights are
1.4 GB and must come from HBM; 128 sets of R8 cores are 11 MB:

| case | dense graph | ours B graph | dense eager | ours B eager |
|---|---|---|---|---|
| R8 t1  | **6.71** | 6.99 | 9.39 | **8.13** |
| R8 t8  | **6.83** | 11.55 | **9.58** | 12.80 |
| R16 t1 | **6.69** | 9.85 | **9.30** | 10.84 |
| R16 t8 | **6.83** | 23.43 | **9.32** | 24.57 |

Dense's HBM floor is 3.30 µs per layer (11.06 MB / 3.35 TB/s); at 6.7 µs it reaches ~49% of
HBM bandwidth.

Kernel time with the L2 flushed before each call (submitted-code session before V3T,
`results/h100/graphs/cold.json`): R8 t1 dense 4.97 → **7.87** µs, ours 6.84 → **7.17**;
R16 t1 dense 4.93 → 7.79, ours 9.81 → 10.22.

Reading:

- Under graphs a single dense layer wins everywhere. At R8 t1 the gap is 1.6 µs, about our
  finish (fence, counter, convert).
- In the setting closest to a model (distinct weights, graphs) the R8 ring layer **ties**
  dense at t = 1 (7.0 vs 6.7 µs) with 124× less weight memory, and is 1.7× behind at t = 8.

## 4. System implications

- **Larger ranks.** Stage 2's work grows as R³ and B's shared-memory footprint as R². At
  R = 32, B (~240 KB) would not fit one block, so q-splitting (implemented) or streaming B
  becomes mandatory. The ring's arithmetic per token grows past dense's quickly (25× at
  R16), so higher ranks favour dense unless the memory saving itself is the goal.
- **Larger batches (tokens).** Dense's cost stays flat until its GEMM becomes compute-bound;
  ours grows with t, with 3.6–25× more FLOPs. V3T narrowed the gap at R8 t8 from 2.2× to
  1.24×, but the factorized kernel wins only in the **decode regime (t = 1)**, and only at R8.
- **To establish benefit in a complete system, measure:**
  - end-to-end decode latency and tokens/s with all layers factorized, **under CUDA
    graphs** (which remove the host-path part of our t = 1 win: measured above, a single
    dense layer then wins);
  - L2 contention with the rest of the model (attention, KV cache). Our distinct-layer
    chain shows the effect on the weights alone: dense loses 2.9 µs per call once its weight
    is not in L2, the ring 0.3 µs;
  - a persistent multi-layer kernel. A side study with all layers in one kernel
    (`kernel-design/WEIGHT_STATIONARY.md`) reached 7.35 µs per layer at R8 t1, against 7.7
    for a cuBLAS chain under a CUDA graph; published dense megakernels reach ~60–78% of HBM
    bandwidth (~4.5 µs per layer here), which would beat both;
  - the memory freed, and the batch size or KV cache that memory buys;
  - the model quality of the compressed weights (synthetic here, so not measured);
  - per-layer timing inside the model rather than this isolated operator.
- **Integration work:**
  - register the op with `torch.library`, so it composes with `torch.compile` and graph
    capture (capture already works: §3);
  - one workspace per stream for design B, the default. A prepared object must not be
    called concurrently from two streams; `TR_DESIGN=A` is the stateless alternative;
  - a deterministic reduction option, since atomics make the last bits vary between runs.
- **What would make me change approach:**
  - For t > 1 the next steps are stage 1 on `mma`, a persistent grid that takes work items
    dynamically (no wave tail), and double buffering over k. If R8 t8 still cannot reach
    dense, try bounded reconstruction instead: build W tiles on-chip from the cores inside a
    GEMM, trading arithmetic for dense-like data flow.
  - If the target moves to large batches, dense (or a low-rank alternative with less
    arithmetic) is the better tool.
  - Measured and dropped: dependency counters instead of grid barriers in the multi-layer
    kernel (ceiling ≤ 0.2 µs per layer, WEIGHT_STATIONARY.md); several `mma` chains per warp
    in V3T (no gain, G6 §4).

## 5. Reproduction and disclosure

- **Commands, source revision/archive identifier, environment and dependencies:**
  - Source: reproduce from the commit that contains this report.
  - Environment: `results/h100/environment.txt` (pip freeze) and `results/h100/gpu.txt`
    (nvidia-smi).
  - Setup on a fresh GPU instance, no sudo: `bash tools/h100_setup.sh`. It installs a uv
    venv with Python 3.12, torch matching the driver and pip nvcc, builds the extension,
    and runs `pytest` and `tools/check_kernel.py`.
  - Measurements: `SKIP_SWEEP=1 bash tools/h100_session.sh`. It runs `tools/check_v3.py` and
    `tools/check_v3t.py`, then the README's two harness commands once with `TR_DESIGN=B`
    and once with `A` (`--device cuda:0`), kernel measurements, `torch.compile`, Nsight
    Compute, and the optional graph and chain measurements.
  - Optional measurements on the submitted code before V3T (cuBLAS workspace, cold L2,
    harness repeat): `tools/h100_graphs.sh` → `results/h100/graphs/`.
  - V3T tiling sweeps: `python tools/v3t_sweep.py --out …` → `results/h100/v3t/`.
  - Experiment sessions for V3 and the finish: `tools/h100_v3.sh`, `tools/h100_tail.sh`,
    with results in `results/h100/v3`, `results/h100/v3tail`.
  - Report table: `python tools/report_table.py`.
  - Runbook: `kernel-guides/H100-RUNBOOK.md`.
- **Raw results and profiler artifacts:**
  - `results/h100/{A,B}/rank{8,16}.json` — harness;
  - `traces/h100/{A,B}/…` — token-1 profiler traces, for Perfetto;
  - `results/h100/kernels.json` — kernel-level, from the final session;
  - `results/h100/graphs_final/`, `graphs/` — optional graph, chain, cold-L2, cuBLAS;
  - `results/h100/v3t/` — V3T sweeps (round 1 as a console copy: the instance was deleted
    before its JSON was pulled);
  - `results/h100/v3/`, `v3tail/`, `sweep_v2.json`, `experiments_v1.json` — history;
  - `profiles/h100/fused_R8_T1.ncu-rep` (V3), `fused_R8_T8.ncu-rep`,
    `fused_R16_T32.ncu-rep` (V3T) — Nsight Compute; summary `results/h100/ncu_summary_final.txt`;
  - `results/h100/session.log`.
- **GPU hours used and whether compute was stopped:** several short H100 SXM5 sessions on
  2026-09-26 and 2026-09-27, each instance deleted afterwards. *(Exact figure: confirm from
  Verda billing.)*
- **AI tools used and what you independently verified:**
  - Claude Code (Anthropic, Claude Opus 5.5) wrote the CUDA kernels, the host glue, the
    tooling scripts and the guides, and drove the H100 sessions over SSH under my
    supervision.
  - Design decisions were mine, among them:
    - the route (CUDA C++ on H100);
    - t = 1 as the headline case;
    - pursuing V3 and the finish;
    - design B as the default;
    - the optional graph and chain measurements, and V3T for t > 1.
  - Independently verified by me
