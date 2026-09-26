# Performance report

Keep answers concise and link to supporting artifacts.

GPU: NVIDIA H100 80GB HBM3 (SXM5), driver 580.178.04, CUDA 13.0 runtime (Verda). All numbers
below are measured on it unless marked *(estimate)*. The results table and traces come from
one final session on one instance (`results/h100/session.log`, 2026-09-26).

## 1. Findings

**Implemented:** a CUDA C++ implementation of the whole three-core tensor-ring operator, in
two kernels.

- **V3 kernel, t = 1** (`tr_ring_fused_v3_kernel`), on the two required shapes:
  - stage 2 runs on PTX `mma.sync` Tensor Core instructions;
  - its FP32 result is packed to FP16 **in registers** and fed straight into stage 3's
    `mma.sync`, so it never goes through shared memory;
  - the cores are loaded with `cp.async`.
- **v2 kernel, t > 1** and any other shape (`tr_ring_fused_kernel`): stage 1 on CUDA cores,
  stages 2 and 3 with WMMA.
- **The ring is cut the same way in both.** There is one independent piece per (ring link
  `a`, chunk of input digit `k`, chunk of output digit `q`). A block computes its piece
  on-chip and adds it into an FP32 workspace.
- **Two endings were built and measured:**
  - **B, the default:** one launch. The last block to finish converts FP32 to FP16 and
    clears the workspace. It keeps a persistent workspace between calls.
  - **A:** atomics, then a separate convert kernel. Three launches, stateless.

**What happened:**

- **Correct in all five required cases, both designs.** Max abs error ≤ 0.0023 against a
  tolerance of 0.02.
- **Faster than the factorized reference in every case:** 14.4× at R8 t1, down to 1.7× at
  R16 t32.
- **Against dense:**
  - **R8 t1: 7.9 vs 10.3 µs, 1.3× faster.**
  - R16 t1: 10.5 vs 10.4 µs, a tie.
  - Slower with more tokens: 2.2× at R8 t8, 4.7× at R8 t32, 12.3× at R16 t32. The ring
    does 3.6× (R8) to 25× (R16) more arithmetic per token than dense.
- **~35–140× less resident memory than dense:** 0.30–1.23 MiB against 42.6 MiB. Of dense's
  42.6 MiB, 10.5 MiB is its weight; the rest is probably cuBLAS workspace (see §3).
- **The t = 1 win comes from two changes to the kernel**, same instance each time,
  `results/h100/v3*`:
  1. V3 cut the R8 kernel from 8.4 to 4.5 µs, with 3.1× fewer executed instructions.
  2. Design B's finish loaded values one after another: 12 dependent L2 round trips per
     thread. Now the loads are issued together. This cut the finish from ~4.4 to ~2.2 µs,
     and the call went from 10.1 to 7.9 µs.

**Uncertain:**

- **Kernel against kernel, dense is still faster:** 5.0 vs 6.7 µs, R8 t1
  (`results/h100/kernels.json`). Our call-level win comes partly from a shorter host path:
  one extension call and one launch, against torch dispatch plus cuBLAS. The required
  measurement (uncaptured) includes exactly that path. Under CUDA graphs a single dense
  layer would likely win; that comparison was not made.
- **How much dense gains from its 11 MB weight staying in L2** in this repeated-call
  microbenchmark. It would not in a full model.

## 2. Kernel and evidence

- **Kernel source files/entry point, toolchain, and any upstream code adapted:**
  - Kernels, all in `src/factorized_inference/csrc/tr_ring.cu`:
    - `v3::tr_ring_fused_v3_kernel<design, tail, R, kc, qc>` for t = 1;
    - `tr_ring_fused_kernel<design, shape>` for everything else;
    - `tr_ring_convert_kernel` for design A.
  - Launchers: `tr_ring_forward_v3` and `tr_ring_forward`.
  - Host side, `src/factorized_inference/tr_kernel.py`: packing, tiling, and
    `PreparedTRKernel`, which picks V3 for t = 1 when its tiling is compiled.
  - Entry point: `prepare_optimized` in `submission.py`. It uses the kernel for CUDA FP16
    cores and the reference otherwise.
  - Switches: `TR_DESIGN=A|B` (default B), `TR_V3=0` (disables V3), `TR_V3_TAIL=0|1|2`
    (design B finish, default 2).
  - Toolchain: built with `torch.utils.cpp_extension.load` (JIT, cached), using `nvcc`
    13.0.88 from the pip `cuda-toolkit[nvcc,cccl]==13.0.3` packages, `-O3 -lineinfo`,
    sm_90. torch 2.14.0+cu130, Python 3.12.14.
  - **No upstream kernel code adapted.** Ideas used:
    - the fence-and-counter "last block" pattern (CUDA `threadFenceReduction` sample);
    - keeping a matrix-multiply result in registers as the next multiply's input
      (FlashAttention-2, Dao 2023);
    - the `mma.sync` fragment layouts from the PTX ISA.
    The code is our own.
  - Walkthroughs: `kernel-guides/G2-kernel-walkthrough.md` (v2) and
    `kernel-guides/G5-v3-and-beating-dense.md` (V3).
- **Main reference bottleneck and profiler evidence:**
  - Token-1 traces: `traces/h100/{A,B}/rank{8,16}/factorized_reference_*_tokens1.json`.
  - One reference call is **8 GPU ops: 3 cuBLAS GEMMs (`nvjet_sm90_*`) and 5 elementwise
    copy kernels** that re-lay out the intermediates.
  - At t = 1 the GPU work is ~20 µs, but a call takes 114 µs. It is **host-bound on three
    einsum dispatches**.
  - At R16 t32 it turns GPU-bound: 198 µs of kernels in a 212 µs call. There the copies
    move the large intermediates (S1/S2) through HBM.
- **Your kernel's calculation, layout and memory movement; where it appears in the trace:**
  - **V3 (t = 1).** One block (256 threads) per piece (a, 2–4 k's, 4 q's): 240 blocks at
    R8.
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
    - The reference's 5 copy kernels and v2's S2 round trip through shared memory have no
      counterpart.
  - **v2 (t > 1).** The same pieces, plus token tiles. The block runs stage 1 with
    register-tiled FMAs, then WMMA stages 2 and 3. Each stage writes its result in the
    layout the next stage reads.
  - **Design B finish.** Each block fences, then increments a counter. With V3 there is one
    counter per q chunk; with v2, one per token tile. The last block converts its outputs
    and clears the workspace, with all its loads issued before any store.
  - **Per-call HBM traffic:** x, the packed cores (mostly L2 hits), and the FP32 workspace
    and FP16 y.
  - **In the trace:**
    - design B, t = 1: one `v3::tr_ring_fused_v3_kernel<true, 2, 8, 2, 4>` per call
      (R16: `<true, 2, 16, 4, 4>`);
    - design A: a fill kernel, `tr_ring_fused_v3_kernel<false, …>`, then
      `tr_ring_convert_kernel`.
  - Diagrams: `kernel-guides/G1-overview.md`, `G5-v3-and-beating-dense.md`.
- **Preparation, persistent state, temporary buffers and remaining bottlenecks:**
  - **Preparation** (~90–100 ms per process): loading the compiled extension and packing
    the cores once. The packed cores are the same values plus zero padding to multiples of
    16; no dense W is formed.
  - The one-time extension compile (~1–2 min per machine) runs in `tools/h100_setup.sh`,
    before measurement.
  - **Persistent state:** the packed cores. Design B also keeps an FP32 workspace (T × 2880)
    and up to 16 counters; the kernel leaves both zeroed.
  - **Temporary:** design A allocates a T × 2880 FP32 workspace per call.
  - **Remaining bottlenecks.** Nsight Compute: `profiles/h100/fused_R8_T1.ncu-rep` (V3),
    `fused_R16_T32.ncu-rep` (v2).
    - V3 at R8 t1 executes 0.92 M instructions, against 2.88 M for v2 in the previous
      report. It uses 40 registers per thread (v2: 127).
    - With 240 blocks of 8 warps, achieved occupancy is 23% and a warp waits ~19 cycles
      per issued instruction. At t = 1 the kernel is now **latency-bound**: loads, the
      barrier after stage 1, and the finishing step, which is ~2.2 µs of the 6.7.
    - At R16 t32 the v2 kernel is still instruction-bound: ~5% of its instructions are
      Tensor Core instructions.
  - Optimisation history with measurements: `kernel-guides/G3-optimisation-journey.md`
    (v0 → v2) and `G5-v3-and-beating-dense.md` (V3, the finish).

## 3. Correctness and results

Tests:

- `pytest -q`: 21 passed on the H100.
- `tools/check_kernel.py`: A and B against the FP64 dense oracle, 15 configurations each —
  odd small shapes (including an output size not divisible by 4), forced tilings with
  ragged k / token / q chunks, changing token counts on one prepared object, and repeated
  calls, which check that B leaves its workspace clean. All pass.
- `tools/check_v3.py`: the V3 path against the FP64 oracle and against the v2 kernel, for
  every compiled tiling × {A, B with each finish variant}, three calls each. 16/16 pass.
- A deliberately broken kernel (wrong C slice) fails the same checks.
- Tolerances: unchanged (FP16 atol = rtol = 0.02).
- Observed errors: max abs 0.0013–0.0023, mean abs ~2.8e-4, relative L2 ~3.6e-4 in every
  case, both designs.
- Harness changes: none. The only difference from the README commands is `--device cuda:0`
  instead of `--device cuda`: torch 2.14 rejects `torch.cuda.set_device("cuda")` in the
  harness. The session runs each command twice, with `TR_DESIGN=B` (the default) and
  `TR_DESIGN=A`.

Generated from the raw JSON by `tools/report_table.py` (`results/h100/report_table.md`).
Resident / peak / incremental are PyTorch allocator counters as defined in
`BENCHMARK_NOTES.md`.

| Rank | Tokens | Method | Host median ms | CUDA stream median ms | Resident MiB | Steady allocated peak MiB | Incremental workspace/output MiB | Preparation ms | First call ms | Max abs error |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | dense | 0.0179 | 0.0103 | 42.55 | 42.56 | 0.006 | 81.1 | 1.75 | 0.0016 |
| 8 | 1 | factorized_reference | 0.1161 | 0.1138 | 32.09 | 33.10 | 1.011 | 0.0 | 83.97 | 0.0013 |
| 8 | 1 | **ours, design B (default)** | 0.0162 | 0.0079 | 0.30 | 0.30 | 0.006 | 97.6 | 2.05 | 0.0013 |
| 8 | 1 | ours, design A | 0.0207 | 0.0141 | 0.29 | 0.30 | 0.017 | 97.1 | 2.05 | 0.0013 |
| 8 | 8 | dense | 0.0178 | 0.0104 | 42.58 | 42.62 | 0.044 | 82.1 | 1.75 | 0.0017 |
| 8 | 8 | factorized_reference | 0.1199 | 0.1158 | 32.11 | 40.10 | 7.983 | 0.0 | 83.57 | 0.0018 |
| 8 | 8 | **ours, design B (default)** | 0.0315 | 0.0233 | 0.40 | 0.44 | 0.044 | 90.3 | 2.21 | 0.0018 |
| 8 | 8 | ours, design A | 0.0361 | 0.0255 | 0.31 | 0.44 | 0.132 | 97.8 | 2.08 | 0.0018 |
| 8 | 32 | dense | 0.0182 | 0.0106 | 42.66 | 42.84 | 0.176 | 82.0 | 1.73 | 0.0017 |
| 8 | 32 | factorized_reference | 0.1217 | 0.1196 | 32.20 | 65.59 | 33.390 | 0.0 | 83.76 | 0.0023 |
| 8 | 32 | **ours, design B (default)** | 0.0577 | 0.0496 | 0.75 | 0.93 | 0.176 | 94.4 | 1.88 | 0.0023 |
| 8 | 32 | ours, design A | 0.0618 | 0.0510 | 0.40 | 0.93 | 0.527 | 97.4 | 2.08 | 0.0023 |
| 16 | 1 | dense | 0.0179 | 0.0104 | 42.55 | 42.56 | 0.006 | 81.6 | 1.72 | 0.0016 |
| 16 | 1 | factorized_reference | 0.1169 | 0.1153 | 32.34 | 36.39 | 4.043 | 0.0 | 90.91 | 0.0017 |
| 16 | 1 | **ours, design B (default)** | 0.0186 | 0.0105 | 0.77 | 0.78 | 0.006 | 97.2 | 2.01 | 0.0017 |
| 16 | 1 | ours, design A | 0.0238 | 0.0140 | 0.76 | 0.78 | 0.017 | 96.4 | 2.02 | 0.0017 |
| 16 | 32 | dense | 0.0180 | 0.0103 | 42.66 | 42.84 | 0.176 | 82.3 | 1.73 | 0.0018 |
| 16 | 32 | factorized_reference | 0.2413 | 0.2117 | 32.46 | 162.52 | 130.059 | 0.0 | 84.06 | 0.0019 |
| 16 | 32 | **ours, design B (default)** | 0.1346 | 0.1262 | 1.23 | 1.40 | 0.176 | 94.4 | 1.95 | 0.0019 |
| 16 | 32 | ours, design A | 0.1370 | 0.1260 | 0.88 | 1.40 | 0.527 | 97.5 | 2.14 | 0.0019 |

Dense and reference rows are from the design-B run; the A run's copies agree within noise.

What changes across ranks and token counts:

- **vs reference.** The gain is largest at t = 1: 14.4× at R8, 11.0× at R16. There the
  reference pays three einsum dispatches and 8 kernels for very little arithmetic. The gain
  shrinks with work, to 1.7× at R16 t32: both become GPU-bound, and our kernel is at ~7%
  of Tensor Core peak.
- **vs dense.**
  - Dense costs the same ~10.4 µs at every size: one GEMM (~5 µs of GPU time) on an 11 MB
    weight that likely stays in the 50 MB L2 across the repeated calls, plus ~5 µs of
    torch/cuBLAS call path.
  - At t = 1 our whole call (7.9 µs at R8) is shorter than that. The kernel itself is
    6.7 µs, of which 4.5 µs is the ring's arithmetic and ~2.2 µs the finish.
  - With more tokens our cost grows with the ring's arithmetic, 3.6× (R8) to 25× (R16)
    dense's per token. So dense wins from t = 8 on, by up to 12.3× at R16 t32.
- **A vs B.** B is faster or equal everywhere:
  - 6.2 µs faster at R8 t1, where A's three launches cost more than its kernel;
  - equal at t = 32.
  In the previous report B lost ~6 µs at t = 32. The cause was the same serial finish:
  loads issued one after another. Traces and discussion: `kernel-guides/G4-evidence-and-A-vs-B.md`.
- **Memory.**
  - Both baseline processes hold ~32 MiB that ours does not. It is probably cuBLAS
    workspace (our process never calls cuBLAS); not verified.
  - Dense additionally holds its 10.5 MiB weight.
  - The reference's temporaries reach 130 MiB at R16 t32. Ours peak at 1.4 MiB.

Kernel-level measurements (`results/h100/kernels.json`, `tools/measure_kernels.py`):

- An empty kernel takes ~0.9 µs of GPU time and ~3 µs of stream time.
- **R8 t1:**
  - V3 kernel alone: 4.5 µs;
  - design B kernel, including the finish: 6.7 µs;
  - dense GEMM: 5.0 µs.
- **R16 t32:** our kernel 120–124 µs; dense 4.9 µs.
- *Optional, not a required result:* `torch.compile` of the reference (default mode, no
  CUDA graphs) runs at 106–240 µs per call.
- No CUDA-graph comparison was made.

## 4. System implications

- **Larger ranks.** Stage 2's work grows as R³ and B's shared-memory footprint as R². At
  R = 32, B (~240 KB) would not fit one block, so q-splitting (implemented) or streaming B
  becomes mandatory. The ring's arithmetic per token grows past dense's quickly (25× at
  R16), so higher ranks favour dense unless the memory saving itself is the goal.
- **Larger batches (tokens).** Dense's cost stays flat until its GEMM becomes compute-bound;
  ours grows linearly, with 3.6–25× more FLOPs. So the factorized kernel wins only in the
  **decode regime (t = 1)**, where dense is bound by call overhead and weight traffic rather
  than arithmetic. We win there at R8 and tie at R16.
- **To establish benefit in a complete system, measure:**
  - end-to-end decode latency and tokens/s with all layers factorized;
  - the same under CUDA graphs, which remove most of the host-path difference that part
    of our t = 1 win comes from;
  - L2 contention. A model's dense weights do not stay in L2 as this benchmark's 11 MB
    do, while each layer's ring cores are 87–348 KiB: a 128-layer model's cores fit in
    L2. A side study with all layers in one persistent kernel
    (`kernel-design/WEIGHT_STATIONARY.md`) reached 7.35 µs per layer at R8 t1, against
    7.7 for a cuBLAS chain under a CUDA graph;
  - the memory freed, and the batch size or KV cache that memory buys;
  - the model quality of the compressed weights (synthetic here, so not measured);
  - per-layer timing inside the model rather than this isolated operator.
- **Integration work:**
  - register the op with `torch.library`, so it composes with `torch.compile` and graph
    capture;
  - one workspace per stream for design B, the default. A prepared object must not be
    called concurrently from two streams; `TR_DESIGN=A` is the stateless alternative;
  - a deterministic reduction option, since atomics make the last bits vary between runs.
- **What would make me change approach:**
  - For t > 1: stack tokens into V3's 16-row tiles, and move stage 1 onto
    `mma.sync.m16n8k8`. If that still leaves the ring 2× behind dense at t = 8, I would
    stop and try bounded reconstruction instead: build W tiles on-chip from the cores
    inside a GEMM, trading arithmetic for dense-like data flow.
  - If the target moves to large batches, dense (or a low-rank alternative with less
    arithmetic) is the better tool.
  - Measured and dropped: dependency counters instead of grid barriers in the multi-layer
    kernel. Their measured ceiling is ≤ 0.2 µs per layer (WEIGHT_STATIONARY.md).

## 5. Reproduction and disclosure

- **Commands, source revision/archive identifier, environment and dependencies:**
  - Source: reproduce from the commit that contains this report. V3 and the finish are
    developed in the commits after `8f5c392` (v2).
  - Environment: `results/h100/environment.txt` (pip freeze) and `results/h100/gpu.txt`
    (nvidia-smi).
  - Setup on a fresh GPU instance, no sudo: `bash tools/h100_setup.sh`. It installs a uv
    venv with Python 3.12, torch matching the driver and pip nvcc, builds the extension,
    and runs `pytest` and `tools/check_kernel.py`.
  - Measurements: `SKIP_SWEEP=1 bash tools/h100_session.sh`. It runs `tools/check_v3.py`,
    then the README's two harness commands once with `TR_DESIGN=B` and once with `A`
    (`--device cuda:0`), plus kernel measurements, `torch.compile` and Nsight Compute.
  - Experiment sessions for V3 and the finish: `tools/h100_v3.sh`, `tools/h100_tail.sh`,
    with results in `results/h100/v3`, `results/h100/v3tail`.
  - Tiling sweep (v2 kernel): `python tools/measure_kernels.py --sweep --out results/h100/sweep_v2.json`.
  - Report table: `python tools/report_table.py`.
  - Runbook: `kernel-guides/H100-RUNBOOK.md`.
- **Raw results and profiler artifacts:**
  - `results/h100/{A,B}/rank{8,16}.json` — harness;
  - `traces/h100/{A,B}/…` — token-1 profiler traces, for Perfetto;
  - `results/h100/kernels.json` — kernel-level, from the final session;
  - `results/h100/v3/`, `v3tail/`, `sweep_v2.json`, `experiments_v1.json` — history;
  - `profiles/h100/fused_R8_T1.ncu-rep` (V3), `fused_R16_T32.ncu-rep` (v2) — Nsight Compute;
  - `results/h100/session.log`.
- **GPU hours used and whether compute was stopped:** several short H100 SXM5 sessions, each
  instance deleted afterwards. *(Exact figure: confirm from Verda billing.)*
- **AI tools used and what you independently verified:**
  - Claude Code (Anthropic, Claude Opus 5.5) wrote the CUDA kernels, the host glue, the
    tooling scripts and the guides, and drove the H100 sessions over SSH under my
    supervision.
  - Design decisions were mine, among them:
    - the route (CUDA C++ on H100);
    - t = 1 as the headline case;
    - pursuing V3 and the finish;
    - design B as the default.
  - Independently verified by me: *[to fill in: e.g. which checks you re-ran yourself, the
    guides and code sections you worked through, the numbers you cross-checked against the
    raw JSON]*.
