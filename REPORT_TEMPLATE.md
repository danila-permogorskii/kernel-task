# Performance report

Keep answers concise and link to supporting artifacts.

GPU: NVIDIA H100 80GB HBM3 (SXM5), driver 580.178.04, CUDA 13.0 runtime (Verda). All numbers
below are measured on it unless marked *(estimate)*.

## 1. Findings

**Implemented:** a fused CUDA C++ kernel that runs the whole three-core tensor-ring operator
in one launch. The ring is cut into independent pieces, one per (ring link `a`, input digit
`k`); a thread block computes a set of pieces entirely on-chip and adds its partial result
into an FP32 workspace. Stage 1 runs on CUDA cores, stages 2 and 3 on Tensor Cores (WMMA,
FP16 in, FP32 accumulate). Two endings were built and measured: **A** (atomics, then a
separate FP32→FP16 convert kernel; 3 launches, stateless; the default) and **B** (the last
block converts and clears; 1 launch, persistent workspace).

**What happened:**

- Correct in all five required cases for both designs (max abs error ≤ 0.0023, tolerance 0.02).
- **Faster than the factorized reference in every case:** 8.3× at R8 t1 down to 1.7× at
  R16 t32 (CUDA-event stream latency).
- **Slower than dense in every case:** 1.3× (R8 t1) to 11.8× (R16 t32). Dense is one cuBLAS
  call at ~10.5 µs per call regardless of size; the ring does 3.6× (R8) to 25× (R16) more
  arithmetic than dense.
- **40–160× less extra resident memory than dense** (0.21–0.81 MB vs 33.6 MB).
- Best Tensor Core utilisation: **7.4% of FP16 peak** (R16 t32).
- **Design B** saves ~1 µs per call at t = 1 (5%) and loses ~6 µs at t = 32, where its
  last-block conversion becomes a serial tail. A remains the default.

**Uncertain:**

- How much of dense's advantage comes from its 11 MB weight staying in L2 in this
  repeated-call microbenchmark. It would not in a full model.
- Whether the proposed v3 (stage 2 → 3 chained in registers via `mma.sync`,
  `kernel-design/V3_IDEAS.md`) would reach a t = 1 tie with dense, as estimated. Not
  implemented.

## 2. Kernel and evidence

- **Kernel source files/entry point, toolchain, and any upstream code adapted:**
  `src/factorized_inference/csrc/tr_ring.cu`: kernels `tr_ring_fused_kernel<design, shape>`
  and `tr_ring_convert_kernel`, launcher `tr_ring_forward`. Host side in
  `src/factorized_inference/tr_kernel.py` (packing, tiling, `PreparedTRKernel`). Entry point
  `prepare_optimized` in `submission.py`: the kernel for CUDA FP16 cores, the reference
  otherwise.
  - Toolchain: built with `torch.utils.cpp_extension.load` (JIT, cached), `nvcc` 13.0.88
    from the pip `cuda-toolkit[nvcc,cccl]==13.0.3` packages, `-O3 -lineinfo`, sm_90; torch
    2.14.0+cu130, Python 3.12.14.
  - No upstream kernel code adapted. Design B uses the standard fence-and-counter pattern
    (as in the CUDA `threadFenceReduction` sample); the code is our own.
  - Walkthrough: `kernel-guides/G2-kernel-walkthrough.md`.
- **Main reference bottleneck and profiler evidence:**
  - Token-1 traces: `traces/h100/A/rank{8,16}/factorized_reference_*_tokens1.json`. One
    reference call is **8 GPU ops: 3 cuBLAS GEMMs (`nvjet_sm90_*`) and 5 elementwise copy
    kernels** that re-lay out the intermediates.
  - At t = 1 the GPU work is ~20 µs, but a call takes 112 µs: it is **host-bound on three
    einsum dispatches**.
  - At R16 t32 it turns GPU-bound (197 µs of kernels in a 212 µs call), where the copies
    move the large intermediates (S1/S2) through HBM.
- **Your kernel's calculation, layout and memory movement; where it appears in the trace:**
  - Grid: one block (256 threads) per (k chunk × q chunk, `a`, token tile), tiling tuned per
    case (`H100_TUNED`).
  - The block loads its slice of B, A[a], C[:,:,k,a] and x into shared memory once, then
    per k runs:
    - stage 1 `S1[(t,p),(j,b)] = Σ_i x·A`, register-tiled FMAs;
    - stage 2 `S2 = S1 @ B2`, WMMA;
    - stage 3 `Y[(t,p,q),r] += S2 @ C`, WMMA into register accumulators.
  - Each stage **writes its result in the layout the next stage reads** (S1 is written as
    rows (t,p); S2's bytes are read as `[(t,p,q) × c]` with row stride Rc). The reference's
    5 copy kernels therefore have no counterpart.
  - Per-call HBM traffic: x, the packed cores (mostly L2 hits) and the FP32 workspace / FP16
    y.
  - In the trace: `tr_ring_fused_kernel<false, Shape<8, 12, 12, 10, 24, 8>>` (design A, R8)
    between a fill kernel and `tr_ring_convert_kernel`. Design B shows a single
    `tr_ring_fused_kernel<true, …>`.
  - Diagrams: `kernel-guides/G1-overview.md`.
- **Preparation, persistent state, temporary buffers and remaining bottlenecks:**
  - Preparation (~75–100 ms per process) = loading the compiled extension and packing the
    cores once. The packed cores are the same values plus zero padding to multiples of 16;
    no dense W is formed.
  - The one-time extension compile (~1 min per machine) runs in `tools/h100_setup.sh`, before
    measurement.
  - Persistent state: A, the packed cores only. B also keeps an FP32 workspace and one
    counter per token tile.
  - Temporary: A allocates a T × 2880 FP32 workspace per call (0.017–0.53 MiB incremental).
  - Remaining bottleneck (Nsight Compute, `profiles/h100/*.ncu-rep`): only ~5% of executed
    instructions are Tensor Core instructions (2.15 M of 40.3 M at R16 t32), 8 warps per SM,
    issue rate 0.34. The kernel is limited by instruction overhead (stage 1 FMAs, operand
    loads, the S2 round trip through shared memory), not by memory or Tensor Core
    throughput.
  - Optimisation history with measurements: `kernel-guides/G3-optimisation-journey.md`.

## 3. Correctness and results

Tests:

- `pytest -q`: 21 passed on the H100.
- `tools/check_kernel.py`: A and B against the FP64 dense oracle, 15 configurations each —
  odd small shapes, forced tilings with ragged k / token / q chunks, changing token counts
  on one prepared object, and repeated calls, which check that B leaves its workspace
  clean. All pass.
- A deliberately broken kernel (wrong C slice) fails the same checks.
- Tolerances: unchanged (FP16 atol = rtol = 0.02).
- Observed errors: max abs 0.0013–0.0023, mean abs ~2.8e-4, relative L2 ~3.5e-4 in every
  case, both designs.
- Harness changes: none. The only difference from the README commands is `--device cuda:0`
  instead of `--device cuda`: torch 2.14 rejects `torch.cuda.set_device("cuda")` in the
  harness. Design B was selected with the environment variable `TR_DESIGN=B`.

Generated from the raw JSON by `tools/report_table.py` (`results/h100/report_table.md`).
Resident / peak / incremental are PyTorch allocator counters as defined in
`BENCHMARK_NOTES.md`.

| Rank | Tokens | Method | Host median ms | CUDA stream median ms | Resident MiB | Steady allocated peak MiB | Incremental workspace/output MiB | Preparation ms | First call ms |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | dense | 0.0181 | 0.0106 | 42.55 | 42.56 | 0.006 | 84.9 | 1.76 |
| 8 | 1 | factorized_reference | 0.1158 | 0.1120 | 32.09 | 33.10 | 1.011 | 0.0 | 85.65 |
| 8 | 1 | **ours, A** | 0.0244 | 0.0142 | 0.29 | 0.30 | 0.017 | 99.1 | 1.13 |
| 8 | 1 | ours, B | 0.0218 | 0.0135 | 0.30 | 0.30 | 0.006 | 97.6 | 1.13 |
| 8 | 8 | dense | 0.0180 | 0.0104 | 42.58 | 42.62 | 0.044 | 84.9 | 1.77 |
| 8 | 8 | factorized_reference | 0.1241 | 0.1190 | 32.11 | 40.10 | 7.983 | 0.0 | 86.55 |
| 8 | 8 | **ours, A** | 0.0367 | 0.0260 | 0.31 | 0.44 | 0.132 | 99.7 | 1.10 |
| 8 | 8 | ours, B | 0.0335 | 0.0252 | 0.40 | 0.44 | 0.044 | 75.3 | 1.00 |
| 8 | 32 | dense | 0.0180 | 0.0107 | 42.66 | 42.84 | 0.176 | 85.2 | 1.76 |
| 8 | 32 | factorized_reference | 0.1212 | 0.1187 | 32.20 | 65.59 | 33.390 | 0.0 | 86.56 |
| 8 | 32 | **ours, A** | 0.0611 | 0.0504 | 0.40 | 0.93 | 0.527 | 99.3 | 1.11 |
| 8 | 32 | ours, B | 0.0648 | 0.0566 | 0.75 | 0.93 | 0.176 | 87.4 | 1.02 |
| 16 | 1 | dense | 0.0181 | 0.0108 | 42.55 | 42.56 | 0.006 | 99.5 | 1.73 |
| 16 | 1 | factorized_reference | 0.1229 | 0.1189 | 32.34 | 36.39 | 4.043 | 0.0 | 94.25 |
| 16 | 1 | **ours, A** | 0.0294 | 0.0190 | 0.76 | 0.78 | 0.017 | 100.0 | 1.11 |
| 16 | 1 | ours, B | 0.0261 | 0.0181 | 0.77 | 0.78 | 0.006 | 98.7 | 1.12 |
| 16 | 32 | dense | 0.0181 | 0.0107 | 42.66 | 42.84 | 0.176 | 85.1 | 1.73 |
| 16 | 32 | factorized_reference | 0.2456 | 0.2120 | 32.46 | 162.52 | 130.059 | 0.0 | 87.37 |
| 16 | 32 | **ours, A** | 0.1379 | 0.1266 | 0.88 | 1.40 | 0.527 | 100.3 | 1.20 |
| 16 | 32 | ours, B | 0.1415 | 0.1329 | 1.23 | 1.40 | 0.176 | 99.3 | 1.22 |

Dense and reference rows are from the design-A run; the B run's copies agree within noise.

What changes across ranks and token counts:

- **vs reference.** The gain is largest at t = 1 (8.3× at R8, 6.6× at R16), where the
  reference pays three einsum dispatches and 8 kernels for ~0.04 µs of arithmetic. It
  shrinks with work (1.7× at R16 t32): both become GPU-bound there, and our kernel is at
  7.4% of Tensor Core peak.
- **vs dense (unfavourable everywhere).**
  - Dense costs the same ~10.5 µs at every size: one GEMM on an 11 MB weight, which likely
    stays in the 50 MB L2 across the repeated calls, plus call overhead.
  - Our cost grows with the ring's arithmetic, which is 3.6× (R8) and 25× (R16) dense's
    per token. The gap is smallest at t = 1 (+2.9 µs R8, +7.3 µs R16) and largest at
    R16 t32 (11.8×).
- **A vs B.** B is faster at t ≤ 8 (one launch instead of three) and slower at t = 32 (the
  last block's serial conversion). Traces and discussion: `kernel-guides/G4-evidence-and-A-vs-B.md`.
- **Memory.**
  - Both baseline processes hold ~32 MiB that ours does not. It is probably cuBLAS
    workspace (our process never calls cuBLAS); not verified.
  - Dense additionally holds its 10.5 MiB weight.
  - The reference's temporaries reach 130 MiB at R16 t32. Ours peak at 1.4 MiB.

Kernel-level measurements (`results/h100/kernels.json`, `tools/measure_kernels.py`):

- An empty kernel takes 0.87 µs of GPU time and ~3 µs of stream time.
- Our fused kernel takes 8.4 µs (R8 t1) to 121 µs (R16 t32).
- *Optional, not a required result:* `torch.compile` of the reference (default mode, no
  CUDA graphs) runs at 106–240 µs per call.
- No CUDA-graph comparison was made.

## 4. System implications

- **Larger ranks.** Stage 2's work grows as R³ and B's shared-memory footprint as R². At
  R = 32, B (~240 KB) would not fit one block, so q-splitting (implemented) or streaming B
  becomes mandatory. The ring's arithmetic per token grows past dense's quickly (25× at
  R16), so higher ranks favour dense unless the memory saving itself is the goal.
- **Larger batches (tokens).** Dense's cost stays flat until its GEMM becomes compute-bound;
  ours grows linearly, with ~25× more FLOPs at R16. The factorized kernel therefore only
  has a chance in the **small-batch / decode regime**, where dense is bound by weight
  traffic and call overhead rather than arithmetic.
- **To establish benefit in a complete system, measure:**
  - end-to-end decode latency and tokens/s with all layers factorized;
  - the same under CUDA graphs, where launch gaps (the A/B difference) mostly disappear;
  - L2 contention: a model's worth of dense weights will not stay in L2 as this benchmark's
    11 MB do, while each layer's ring cores are 87–348 KiB;
  - the memory freed, and the batch size or KV-cache that memory buys;
  - the model quality of the compressed weights (synthetic here, so not measured);
  - per-layer timing inside the model rather than this isolated operator.
- **Integration work:**
  - register the op with `torch.library`, so it composes with `torch.compile` and graph
    capture;
  - one workspace per stream if B is used;
  - a deterministic reduction option, since atomics make the last bits vary between runs.
- **What would make me change approach:**
  - If v3 (`kernel-design/V3_IDEAS.md`) still leaves the kernel instruction-bound at t = 1, I
    would stop optimising the pieces and instead try bounded reconstruction: build W tiles
    on-chip from the cores inside a GEMM, trading arithmetic for dense-like data flow.
  - If the target moves to large batches, dense (or a low-rank alternative with less
    arithmetic) is the better tool.

## 5. Reproduction and disclosure

- **Commands, source revision/archive identifier, environment and dependencies:**
  - Source: kernel and evidence at commit `8f5c392` of this repository; reproduce from the
    commit that contains this report.
  - Environment: `results/h100/environment.txt` (pip freeze) and `results/h100/gpu.txt`
    (nvidia-smi).
  - Setup on a fresh GPU instance, no sudo: `bash tools/h100_setup.sh` (uv venv Python 3.12,
    torch matching the driver, pip nvcc, extension build, `pytest`, `tools/check_kernel.py`).
  - Measurements: `bash tools/h100_session.sh`. It runs the README's two harness commands
    once with `TR_DESIGN=A` and once with `B`, with `--device cuda:0`, plus kernel
    measurements, `torch.compile`, and Nsight Compute.
  - Tiling sweep: `python tools/measure_kernels.py --sweep --out results/h100/sweep_v2.json`.
  - Report table: `python tools/report_table.py`.
  - Runbook: `kernel-guides/H100-RUNBOOK.md`.
- **Raw results and profiler artifacts:**
  - `results/h100/{A,B}/rank{8,16}.json` — harness;
  - `traces/h100/{A,B}/…` — token-1 profiler traces, for Perfetto;
  - `results/h100/kernels*.json`, `sweep_v2.json`, `experiments_v1.json` — kernel-level and
    history;
  - `profiles/h100/fused_R16_T32.ncu-rep`, `fused_R8_T1.ncu-rep` — Nsight Compute;
  - `results/h100/session.log`.
- **GPU hours used and whether compute was stopped:** one H100 SXM5 instance, ~1.2 hours,
  deleted after the session. *(Exact figure: confirm from Verda billing.)*
- **AI tools used and what you independently verified:**
  - Claude Code (Anthropic, Claude Opus 5.5) wrote the CUDA kernel, the host glue, the
    tooling scripts and the guides, and drove the H100 session over SSH under my
    supervision.
  - Design decisions were mine, among them the route (CUDA C++ on H100), A as default and B
    as an experiment, and t = 1 as the headline case.
  - Independently verified by me: *[to fill in: e.g. which checks you re-ran yourself, the
    guides and code sections you worked through, the numbers you cross-checked against the
    raw JSON]*.
