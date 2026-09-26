# Reading guide: exactly what to read before writing the kernel

Every entry names **the exact section**, **the question it answers for our kernel**, and
**what to write down**. Read nothing else from that source. If a section does not answer
its question, stop and ask; do not scroll the whole manual.

Links and section anchors were checked against the live pages on 2026-09-25 (PTX ISA 9.4,
CUDA Programming Guide in its restructured 13.x layout, PyTorch 2.14 docs). Three papers
are saved in `papers/`.

---

## Reading order

```
   WHY the method is legal        HOW the H100 works        HOW to write it          HOW to wire and measure it
   ────────────────────────       ──────────────────        ──────────────           ──────────────────────────
   A. papers (~1.5 h)        ──►  B1. CUDA guide      ──►   B2. PTX mma.sync   ──►   D1. PyTorch custom op
      ring = trace,               B3. Hopper guide          C.  CuTe (optional)      D2. Nsight Compute
      cut anywhere,               (smem, atomics,           (fragments, layouts)     (is it fast, and why)
      TT-layer cost               streams, limits)

   feeds KERNEL_IDEA:  §3              §4.1, §4.3, §5            §4.2                     §7 steps 2 and 5
```

The time estimates below are guesses for a first careful read. The papers come first,
because they confirm that the "cut the ring" idea is mathematically sound before any
CUDA is written.

---

## A. Papers: why the method is legal

### A1. Zhao et al., *Tensor Ring Decomposition*, arXiv:1606.05535 (2016)

`papers/zhao2016_tensor_ring_decomposition.pdf` · <https://arxiv.org/abs/1606.05535> ·
an arXiv preprint (the original TR definition); ~30 min

| Read | Answer for our kernel |
|---|---|
| **§2 "Tensor ring model"** (p. 2): Eq. (1) and Fig. 1 | Map their `Z_k(i_k)` (an R×R slice) onto our `A[:, p, i, :]`. Their trace is our `W[(p,q,r),(i,j,k)]`. |
| **Theorem 2.1** "Circular dimensional permutation invariance" and Eq. (4) (p. 2–3) | The trace can start at any core. **This is why we may cut the ring at link `a`.** Write the one-line argument in your own words. |
| **§5.3 "TT decomposition"** (p. 8) | A ring with one link of size 1 is a tensor train. This is guide 03's "R = 1 dial" in the literature. |

**Skip:** §3 (learning algorithms: SVD, ALS), §4 beyond the definitions, §6 experiments.
We only use a ring; we never fit one.

### A2. Novikov et al., *Tensorizing Neural Networks*, NeurIPS 2015, arXiv:1509.06569

`papers/novikov2015_tensorizing_neural_networks.pdf` · <https://arxiv.org/abs/1509.06569> ·
peer-reviewed; ~30 min

| Read | Answer for our kernel |
|---|---|
| **§3.1 "TT-representations for vectors and matrices"** (p. 3) | Their cores `G_k[i_k, j_k]` are exactly our cores: each pairs one **output** digit with one **input** digit. Our operator is their TT-matrix, closed into a ring. |
| **§4 "TT-layer"** (p. 4): Eq. (5) and the forward complexity `O(d r² m max{M, N})` | Their forward pass is our chain of contractions. **Check:** their cost has r², ours has R³ in stage 2. Explain the extra R in one sentence (hint: the ring's closing link `a` stays open the whole way, guide 04). |
| **Table 1** (p. 4) | Forward-pass complexity and memory, TT vs dense. |
| **§6.4 "Implementation details"** and **Table 3** (p. 8) | Their TT-layer on a GTX 980 GPU against a dense layer. Note which one was faster, and at what batch size. |

**Skip:** §5 (learning), §6.1–6.3 (accuracy experiments).

### A3. Wang et al., *Wide Compression: Tensor Ring Nets*, CVPR 2018, arXiv:1802.09052

`papers/wang2018_tensor_ring_nets_cvpr.pdf` ·
<https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Wide_Compression_Tensor_CVPR_2018_paper.pdf> ·
peer-reviewed; ~20 min

| Read | Answer for our kernel |
|---|---|
| **§3 "merge" and "Merge ordering", Theorem 1** (p. 2–3) | Merging cores into bigger pieces costs `2R³I`–`4R³I` FLOPs and `R²I`–`2R²I` storage. **The R³ and the R² are the same two costs we found** (stage 2's R³ FLOPs, T1/T2's R² size). |
| **§3.1 "Fully Connected Layer Compression"**, Eq. (3) and Fig. 4 (p. 3–4) | **Difference to note:** their ring has `d + d̂` cores, each holding *either* an input *or* an output mode. Ours has 3 cores, each holding one of each (Novikov-style). Their merge-order results therefore do not transfer one-to-one. Write down why. |

**Skip:** §3.2 (convolution), §4 (experiments).

### A4. Deng et al., *TIE: Energy-efficient Tensor Train-based Inference Engine for Deep Neural Network*, ISCA 2019 (optional)

DOI [10.1145/3307650.3322258](https://doi.org/10.1145/3307650.3322258) · ACM Digital Library,
may need university access; peer-reviewed; ~20 min for the abstract and dataflow section

| Read | Answer |
|---|---|
| Abstract and the section on the **inference dataflow** | Their claim is a TT-inference scheme without redundant multiplications. Compare it with our (t, a, k) pieces: does any piece recompute something another piece already has? |

This is a hardware accelerator, not a GPU kernel. Read it for the dataflow idea only.

### A5. Oseledets, *Tensor-Train Decomposition*, SIAM J. Sci. Comput. 33(5), 2011 (background only)

DOI [10.1137/090752286](https://doi.org/10.1137/090752286). Only if A2's notation is
unclear: the TT definition and its first figure. Nothing else is needed.

---

## B. NVIDIA core docs: how the H100 runs it

### B1. CUDA Programming Guide (restructured edition)

Base URL: <https://docs.nvidia.com/cuda/cuda-programming-guide/> · ~1.5 h in total

| Page → section | Answer for our kernel | Feeds |
|---|---|---|
| [02-basics/writing-cuda-kernels → Shared memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory), then [dynamic allocation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#dynamic-allocation-of-shared-memory) | How to allocate our 51–155 KiB per block. **Find the opt-in needed above 48 KB** and write down the exact call. | §4.3 |
| same page → [Shared memory bank conflicts](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-bank-conflicts) and [transpose example using shared memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#matrix-transpose-example-using-shared-memory) | Writing S1 "in the next stage's order" is a transpose inside shared memory. This is the classic example of doing it without bank conflicts. | §4.2 |
| same page → [Atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#atomics) and [Kernel launch and occupancy](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#kernel-launch-and-occupancy) | Option A's `atomicAdd`; how many of our blocks fit per SM at 131 KiB shared memory. | §5, §4.1 |
| [05-appendices/cpp-language-extensions → atomicAdd](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomicadd) | Which types `atomicAdd` supports (FP32 yes; check `__half2`). | §5 |
| same page → [Warp shuffle functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions) and [launch bounds](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#launch-bounds) | Summing partials inside a warp before one atomic; capping registers per thread. | §5 |
| [02-basics/asynchronous-execution → Launching kernels in CUDA streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#launching-kernels-in-cuda-streams) and [default stream](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#blocking-and-non-blocking-streams-and-the-default-stream) | The project requires output "usable on the caller's stream". Our memset, kernel and convert must all go on that one stream. | §5, D1 |
| [05-appendices/compute-capabilities → Features and technical specifications](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications) | The **compute capability 9.0** column: max shared memory per block, registers per SM, max resident blocks per SM. Copy those numbers into the design. | §4.3 |
| *later, for v2:* [03-advanced/advanced-kernel-programming → Asynchronous data copies](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#asynchronous-data-copies) | Overlapping the next tile's loads with this tile's math. Not needed for v0/v1. | §7 step 4 |

**Skip:** unified memory, graphics interop, multi-GPU, dynamic parallelism, CUDA graphs
(not allowed in the required runs anyway).

### B2. PTX ISA 9.4: the Tensor Core instruction

Base URL: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html> · ~1 h,
most of it on one figure

| Section (anchor) | Answer for our kernel |
|---|---|
| [`mma` instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma) | The exact form we use: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (FP16 in, FP32 accumulation, as the harness requires). |
| [Matrix fragments for m16n8k16, floating point](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float), figures [A](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16816-a-f16), [B](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16816-b-f16), [C/D](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16816-c) | **The most important page.** Which of the 32 threads holds which element of A (16×16), B (16×8) and C (16×8). Draw it for one stage-2 tile: rows = (p, t), K = (j, b), N = (q, c). |
| [`ldmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix) and [its fragments](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-ldmatrix-fragments) | How to load those fragments from shared memory in one instruction instead of 8 scalar loads. |
| [`red`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red) (and [`atom`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom) for contrast) | `red.global.add.f32` is an atomic add that returns nothing: the right instruction for option A. Check which vector widths it supports. |
| *only if step 4 demands it:* [`wgmma.mma_async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma) and [shared memory layout / matrix descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor) | Hopper's full-rate Tensor Core path. Read only if R = 16, t = 32 turns out FLOP-bound in stage 2. |

**Skip:** everything else in PTX. It is 4 MB of HTML.

### B3. Hopper Tuning Guide and the H100 whitepaper

<https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html> · ~20 min

| Section | Answer |
|---|---|
| [Occupancy](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy) | 228 KB shared memory per SM, 227 KB max per block: the ceiling in KERNEL_IDEA §4.3. Confirm both numbers. |
| [Increased L2 capacity](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#increased-l2-capacity) | 50 MB L2. Dense's 11 MB W can stay warm there in the harness's repeated calls, making dense faster than an HBM model predicts. |
| [Unified shared memory / L1 / texture cache](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#unified-shared-memory-l1-texture-cache) | The shared memory / L1 split, and how the carveout is chosen. |
| [Tensor Memory Accelerator](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator), [Thread block clusters](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#thread-block-clusters) | Skim only: know they exist. A cluster could share B across blocks via distributed shared memory. That is a v2 idea, not v1. |

H100 whitepaper: <https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c>.
Read only the **H100 SXM5 specification table** (SM count, FP16 Tensor TFLOPS, HBM3
bandwidth, L2 size) and the **SM block diagram**. Check that FP16 with **FP32
accumulation** runs at full rate on the H100. On the laptop's GeForce GPU it ran at 57%
(guide 06, step 3).

---

## C. CUTLASS / CuTe: how others lay out Tensor Core tiles

Repository: <https://github.com/NVIDIA/cutlass> · license **BSD-3-Clause**
(`LICENSE.txt`). If you adapt any code, record the source, the license and your changes;
`IMPLEMENTATION.md` requires this. ~1 h

**Decide first:** v0/v1 can be written with raw `mma.sync` (B2) and no CUTLASS. Read C to
*think* about layouts; adopt CuTe only if hand-writing the fragment indexing becomes the
bottleneck.

| Document | Answer |
|---|---|
| [`media/docs/cpp/cute/00_quickstart.md`](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md) | What CuTe is; how to build its examples. |
| [`media/docs/cpp/cute/01_layout.md`](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md) | **Layout = (Shape, Stride)**: a coordinate becomes an offset. That is guide 02's "an index is a number made of digits", as a C++ type. Express S1's layout `[(p,t) × (j,b)]` in this notation. |
| [`media/docs/cpp/cute/0t_mma_atom.md`](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0t_mma_atom.md) | The `SM80_16x8x16_F32F16F16F32_TN` atom: the same instruction as B2, with its thread–value layout. Check it matches the PTX figure you drew. |
| [`media/docs/cpp/cute/02_layout_algebra.md`](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md) | Skim `composition` only, as a way to express "write S1 in the next stage's order". |
| [`examples/cute/tutorial/sgemm_sm80.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu) | The structure of a tiled GEMM kernel: global → shared → registers → compute → write. Note: this tutorial uses plain FMA, not Tensor Cores (CUTLASS issue #1520); the structure is what matters. |
| *later:* [`examples/cute/tutorial/hopper/wgmma_sm90.cu`](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/hopper) | Only together with B2's wgmma sections. |

**Skip:** the CUTLASS 2.x/3.x device-level GEMM APIs and collective builders. They solve
large GEMMs; ours are tiny and fused.

---

## D. Wiring and measuring

### D1. PyTorch custom C++/CUDA operator

~45 min

| Source → section | Answer |
|---|---|
| Tutorial [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html): [build system](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html#setting-up-the-build-system), [defining an operator](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html#defining-an-operator), [registering backend implementations](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html#registering-backend-implementations-for-an-operator), [testing an operator](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html#testing-an-operator) | How `prepare_optimized` calls our CUDA code as one PyTorch op. |
| [`torch.utils.cpp_extension.load`](https://docs.pytorch.org/docs/2.14/cpp_extension.html#torch.utils.cpp_extension.load) / [`load_inline`](https://docs.pytorch.org/docs/2.14/cpp_extension.html#torch.utils.cpp_extension.load_inline) / [`CUDAExtension`](https://docs.pytorch.org/docs/2.14/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension) | Compile at first use (`load`) or at install (`CUDAExtension`). Where to pass `-arch=sm_90`. Where compile time lands: `preparation_ms` or `first_call_ms`. **The harness reports both.** |
| [CUDA semantics → CUDA streams](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-streams) | Launch on `at::cuda::getCurrentCUDAStream()`, never on the default stream: "usable on the caller's stream" (`IMPLEMENTATION.md`). |

**Skip:** the tutorial's sections on autograd, `torch.compile`, mutable operators and the
stable ABI. Training is out of scope.

### D2. Nsight Compute (Nsight Systems optional)

[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) ·
[CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) · ~30 min

| Read | Answer |
|---|---|
| Profiling Guide → [Sets and sections](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sets-and-sections) | What each section measures. |
| Profiling Guide → [Metrics guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide) (memory part only) | How "achieved bandwidth" is defined, to compare with guide 06's `BW`. |

The section identifiers below are copied from `ncu --list-sections` on this machine
(Nsight Compute 2026.1):

```
   question                                       --section
   ─────────────────────────────────────────────  ─────────────────────────────────────────────
   how close to peak (bytes and compute)?         SpeedOfLight
   achieved DRAM / L2 bandwidth, bank conflicts   MemoryWorkloadAnalysis, MemoryWorkloadAnalysis_Tables
   Tensor Core pipe utilisation                   ComputeWorkloadAnalysis
   roofline, Tensor Core                          SpeedOfLight_HierarchicalTensorRooflineChart
   occupancy; grid and block sizes; registers     Occupancy, LaunchStats
   why warps stall                                WarpStateStats
```

Example: `ncu --section SpeedOfLight --section MemoryWorkloadAnalysis_Tables --kernel-name regex:tr_ -c 1 python ...`

Nsight Systems ([User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html))
is not needed: the harness's `torch.profiler` trace (guide 08) already shows the timeline.
Use it only if the H100 box has it and you want host-side detail.

---

## What you should be able to say after reading

- [ ] Why cutting the ring at `a` is legal (A1, Theorem 2.1), and why our stage 2 costs R³
      where a TT-layer costs r² (A2)
- [ ] The exact opt-in call for more than 48 KB of shared memory, and the CC 9.0 limit (B1)
- [ ] Which thread holds which element in an m16n8k16 fragment, drawn for one stage-2
      tile (B2)
- [ ] Why `red.global.add.f32` rather than `atom` for option A (B2)
- [ ] Whether FP16 with FP32 accumulation runs at full rate on the H100 (B3)
- [ ] How our op gets onto the caller's stream, and where its compile time is reported (D1)
- [ ] Which three `ncu` sections answer "is my kernel bandwidth-bound or compute-bound"
      (D2)
