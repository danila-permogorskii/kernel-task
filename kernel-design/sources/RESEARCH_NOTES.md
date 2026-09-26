# Research notes: prior art and H100 behaviour that matter for our kernel

Web research on 2026-09-25, done through the local research tool (search plus page
reading), with the key claims then checked at the source. Every claim is tagged:

- **[verified]** read at the primary source (official docs, the paper itself, the repo)
- **[secondary]** from a forum, blog or summary site; plausible, not confirmed
- **[idea]** my inference from the sources, not something a source states

Companion to `../KERNEL_IDEA.md` (the design) and `READING_GUIDE.md` (what to read).

---

## 1. Prior art: has anyone built this kernel?

**Short answer: not that we could find.** No published GPU kernel was found that fuses a
tensor-ring (or tensor-train) *linear layer* forward pass into one launch. The closest
work solves neighbouring problems, and each one confirms part of our design.

```
                          fuses a chain    keeps intermediates    avoids layout      applies to
                          of contractions  on-chip                shuffles           our operator?
   ─────────────────────  ───────────────  ─────────────────────  ─────────────────  ──────────────
   FastKron (PPoPP'24)    yes (several     shared memory          yes, by design     R = 1 case
                          Kron factors)                                              (guide 03 step 5)
   CUTLASS ex. 13 (B2B)   2 GEMMs          registers or smem      n/a                2 of our 3 stages
   FlashAttention-2/3     2 GEMMs + softmax registers / smem      register shuffles  pattern only
   TT/TR LLM compression  no (einsum,      no                     no                 same math,
   papers 2023–25         MAC counts)                                                no GPU kernel
   ─────────────────────  ───────────────  ─────────────────────  ─────────────────  ──────────────
   our kernel             3 stages         shared memory          by write order     exactly
```

### 1.1 FastKron: the nearest relative **[verified: abstract]**

Jangda & Yadav, *Fast Kronecker Matrix-Matrix Multiplication on GPUs*, PPoPP 2024,
[arXiv:2401.10187](https://arxiv.org/abs/2401.10187), code: <https://github.com/abhijangda/FastKron>.

- Abstract: existing Kron-Matmul implementations "utilize existing tensor algebra
  operations, such as matrix multiplication, transpose, and tensor matrix
  multiplication", which "prevents several Kron-Matmul specific optimizations". FastKron
  is "independent of linear algebra operations" and reaches "up to 40.7x" on one GPU.
- **[secondary]** It avoids the transpose by writing each output element directly at its
  final index, and fuses several factors in one kernel with intermediates in shared
  memory.
- **Why it matters:** our operator at R = 1 *is* a Kronecker product (guide 03, step 5).
  FastKron's result is the evidence that replacing "GEMM + transpose" chains with a
  purpose-built kernel pays off by large factors. That is our argument against
  `einsum`'s permute copies, made by someone else on real GPUs.
- **Read:** the section on how it avoids transposes, and the one on fusing multiple
  factors. Compare with KERNEL_IDEA §4.2 ("write each stage in the next stage's order").

### 1.2 CUTLASS example 13: back-to-back GEMM fusion **[verified: README]**

<https://github.com/NVIDIA/cutlass/tree/main/examples/13_two_tensor_op_fusion> (BSD-3-Clause, targets sm75/sm80).

Fuses `D0 = relu(A0·B0)` then `D1 = relu(D0·B1 + C1)` in one kernel. Its constraints,
quoted:

- register-resident: `thread_block_tile_N = problem_N` and `warp_tile_N = thread_block_tile_N`,
  so "the operation can be fully register-file-resident";
- otherwise the constraint "can be relaxed if the output accumulator of the 1st GEMM/CONV
  is staged in the shared memory".

**What it tells us:** a GEMM's accumulator can feed the next GEMM from registers only if
each warp owns **whole rows** of the intermediate. **[idea]** In our chain, the index `b`
is an *output column* of stage 1 but a *summed index* (K) of stage 2, so a row of S1 has
to be re-grouped: `(j,t) × (p,b)` → `(p,t) × (j,b)`. That makes **stage 1 → 2 a
shared-memory hop** by necessity. Stage 2 → 3 may be different: check whether S2's rows
`(p,q,t)` can stay in one warp. This is a paper exercise to do before coding.

### 1.3 FlashAttention-2 and -3: the chained-GEMM pattern on Hopper **[verified: abstracts, secondary for details]**

FA-2: [arXiv:2307.08691](https://arxiv.org/abs/2307.08691). FA-3: [arXiv:2407.08608](https://arxiv.org/abs/2407.08608).

- Two GEMMs with the intermediate never written to HBM: the same idea as ours, at much
  larger sizes.
- **[secondary]** FA-3 on H100 uses `wgmma`, TMA, warp specialisation and ping-pong
  scheduling, reaching about 740 TFLOP/s in FP16 (~75% of peak).
- **Why it matters:** it shows what the Hopper-specific path buys at scale, and how
  complex it gets. Our stages are small (M up to 384–3840, K 8–192). **[idea]** Our v1
  should *not* start with FA-3's machinery; stay with `mma.sync` until measurement shows
  stage 2 is FLOP-bound.

### 1.4 Tensor-train / tensor-ring LLM compression papers (2023–2025) **[secondary]**

| Paper | What they report |
|---|---|
| TensorGPT, [arXiv:2307.00526](https://arxiv.org/abs/2307.00526) | TT-compressed embeddings; latency on a Raspberry Pi, not a GPU kernel |
| TT-LoRA, ICMLA 2024, [arXiv:2408.01008](https://arxiv.org/abs/2408.01008) | claims reduced latency against other LoRA variants; no kernel described |
| Saten, Findings of EMNLP 2025 | MAC counts, not measured GPU latency |
| TT decoding on FPGA, AIMS Mathematics 2025 | removes redundant **reshapes** between contractions: our permute problem, on an FPGA |

**Pattern:** these papers count multiply-accumulates and use `einsum`/GEMM. None measures a
fused GPU kernel against dense at small batch. **[idea]** That gap is exactly what our
report fills: a measured answer to "is the factorized layer faster on a real GPU?"

Older, not about linear layers: *cuTensor-TT/TR* (Tensor Workshop 2020) accelerates the
**decomposition** (SVD), not inference. TT-Rec (MLSys 2021) compresses embedding tables.

---

## 2. H100 behaviour that matters, and what to test

### 2.1 First: which H100?

"H100" is more than one product. The five-case table depends on which one we get:

```
                    H100 SXM                 H100 NVL (PCIe card)       older H100 PCIe
   SMs              132                      check on the machine       114 (unverified)
   memory           80 GB, 3.35 TB/s         94 GB, 3.9 TB/s            ~2.0 TB/s (unverified)
   FP16 Tensor      1,979 TFLOP/s sparse     1,671 TFLOP/s sparse       —
                    = ~989 dense             = ~835 dense
   max power        up to 700 W              350–400 W                  350 W (unverified)
```

SXM and NVL columns: **[verified]** from NVIDIA's product page
(<https://www.nvidia.com/en-us/data-center/h100/>, which quotes FP16 *with sparsity*; the dense
rate is half) and the Hopper tuning guide (132 SMs). The older PCIe column is from memory
and is no longer on NVIDIA's page: **[unverified]**.

**Test on day 1:** `nvidia-smi -q` (model, power limit, clocks) and
`torch.cuda.get_device_properties(0)` (SM count, L2 size). Recompute KERNEL_IDEA §6 with
the real numbers.

### 2.2 `mma.sync` tops out near 63% of peak on Hopper **[verified: paper]**

Luo et al., *Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and
Multiple Level Analysis*, [arXiv:2501.12084](https://arxiv.org/abs/2501.12084) (measured on
an **H800 PCIe**):

- "on Hopper Tensor Cores, mma instructions can only attain an average of 62.9% of the
  theoretical peak performance" (§6.2, Table 7);
- `wgmma` exceeds 95% of peak with zero-filled matrices (Table 8), but "a decrease in
  Tensor Core performance when initializing matrices with random values, especially
  pronounced when utilizing FP16 as the computation type and FP32 for accumulation".

**Implications:**

- v1 with `mma.sync` has a ceiling of about 0.63 × peak for stage 2. Put that into the
  model as `eff` and see which cases it decides. **[idea]** Per guide 05's model, only
  R = 16, t = 32 is FLOP-sensitive.
- **Benchmark with random data, never zeros.** Zero-filled matrices overstate Tensor Core
  throughput. Our FP16-input, FP32-accumulate setting is exactly the one that loses most
  with random data.

### 2.3 The L2 is two halves: near and far **[verified: paper, Table 4]**

Same paper, H800 PCIe: L2 near hit **258** cycles, far hit **414** cycles (A100: 208 / 357).
**[secondary]** Chips and Cheese measured the same split on an H100 PCIe and saw clocks
drop to 1395 MHz at the 350 W limit during L2-bandwidth tests:
<https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth>.

**Implications:**

- Global atomics are resolved in L2 **[secondary]**, so atomics to lines in the far half
  cost more. Option A's cost depends on this: measure it (§3, test 4).
- Dense's 11 MB W fits in 50 MB of L2, and the harness repeats calls on the same weight
  (a "warm-cache microbenchmark", `BENCHMARK_NOTES.md`). **Expect dense to beat an
  HBM-based prediction.** Say so in the report rather than be surprised by it.

### 2.4 Vector atomics: `atomicAdd` on `float2` / `float4` **[verified: CUDA Programming Guide]**

From the [atomicAdd reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomicadd):
"float2, float4 on devices of compute capability 9.x and higher, and only supported for
global memory addresses. The atomicity of atomicAdd() applied to vector types … is
guaranteed separately for each of the components".

**Implication for option A** (KERNEL_IDEA §5): each thread adds 4 adjacent outputs with one
instruction, so there are 4× fewer atomic instructions. At R = 16, t = 32 that is 29.5 M →
7.4 M. **[idea]** Arrange stage 3's output fragment so each thread holds 4 consecutive
`(p,q,r)` values.

### 2.5 `stmatrix`: write a fragment to shared memory, transposed if needed **[verified: PTX ISA 9.4 anchor exists]**

[`stmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-stmatrix)
(sm_90) is the store-side twin of `ldmatrix`: a warp writes 8×8 matrix tiles from its
registers to shared memory, with an optional `.trans`.

**[idea]** This is the instruction for KERNEL_IDEA §4.2's key trick: store the stage-1
accumulator into shared memory **directly in stage 2's operand layout**, with the transpose
happening in the store itself. Read the
[fragment figure](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-stmatrix-fragments)
before designing the S1 layout. `ldmatrix.trans` does the same on the load side.

### 2.6 Programmatic Dependent Launch (PDL) **[verified: docs exist; secondary for behaviour]**

[CUDA guide: Programmatic dependent launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html);
PTX [`griddepcontrol`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol).

A kernel launched with `cudaLaunchAttributeProgrammaticStreamSerialization` can start its
prologue (index maths, loading cores into shared memory) while the previous kernel in the
stream is still finishing; it waits at `cudaGridDependencySynchronize()` before touching
the previous kernel's output. **[secondary]** The gain is latency, and it grows as kernels
get shorter and chains get longer.

**Implication:** our call is a short chain (zero workspace → main kernel → convert). PDL
hides part of the per-kernel gap `g`, which guide 06 showed is a real cost. It is an
ordinary launch attribute, **not** graph capture, so it stays within the required-results
rules. **[idea]** Test it in step 4, not v0.

### 2.7 Remove one kernel: "read-and-clear" workspace **[idea]**

Option A needs an FP32 workspace set to zero before each call (a memset: one more op and
one more `g`). Instead, let the **convert kernel** read the FP32 sums, write the FP16
output, and **write zeros back** into the workspace, ready for the next call. Zero it once
in `prepare_optimized`. The per-call chain drops from 3 GPU ops to 2.

Conditions: the workspace belongs to one prepared callable and one stream, is sized for
the largest token count seen (reallocated and re-zeroed when t grows), and is documented as
reusable storage, as `IMPLEMENTATION.md` requires.

### 2.8 Clocks, power and the profiler **[secondary, but widely reported]**

- H100 clocks move with power and temperature. FP16 Tensor Core work on random data is
  where the power limit bites (2.2).
- `nvidia-smi -pm 1` and `nvidia-smi -lgc <MHz>` fix the clocks, **if** we have root on the
  provided machine. We may not; if not, report clock readings alongside results.
- **Nsight Compute locks the GPU to base clock by default** (`--clock-control base`). Its
  kernel times are therefore slower than the harness's. Use `--clock-control none` when
  comparing an `ncu` time with a harness time, or compare ratios only.

### 2.9 Warm vs cold L2 **[secondary]**

The harness measures warm (the same inputs every call). To see the cold case, write a 50 MB
buffer between calls (`torch.empty(L2_size).zero_()` style) and re-measure dense and ours.
Optional: it explains the dense result; it does not change the required numbers.

### 2.10 Larger shared memory needs an explicit opt-in **[verified: tuning guide limits]**

Blocks above 48 KB of dynamic shared memory must opt in with
`cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`,
up to 227 KB on the H100. Our 51–155 KiB budgets (KERNEL_IDEA §4.3) all need it.
Forgetting it gives a launch error, not a slow kernel.

### 2.11 Later, if B's reloads matter: clusters and TMA multicast **[verified: features exist]**

Every block loads all of B (15–60 KiB). A **thread block cluster** could load B once and
share it through distributed shared memory, or TMA can **multicast** one load to all blocks
in the cluster. **[secondary]** DSM is reported several times faster than going through
global memory. This is a v2 optimisation, relevant only if `ncu` shows L2 → SM traffic
limiting the R = 16, t = 32 case.

---

## 3. Tests to run on the H100 before writing the kernel

Each test is small, answers one question, and drives one design decision.

```
   #   test                                        answers                               decides
   ─   ──────────────────────────────────────────  ────────────────────────────────────  ──────────────────────────
   1   nvidia-smi -q; device properties            which H100, SMs, L2, power limit      all model numbers (2.1)
   2   guide 08 on the H100 (park, 2 clocks,       H, g, reference and dense table,      the baseline to beat
       profiler, harness)                          kernels per call
   3   guide 06 step 3 on the H100, RANDOM data    F with FP32 accumulation; does        eff for the model (2.2)
                                                   power throttling show?
   4   atomics microbenchmark: 29.5 M scalar vs    atomic cost at our real pattern       option A vs B vs C, and kc
       7.4 M float4 atomicAdd, 160–320 adders       (2.3, 2.4)
       per output, as our layout would issue them
   5   a 1-warp mma.sync m16n8k16 loop,            achievable mma.sync rate on this      whether v1 needs wgmma
       random data                                 H100 (vs the 62.9% reference)         (2.2)
   6   empty-kernel chain of 3, with and without   the gap g, and how much PDL hides     PDL in step 4 (2.6)
       PDL
   7   host cost of an empty custom op via         the host side of our kernel at t = 1  whether we can beat dense
       torch.library vs F.linear                                                         at t = 1 at all
```

Tests 2 and 3 reuse the lab (guides 06–08); the rest are 20–50 lines each. **[idea]** Test
7 matters most for the headline: if an empty custom op already costs as much host time as
`F.linear`, t = 1 can only tie dense, whatever the kernel does. That is fine under the
brief, but it should be known before any tuning effort goes into t = 1.

---

## 4. Source list

| Source | Type | Status |
|---|---|---|
| FastKron, PPoPP 2024, arXiv:2401.10187 | peer-reviewed | abstract verified |
| Luo et al., arXiv:2501.12084 (Hopper microbenchmarks, H800 PCIe) | arXiv | §6.2, Tables 4, 7, 8 verified |
| Luo et al., *Benchmarking and Dissecting the Nvidia Hopper GPU Architecture*, IPDPS 2024, arXiv:2402.13499 | peer-reviewed | abstract verified; code: <https://github.com/HPMLL/NVIDIA-Hopper-Benchmark> |
| CUTLASS example 13 README | official repo | constraints quoted |
| FlashAttention-2, arXiv:2307.08691; FlashAttention-3, arXiv:2407.08608 | arXiv (FA-3 also NeurIPS 2024) | links verified; details secondary |
| CUDA Programming Guide: atomicAdd, PDL, L2 cache control | official docs | atomicAdd text quoted; pages exist |
| PTX ISA 9.4: `stmatrix`, `griddepcontrol` | official docs | anchors verified |
| Chips and Cheese, *Nvidia's H100: Funny L2, and Tons of Bandwidth* | blog | secondary |
| TensorGPT, TT-LoRA, Saten, AIMS Math 2025 | mixed | secondary (search summaries) |
| NVIDIA forums (launch overhead ~5 µs, atomics resolved in L2) | forum | secondary |

Not found despite searching: a published fused GPU kernel for tensor-ring *linear layers*,
and measured FP32 atomic throughput numbers for the H100. Test 4 produces the second.
