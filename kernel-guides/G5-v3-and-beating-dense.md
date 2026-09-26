# G5: from "1.3× slower than dense" to "1.3× faster" at t = 1

What changed after the v2 report (G3, G4), step by step, with the measurement behind every
step. Then an honest reading, what is left, and where the submission stands.

```
   G1  overview: what runs where
   G2  kernel walkthrough: tr_ring.cu, section by section
   G3  the optimisation journey: v0 → v2, every step measured
   G4  evidence: harness, traces, A vs B, what to say about it
   G5  v3 and the tail: how t = 1 got past dense, and what is left   ◄── you are here
```

All numbers: H100 SXM5, FP16, CUDA-event stream latency per call from the README harness,
unless marked *kernel* (profiler time of our kernels only, `tools/measure_kernels.py`).
Three instances were used on 2026-09-26. Comparisons are only made **within one
instance**; the instance is named where it matters.

---

## 0. The result in one picture

```
   R8, t = 1, µs per call (lower is better)

   reference (einsum)        ████████████████████████████████████████████████████  112
   v2, design A              ███████                                               14.2
   v2, design B              ██████▌                                               13.5
   dense (cuBLAS)            █████                                                 10.6
   V3, design B, old tail    ████▊                                                 10.1
   V3, design B, new tail    ███▊                                                   8.0   ◄ now
```

| case | dense | v2 B (morning) | **now, design B** | ring cores vs dense W |
|---|---|---|---|---|
| R8, t = 1  | 10.6 | 13.5 | **8.0** (1.3× faster than dense) | 89 KB vs 11 MB (124× smaller) |
| R16, t = 1 | 10.4 | 18.1 | **10.6** (a tie)                  | 356 KB vs 11 MB (31× smaller) |
| R8, t = 32 | 10.1 | 56.6 | 56.7 (5.6× slower)                | |

Error is unchanged: max |err| 1.3e-3 (R8) and 1.7e-3 (R16), tolerance 0.02.

---

## 1. Where we started (the v2 report)

- One fused kernel per call: stage 1 on CUDA cores, stages 2 and 3 on Tensor Cores (WMMA).
- R8 t1: 13.5 µs per call (B), with a **12.1 µs kernel**. Dense: 10.6 µs.
- Nsight Compute: the kernel is **instruction-bound**. Only ~5% of instructions are Tensor
  Core ones. Most of the rest are operand loads, stage 1 FMAs, and the S2 round trip
  through shared memory.

---

## 2. Step 1: V3, stages 2 → 3 chained in registers

**Idea** (the FlashAttention-2 trick, `kernel-design/V3_IDEAS.md`). With WMMA, the result of
stage 2 (S2) must be stored to shared memory and loaded back as the input of stage 3. With
PTX `mma.sync`, each thread knows exactly which rows and columns its accumulator registers
hold. That layout is the same one the next `mma` expects for its A operand. So S2 is packed
to FP16 **in registers** and fed straight into stage 3:

```
   v2   stage 1 ─► S1 (smem) ─► stage 2 WMMA ─► S2 (smem) ─► stage 3 WMMA ─► Y ─► atomicAdd
                                                  ▲ store + sync + load, every k
   V3   stage 1 ─► S1 (smem) ─► stage 2 mma.sync ─► S2 in registers ─► stage 3 mma.sync ─► Y ─► red.v2
                                  m16n8k16                 (FP32 → FP16 pack)    m16n8k8 (R8) / k16 (R16)
```

Other changes in the same kernel (`tr_ring_fused_v3_kernel`, `csrc/tr_ring.cu`):

- **One warp per q value.** Each warp holds its whole Y tile in registers for all k of the
  block.
- **Cores loaded with `cp.async`.** The copies run while the threads convert x from FP16
  to FP32.
- **Output with `red.global.add.v2.f32`.** Two floats per atomic, fire-and-forget.
- **Scope:** t = 1 only, the two real shapes only, and tilings compiled in: R8 (2,4),
  (2,5); R16 (4,4), (4,5). Everything else runs the v2 kernel. `TR_V3=0` switches V3 off.

**Measured** (second instance, `results/h100/v3/`, same machine for every row):

| R8 t1 | kernel | call A | call B |
|---|---|---|---|
| v2 (WMMA) | 8.4 | 14.7 | 13.6 |
| V3 | **4.7** | 13.9 | **10.1** |

- The kernel is 1.8× faster. The estimate in V3_IDEAS §4 was "~4–5 µs"; the result is 4.7.
- Design A barely moved (14.7 → 13.9). Its call is 3 launches: zero the workspace, the
  kernel, convert. The gap between launches dominates, not the kernel.
- Design B gained 3.5 µs, but its kernel took **9.1 µs**, not 4.7. Something after the
  computation cost 4.4 µs.

---

## 3. Step 2: the design-B tail

In design B, every block adds its part into an FP32 workspace. The **last** block to finish
(found with a counter) converts all 2,880 outputs to FP16 and clears the workspace for the
next call.

**The bug was in the loop, not the idea.** Each thread did
`v = load(ws[e]); y[e] = half(v); ws[e] = 0` for 12 values in turn. Each load goes to L2
(~0.2 µs), and the compiler did not start the next load before the previous store. So the
loads were **12 dependent round trips**:

```
   old   load ─► store ─► load ─► store ─► … 12 times        ≈ 12 × 0.2 µs + counter
   new   load load load (float4, all issued) ─► store store store   ≈ 1 round trip + counter
```

Three versions, compared on one instance (`TR_V3_TAIL`, `tools/h100_tail.sh`,
`results/h100/v3tail/`, two interleaved rounds):

| tail | R8 call | R8 kernel | R16 call | R16 kernel |
|---|---|---|---|---|
| 0: one block, one float at a time (old) | 10.1 / 10.1 | 9.01 | 12.3 / 12.5 | 11.57 |
| 1: one block, all float4 loads in flight | 8.0 / 8.0 | 6.68 | 10.6 / 10.6 | 9.51 |
| 2: one counter per q chunk + float4 (**default**) | **7.9 / 8.2** | 6.68 | **10.5 / 10.7** | 9.50 |

- **Almost all the gain comes from issuing the loads together** (tail 1). Splitting the
  finish over 2–3 per-chunk blocks (tail 2) adds nothing measurable at this size. It stays
  the default because it scales better if the output grows.
- **Correctness on the H100, before measuring.** `tools/check_v3.py` passes 16/16: every
  tiling × {A, B tail 0/1/2}, three calls each on one object, checking that B leaves the
  workspace clean. Also `tools/check_kernel.py` ALL OK and `pytest` 21 passed.

---

## 4. Step 3 (negative): dependency counters in the stack

This belongs to the weight-stationary side study (`kernel-design/WEIGHT_STATIONARY.md`): all
layers of a model run in **one persistent kernel**, with a grid barrier between layers.
Blocks spend ~30% of the time waiting at that barrier. Megakernel papers replace barriers
with **dependency counters**: a block waits only for the blocks whose output it reads.

**Why it cannot work in the current cut.** A unit of layer l+1 reads a k chunk of its input.
That is an r chunk of layer l's output, and *every* unit of layer l writes all r. So every
unit depends on the whole previous layer, and a counter there is just a barrier. To help,
layer l+1 would have to be re-cut so that each unit depends on one q chunk of layer l
(1/nqc of the layer). That is hours of work.

**So the ceiling was measured first** (mode 9, `kernel_work/stack/dep_bench.py`): today's
units, but each one waits exactly as it would after the re-cut. There is also a control:
the same counters with every group awaited, which must equal a barrier. Numbers are µs
per layer, R8:

```
   grid.sync                        8.00
   monotonic barrier                8.37
   counters, wait all (control)     8.37    ◄ machinery costs nothing
   counters, wait one group         8.00    ◄ the ceiling: 0 gain vs grid.sync
```

(R16: best barrier 13.77 → 13.59, a 1.3% gain.)

**Why so little.** Each group of 80–96 units is spread over all 132 SMs. Its slowest unit is
almost as slow as the slowest of the whole layer. The waiting is **work imbalance between
units**, not the barrier. Dropped; the next step there is to find *which* units are slow.

---

## 5. The stack, for completeness

The weight-stationary stack went from **12.3 to 7.35 µs per layer** (R8 t1; second instance).
The main steps were:

- two-phase loads and a monotonic barrier: 10.7;
- compile-time tiling: 9.3;
- V3: 7.8;
- `cp.async`: 7.35.

For comparison: a dense cuBLAS chain under a CUDA graph runs at 7.7 µs per layer, our own
persistent dense stack at 8.1, and dense's HBM floor is 3.3. Full table in
WEIGHT_STATIONARY §8.

---

## 6. Honest reading: what the 8.0 µs is made of

| R8 t1, µs | dense | V3 B, new tail |
|---|---|---|
| GPU: main computation | 5.0 | 4.5 |
| GPU: finishing (fence, counter, convert, clear) | – | 2.2 |
| launch / host gap (stream − kernel) | 5.6 | 1.3 |
| **call** | **10.6** | **8.0** |

- **Kernel against kernel, dense is still faster** (5.0 vs 6.7 µs). Our call-level win
  comes partly from a **shorter path from Python to the GPU**: one extension call and one
  launch, against torch's `matmul` dispatch plus cuBLAS. The required harness measures
  exactly that path (no CUDA graphs), so the result is valid. But under CUDA graphs a
  single dense layer would likely win again. That comparison is optional and not done.
- **The fair GPU-only comparison is the stack** (no launches between layers): ring
  7.35 µs per layer against 7.7 for cuBLAS under a CUDA graph, so a small win. A
  state-of-the-art dense megakernel (60–78% of HBM bandwidth, ~4.5 µs per layer) would
  beat both.
- **Where the ring really has an edge** is memory the benchmark does not stress. Each
  layer's cores are 89 KB (R8). A 128-layer model's cores (11 MB) fit in the 50 MB L2,
  while its dense weights (1.4 GB) must stream from HBM on every token. This
  microbenchmark re-reads one 11 MB weight, which stays in L2 and flatters dense.
- **Where it loses:** more tokens. The ring does 3.6× (R8) to 25× (R16) more arithmetic per
  token than dense, so from t ≈ 8 the GPU is busy and dense wins (25 vs 11 µs at R8 t8).

---

## 7. What is left (ordered by value for the submission first, then by expected gain)

1. ~~Report and final session~~ **done** (fourth instance, `results/h100/{A,B}`,
   `REPORT_TEMPLATE.md`). The README commands were run for B and A on the final code. The
   token-1 traces show one `v3::tr_ring_fused_v3_kernel<true, 2, 8, 2, 4>` per call.
   Nsight Compute for V3 at R8 t1: 0.92 M executed instructions (v2: 2.88 M), 40 registers
   (v2: 127), 23% achieved occupancy, ~19 cycles per issued instruction. It is now
   latency-bound rather than instruction-bound.
2. ~~Pick the default design~~ **done: B everywhere.** Its caveat stays: one workspace per
   prepared object, so no concurrent calls from two streams; `TR_DESIGN=A` is the
   stateless option.
3. ~~The parallel tail for the v2 kernel's design B~~ **done.** It issues 8 loads before
   any store; plain loads rather than float4, since generic shapes need not be multiples
   of 4. At R8 t32, B went from 56.6 µs (6 µs behind A) to 49.6 µs (A: 51.0).

Final numbers, µs per call: R8 t1 **7.9** (dense 10.3), R8 t8 23.3, R8 t32 49.6, R16 t1
**10.5** (dense 10.4), R16 t32 126.2.
4. **V3 for t > 1.** Stack tokens into the 16-row `mma` tiles (12 rows p per token today)
   to attack t = 8, where the gap to dense is 2.4×.
5. **Stage 1 on `mma.sync m16n8k8`** (V3_IDEAS §3). It is ~25% of instructions.
6. **The ~2.2 µs finish.** Try `atom.acq_rel` instead of fence + atomic, or a
   cluster/DSMEM reduction instead of the global workspace.
7. **Optional, fairness:** a CUDA-graph comparison for all methods (section 6), and the
   stack against a dense megakernel.

---

## 8. Is the assignment done?

The README's bar, point by point:

| requirement | status |
|---|---|
| Profile the reference, name the bottleneck | ✓ report §2 (host-bound einsum at t=1; 8 kernels, 5 copies) |
| Own GPU kernel, substantive, all five required cases | ✓ `tr_ring.cu`: V3 at t = 1, v2 WMMA kernel elsewhere |
| Correctness vs reference and dense, tolerances unchanged | ✓ all cases, max err ≤ 0.0023; `check_kernel`, `check_v3`, pytest |
| Explain results, memory, limitations | ✓ report rewritten for V3, the tail and B as default |
| Raw results, environment, profiler artifacts, commands | ✓ one final session: harness A/B, traces, kernels.json, ncu of V3 and v2 |
| AI disclosure, own verification | **the "independently verified" line is still to be filled in by you** |
| Beating dense | "an objective, not a passing requirement". Now met at R8 t1 (design B) |

**Verdict.** The assignment was already complete with v2: a correct kernel, all cases,
a well-supported negative result, and a report. With V3 and the fixed tail, the headline
turned positive at R8 t1. The report, the table, the traces and the profiles now all come
from the final code on one machine.

**What is left before sending:**

1. Fill in your "independently verified" line in the report.
2. Confirm the GPU hours in the Verda billing.
3. Read the report once end to end, so you can defend every number in the 45-minute
   discussion. G5 §6 is the part they are most likely to press on: kernel against
   kernel, and CUDA graphs.
