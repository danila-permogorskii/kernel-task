# G4: the evidence, and design A vs B

What was measured on the H100 SXM5 (session 1, 2026-09-26), where each number lives, and
what to say about it.

---

## 1. The required harness runs (`results/h100/{A,B}/rank{8,16}.json`)

These are the README's two commands, run once with `TR_DESIGN=A` and once with `B`
(`--device cuda:0`: torch 2.14 rejects `--device cuda`, disclose it).

**CUDA-event stream latency, µs per call** (the headline metric):

| Case | Dense | Reference | Ours A | Ours B | Ours vs reference | Ours vs dense |
|---|---|---|---|---|---|---|
| R8, t1 | 10.6 | 112.0 | 14.2 | **13.5** | 8.3× faster | 1.3× slower |
| R8, t8 | 10.4 | 119.0 | 26.0 | **25.2** | 4.7× faster | 2.4× slower |
| R8, t32 | 10.7 | 118.7 | **50.4** | 56.6 | 2.4× faster | 4.7× slower |
| R16, t1 | 10.8 | 118.9 | 19.0 | **18.1** | 6.6× faster | 1.7× slower |
| R16, t32 | 10.7 | 212.0 | **126.6** | 132.9 | 1.7× faster | 11.8× slower |

(ratios use the better of A and B; the reference column is from the A run)

**Host latency, memory, preparation** (A run):

| Case | Host µs: dense / ref / ours | Extra resident memory: dense / ours | Our preparation |
|---|---|---|---|
| R8, t1 | 18.1 / 115.8 / 24.4 | 33.6 MB / 0.21 MB | 99 ms |
| R16, t32 | 18.1 / 245.6 / 137.9 | 33.6 MB / 0.44 MB | 100 ms |

Our preparation (~100 ms per process) is loading the already-compiled extension and packing
the cores; the first compile of the extension (~1 min) happens once per machine, in
`tools/h100_setup.sh`, and is reported separately. Correctness: max absolute error 0.0013 to
0.0023, relative L2 error 3.5e-4, in every case, both designs (tolerance 0.02).

### What it says

```
   t = 1:   reference ┃████████████████████████ 112 µs: 3 einsums, 8 GPU ops, host-bound
            ours      ┃███ 14 µs:                 1 custom op, 1–3 GPU ops
            dense     ┃██ 11 µs:                  1 F.linear, 1 GPU op
```

- **Against the reference we win everywhere**, most at t = 1, where the reference's cost is
  three einsums worth of Python and eight kernels, not arithmetic.
- **Against dense we lose everywhere.** Dense is one cuBLAS call reading an 11 MB weight,
  which in this repeated-call benchmark likely stays in the H100's 50 MB L2: ~5 µs of GPU
  and ~10.5 µs per call at every size. At
  t = 1 we are within 3–8 µs of it; at R16 t32 the ring does 25× more arithmetic than dense
  (277 M vs 11 M FLOPs per token), and we reach 7.4% of Tensor Core peak, so dense wins by
  11×. The brief allows this: beating dense is an objective, not a requirement.
- **Memory: 40–160× less extra memory than dense** (0.21–0.81 MB vs 33.6 MB).

---

## 2. The token-1 traces (`traces/h100/{A,B}/rank{8,16}/*.json`)

Open in https://ui.perfetto.dev (drag the file in). Each file holds three profiled calls.
Per call on the GPU row:

```
   factorized_reference   elementwise │ nvjet GEMM │ elementwise │ elementwise │ nvjet GEMM │ ...  8 ops
   dense                  nvjet GEMM                                                          1 op
   ours, design A         vectorized_elementwise (fill) │ tr_ring_fused_kernel<false, Shape<…>> │
                          tr_ring_convert_kernel                                              3 ops
   ours, design B         tr_ring_fused_kernel<true, Shape<…>>                                1 op
```

`nvjet_sm90_*` is cuBLAS's Hopper GEMM; `elementwise_kernel` are the reference's re-layout
copies. **`tr_ring_fused_kernel` is our kernel**: its template arguments show the design
(`false` = A, `true` = B) and the fixed shape it ran.

---

## 3. Kernel-level numbers (`results/h100/kernels.json`)

From `tools/measure_kernels.py`, all in one process:

```
   floors   empty extension call (host)      0.1–0.3 µs
            empty kernel, GPU time           0.87 µs
            empty kernel, stream time        ~3 µs      ← the cost of one launch, seen from the stream

   fused kernel, % of FP16 Tensor Core peak (989 TFLOP/s):  0.5 % (R8 t1) … 7.4 % (R16 t32)
   torch.compile of the reference (bonus, default mode, no CUDA graphs):
            106–240 µs per call: host-bound like the reference; our kernel is 2–8× faster
```

t = 1 is a latency problem, so % of peak is not the right score there: the fused kernel takes
8.4 µs (R8) against a 0.87 µs empty kernel, and the whole call ~14 µs against a ~3 µs launch
floor. R16 t32 is the throughput case; there the score is 7.4% of Tensor Core peak.

---

## 4. Design A vs B: what the traces show

```
   A  ┃fill 1.05┃ gap ┃██ fused 8.4 µs ██┃ gap ┃convert 1.1┃     3 launches, GPU busy ~10.6 µs
   B  ┃████████ fused 12.1 µs ████████████┃                     1 launch,  GPU busy ~12.1 µs
                              └── +3.7 µs: every block fences and bumps a counter,
                                  then the LAST block converts all outputs alone
```

| Case | A | B | B − A |
|---|---|---|---|
| R8, t1 | 14.2 | 13.5 | **−0.7 µs (5 % faster)** |
| R16, t1 | 19.0 | 18.1 | **−0.9 µs (5 % faster)** |
| R8, t8 | 26.0 | 25.2 | −0.8 |
| R8, t32 | 50.4 | 56.6 | **+6.2 (12 % slower)** |
| R16, t32 | 126.6 | 132.9 | **+6.3 (5 % slower)** |

- B removes two launch gaps but makes its one kernel longer. Net: about **1 µs saved at
  t = 1**, and a **loss at t = 32**, where the last block's conversion is a serial tail that
  grows with the token count.
- B's production risks (`kernel-design/DISCUSSION_POINTS.md`) are unchanged: state kept
  between calls, one workspace per stream, a dirty workspace after a failed call.
- **Decision: A is the default; B is a documented, measured experiment.** The predicted gain
  was "a few %"; measured: 5% at t = 1, a loss at t = 32. That matches the business
  conclusion: B's gain is second-order, its risk is first-order.
- B could shrink its tail (let the last few blocks share the conversion), but the most it can
  ever save is the two launch gaps, ~2–3 µs per call.

---

## 5. What to say in the discussion (one line each)

1. **Where the reference's time goes:** host-side einsums and 8 kernels, 5 of them copies; at
   t = 1 the arithmetic is 0.04 µs.
2. **The idea:** cut the ring into independent (a, k) pieces that fit on-chip; write every
   intermediate in the order the next stage reads it, so no copy kernel exists.
3. **The result:** 1.7–8.3× faster than the reference, 40–160× less extra memory than dense,
   slower than dense everywhere, most at R16 t32 where the ring does 25× more arithmetic.
4. **Why only 7.4% of peak:** most executed instructions are not Tensor Core work (stage 1,
   the S2 round trip, operand loads) with 8 warps per SM; v3 would attack exactly that.
5. **Before claiming an improvement in a full system:** measure under CUDA graphs (the launch
   gaps that separate A and B mostly disappear), with realistic batch and L2 contention
   (dense's 11 MB weight will not stay in L2 next to a whole model), and per-layer, not per
   operator.
