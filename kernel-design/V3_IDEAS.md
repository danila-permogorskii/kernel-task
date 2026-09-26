# v3: thoughts before writing it

Status: **idea, not implemented.** Numbers marked *(estimate)* are predictions.

## 1. What limits v2 (from the H100 measurements)

R16 t32, one block iteration (one k for one block), measured by Nsight Compute:

```
   15.7 k warp-instructions per block iteration, of which ~840 are Tensor Core (5 %)
   8 warps per SM, issue rate 0.34 per cycle per scheduler

   where the non-Tensor-Core instructions go (estimate from the code):
   stage 1, scalar FMAs + FP16 stores           ~3.6 k   ██████████
   stage 2, operand loads + addressing          ~2.9 k   ████████
   stage 2 → S2 staging (store, convert, store) ~1.8 k   █████
   stage 3                                      ~0.4 k   █
   loop / sync / address overhead               the rest
```

So the kernel spends most of its instructions **moving data between registers and shared
memory**, not multiplying. More occupancy did not help (the q split, G3 step 4). The fix is
fewer instructions per piece.

## 2. The idea: one warp per q, stage 2 → stage 3 in registers

```
   v2 (today)                                   v3
   ──────────                                   ──
   stage 2: warps split S2 tiles                stage 2: warp w owns q = w (10 warps, Q = 10)
            → FP32 → staging → FP16 → S2 smem            its B columns [192 × 16] live in
   stage 3: reload S2 from smem                          REGISTERS for the whole kernel
            → Y in registers                             (B does not depend on k)
                                                stage 2 result (16 × 16, FP32 registers)
                                                   ──re-pack in registers──► stage 3 input
                                                stage 3: Y[(t,p), q=w, r] in registers
```

Why it works: with the low-level `mma.sync.m16n8k16` instruction (instead of the WMMA API),
the register layout of a 16×8 FP32 **result** tile is documented, and two neighbouring result
tiles hold exactly the values a thread needs for a 16×16 FP16 **input** tile. Converting is
a few `cvt` instructions in registers. This is the trick FlashAttention-2 uses to feed the
softmax output P straight into the second matrix multiply without touching shared memory.

For one q, a stage-2 output tile (rows (t,p), columns c) *is* a stage-3 input tile (rows
(t,p), K = c). For R = 8 the tile is 16×8 and stage 3 uses `mma.sync.m16n8k8`, whose input
layout matches a single 16×8 result tile.

What disappears per block iteration:

| Removed | Instructions | Shared memory |
|---|---|---|
| S2 staging (store, convert, store) | ~1.8 k | 10 KiB staging + 15 KiB S2 |
| stage-3 operand loads of S2 | ~0.2 k | — |
| stage-2 B operand loads (B in registers) | about half of ~2.9 k | 63 KiB of B (R = 16) |
| one `__syncthreads` per k | — | — |

Smaller shared memory (no B, no S2, no staging) also lets 2–3 blocks share an SM.

## 3. Stage 1 onto Tensor Cores (second part)

Stage 1 is only 4–7% of the FLOPs but ~25% of the instructions. As a matrix product it is
`X[(t,j) × i] @ A1[i × (p,b)]` with K = ni = 8: `mma.sync.m16n8k8` fits exactly. Its result
rows are (t,j), but stage 2 wants rows (t,p). With `mma.sync` each thread knows which (row,
column) each of its result registers holds, so it can **store each value straight to its
S1[(t,p),(j,b)] address**, with no staging. That keeps "write it where the next stage reads
it" and removes ~3 k scalar instructions per block iteration.

## 4. What it could give *(estimate)*

```
   instructions per block iteration   15.7 k  →  ~4–5 k       (3–4× fewer)
   R16 t32 fused kernel               121 µs  →  ~35–60 µs    (15–25 % of Tensor Core peak)
   R8  t1  fused kernel               8.4 µs  →  ~4–5 µs
```

The interesting consequence is at **t = 1, the headline case**. Today design B takes
13.5 µs per call against dense's 10.6 µs, with a 12.1 µs kernel. If the kernel drops to
~4–5 µs, the call becomes limited by host and launch overhead (~8–10 µs), and **a tie with
dense, or a small win, at t = 1 becomes plausible.** It would still lose at R16 t32: dense
reads an L2-resident weight in ~5 µs; the ring must do 25× more arithmetic.

## 5. Why not Hopper's own Tensor Core path (WGMMA, TMA)

WGMMA works on 64-row tiles per warp group and wants large, regular operands fed by TMA. Our
matrices are small and oddly shaped (M = 12·tt rows, N = 16 per q, K = 8–192). Padding them
to 64 rows would waste most of the work, and the code grows several times. `mma.sync` runs
on the H100 at a lower peak than WGMMA, but we are at 7% of peak: the limit is instruction
overhead, not the Tensor Core rate. WGMMA only matters after v3, if at all.

## 6. Risks and cost

| | |
|---|---|
| Correctness | fragment layouts are easy to get wrong; `tools/check_kernel.py` catches it within minutes, and `mma.sync` m16n8k16 / m16n8k8 run on the laptop's sm_86 too, so all debugging is free |
| Scope | v3 only for the two fixed shapes; v2 stays as the generic path (test shapes), unchanged |
| Registers | B strip in registers is 48 per thread (R = 16); with Y and temporaries ~130 → watch for spills in `-Xptxas -v` |
| Effort | ~2–4 h of coding and laptop checks, then one H100 session (~1 h, ~$4 of the ~$21 left) |
| Payoff | uncertain; the realistic best case is a t = 1 tie with dense and 2–3× at R16 t32 |

## 7. Recommendation

Worth doing **after** the report is written with the v2 numbers (so the submission is
complete regardless), and only for the two fixed shapes. Order: stage 2 → 3 chaining first
(biggest instruction cut, and it frees B and S2 from shared memory), measure; then stage 1
on `mma.sync.m16n8k8` if stage 1 then dominates.

## 8. Result (H100 SXM5, 2026-09-26)

Built for t = 1 only (stages 2 → 3 chained in registers, `cp.async` cores, `red.v2`
output). Stage 1 stays scalar. Measured with `tools/h100_v3.sh`; files in
`results/h100/v3/`. All times are the harness's CUDA-stream medians in µs per call, and
every run was on the same instance.

| case | dense | reference | WMMA A | WMMA B | **V3 A** | **V3 B** |
|---|---|---|---|---|---|---|
| R8 t1  | 10.7 | 113.6 | 14.7 | 13.6 | 13.9 | **10.1** |
| R16 t1 | 10.6 | 115.2 | 18.3 | 17.9 | 13.9 | **12.6** |

- **R8 t1, design B: 10.1 µs, against 10.7 µs for dense.** This is the first time the ring
  beats dense at t = 1 (by 6%). The prediction was "a tie or a small win".
- Kernel alone (`measure_kernels`, R8): 4.68 µs for the V3 kernel in design A, against
  8.4 µs for the v2 fused kernel. The §4 estimate was ~4–5 µs.
- The design B kernel takes 9.07 µs. That is V3 plus the last-block tail: one block
  converts all 2880 outputs and clears the workspace. The ~4.4 µs gap is the next
  target. Candidate fixes: per-output-tile counters, so many blocks each finish one tile,
  or a cluster/DSMEM reduction. Either could bring R8 t1 to ~6–7 µs per call.
- R16 t1 still loses to dense (12.6 vs 10.6), but by less than before (17.9 → 12.6).

**The tail, fixed** (`tools/h100_tail.sh`, `results/h100/v3tail/`, same instance, t = 1,
design B, two interleaved rounds, µs per call):

| tail (`TR_V3_TAIL`) | R8 call | R8 kernel | R16 call | R16 kernel | dense call |
|---|---|---|---|---|---|
| 0: one block, one float at a time (as above) | 10.1 / 10.1 | 9.01 | 12.3 / 12.5 | 11.57 | 10.1–11.0 |
| 1: one block, all float4 loads in flight | 8.0 / 8.0 | 6.68 | 10.6 / 10.6 | 9.51 | |
| 2: counter per q chunk + float4 (default) | **7.9 / 8.2** | 6.68 | **10.5 / 10.7** | 9.50 | |

- The loss came from **serial loads**. The old loop did 12 dependent L2 round trips per
  thread, each ~0.2 µs. With the loads issued together that drops to ~1. Splitting the
  work across per-q-chunk blocks (tail 2) adds nothing on top of that: 2–3 small blocks
  instead of one is noise at this size.
- README harness runs with the new default: R8 t1 **8.0 µs vs dense 10.6 (1.3× faster)**;
  R16 t1 10.6 vs 10.4 (a tie). Error is unchanged: max|err| 1.3e-3 and 1.7e-3.
  Correctness: `check_v3` 16/16 ok, `check_kernel` ALL OK, pytest 21 passed.
- What is left in the design B kernel, above the 4.5 µs V3 compute: ~2.2 µs for the
  fence, the counter and the one-round-trip tail.
- Correctness on the H100: `check_v3` passes on all four tilings × A/B, with
  max|err| ≤ 1.8e-3. `check_kernel` and `pytest` are both ALL OK.

*Read later:* PTX ISA, "Matrix Fragments for mma.m16n8k16" (the register layouts);
FlashAttention-2 paper (Dao, 2023), section 3 on keeping P in registers.
