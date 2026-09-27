# G6: V3T, more tokens per Tensor Core instruction, and one wave

What changed on 2026-09-27 for t > 1, step by step, with the measurement behind each step.
All numbers: H100 SXM5, design B, µs per call (CUDA events, like the harness) unless marked
*kernel* (profiler time of our kernels only). Sweeps: `tools/v3t_sweep.py`,
`results/h100/v3t/`. Each round is compared **within one instance**.

```
   G5  V3 and the tail: how t = 1 got past dense
   G6  V3T: tokens stacked into the mma, and why "one wave" mattered   ◄── you are here
```

---

## 0. The result in one picture

```
   R8, t = 8, µs per call (lower is better)

   WMMA kernel (the submitted v2 path)   ████████████████████████  23.6
   V3T round 1, best tiling              ████████████████▌         16.4
   V3T round 3, kc = 5, tt = 4           █████████████             12.9   ◄ now
   dense (cuBLAS)                        ██████████                10.1
```

| case | WMMA | V3T (round 3) | gain |
|---|---|---|---|
| R8, t = 8   | 23.6  | **12.9** | 1.8× |
| R8, t = 32  | 49.8  | **27.7** | 1.8× |
| R16, t = 8  | 51.6  | **24.2** | 2.1× |
| R16, t = 32 | 126.4 | **82.8** | 1.5× |

(round 3, second instance of 2026-09-27, `sweep_round3.json`)

---

## 1. Where t > 1 stood

At t = 1, V3 (G5) keeps S2 in registers and feeds it straight into stage 3. At t > 1 the
submission still ran the WMMA kernel (v2), with S2 going through shared memory. R8 t8 was
23.6 µs against dense's 10.1.

V3 also wastes rows. An `mma` tile is 16 rows tall, and at t = 1 a block has only P = 12 rows
(one per p). Four rows of every tile are padding.

## 2. The idea: tokens are more rows

The rows of S1, S2 and Y are (t, p). With several tokens, stack them:

```
   V3 (t = 1)                       V3T (tt = 4 tokens per block)
   m16 tile                         three m16 tiles = 48 rows = 4 tokens x 12 p
   ┌────────────┐                   ┌────────────┐
   │ p 0..11    │                   │ t0 p0..11  │
   │ (4 padding)│                   │ t1 p0..3   │ tile 0
   └────────────┘                   ├────────────┤
                                    │ t1 p4..11  │
                                    │ t2 p0..7   │ tile 1
                                    ├────────────┤
                                    │ t2 p8..11  │
                                    │ t3 p0..11  │ tile 2
                                    └────────────┘
```

- `tt * 12` must be a multiple of 16, so tt = 4 (3 tiles) or 8 (6 tiles). No padding rows.
- B and C are the same for every token. They are loaded into shared memory once per block and
  used for all tt tokens.
- A warp's work item is (q, m tile). The block's `qc * tiles` items are spread over the 8
  warps. Each item keeps its Y tile in registers over the block's k loop, exactly as V3.
- The ending (design B) now has one counter per (token tile, q chunk).

Code: `tr_ring_fused_v3t_kernel` in `csrc/tr_ring.cu`. Correctness:
`tools/check_v3t.py` (FP64 oracle, ragged token counts 2..33, several T on one object, both
designs, and agreement with the WMMA kernel). 20, 30 and 28 configurations passed on the three
H100 rounds; pytest and the older checks stayed green.

## 3. Round 1: it works, but less than predicted

Prediction (before measuring): R8 t8 from 23.6 to 12–15 µs.

| R8 t8, tiling (kc, qc, tt) | call | kernel |
|---|---|---|
| WMMA | 23.6 | 21.9 |
| 1, 5, 4 | 25.2 | 23.7 |
| 2, 5, 4 | 16.9 | 15.0 |
| 4, 5, 8 | **16.4** | 14.8 |

(first instance of 2026-09-27; raw JSON lost, console copy in `session_round1_console.log`)

Better, but not 12–15. Two readings of the numbers:

- **Bigger kc was always better** (more k per block: B loaded once for more work, fewer
  atomics).
- My guess at the time was the `mma` dependency chain. One warp does ~6 dependent `mma` per
  k, and the Hopper floor measurement said a dependent `mma` costs ~24 cycles, 4 independent
  chains ~6.5 each.

## 4. Round 2: the guess was wrong

`mg` = m tiles per work item, so each warp runs `mg` independent chains and loads the B and C
fragments once for all of them.

| R8 t8, (kc, qc, tt, mg) | call |
|---|---|
| 4, 5, 8, **1** | 16.6 |
| 4, 5, 8, 2 | 17.1 |
| 4, 5, 8, 3 | 18.1 |
| 4, 5, 4, 3 | **15.7** |

- More chains did **not** help, and even hurt a little (more registers). The `mma` chain is
  not the limit. Recorded as a negative result.
- What did help: tt = 4 with kc = 4. That means more blocks with more k each.

## 5. Round 3: one wave

| (kc, qc, tt, mg) | R8 t8 | R8 t32 | R16 t8 | R16 t32 |
|---|---|---|---|---|
| R8 4,5,4,3 / R16 4,10,4,1 | 15.5 | 31.7 | 37.2 | 90.5 |
| R8 **5,5,4,1** / R16 **5,10,4,1** | **12.9** | **27.7** | **24.2** | **82.8** |
| R8 10,5,4,3 | 16.5 | 30.7 | | |

R16 t8 went from 37.2 to 24.2 µs by changing kc from 4 to 5. The arithmetic is the same; what
changed is the **number of blocks**:

```
   blocks = R x (20 / kc) x (10 / qc) x (T / tt)

   R16 t8, kc = 4:  16 x 5 x 1 x 2 = 160 blocks     1 block per SM (≈ 150-180 KB smem)
                    132 SMs:  ████████████████████ wave 1 (132)
                              ████                 wave 2 (28)    ← the whole GPU waits
                                                                    for 28 blocks
   R16 t8, kc = 5:  16 x 4 x 1 x 2 = 128 blocks
                    132 SMs:  ███████████████████▌ one wave (128)
```

At R16 (≈ 156 KB of shared memory per block at kc = 4, so one block per SM) this is a true
second wave: **wave quantization**. When a grid is a little larger than one wave, the second
wave costs almost a full block time for a few blocks.

R8 t8 has the same counts, 160 blocks at kc = 4 against 128 at kc = 5, but its blocks are
small (≈ 70 KB), so up to 3 fit on one SM and all 160 start at once. There 28 SMs run two
blocks side by side and finish later. It is the same effect in a softer form: **the call ends
when the busiest SM finishes**, and 128 blocks give every SM at most one.

kc = 10 goes too far the other way: 64 blocks at R8 t8, so half the GPU idles.

The defaults are now `V3T_TILING = {8: (5, 5, 4, 1), 16: (5, 10, 4, 1)}` in
`tr_kernel.py`, used for every t > 1.

## 6. Honest reading

- Dense still wins every t > 1 case. R8 t8 is now 1.3× behind dense (was 2.3×).
- What is left at R8 t8: a 12.9 µs call for ~0.3 GFLOP, about 25 TFLOP/s, 2.5% of the
  Tensor Core peak. The kernel is bound by latency and per-block overhead (loading B and C,
  stage 1 on CUDA cores, the barrier after stage 1, the atomics), not by arithmetic.
- The next steps, by expected value: stage 1 on `mma` (it is now a larger share of the
  work), a persistent grid that picks work items dynamically (no wave tail at all), and
  splitting the block's loads from its compute (double buffering over k).

*Read later:* "wave quantization" / "tail effect" in the NVIDIA CUDA C++ Best Practices Guide
(occupancy and grid sizing) and in the CUTLASS docs (stream-K, which exists to remove it).

## 7. The final session (same code, another instance)

README harness, design B, µs per call (`results/h100/{A,B}`, second instance of 2026-09-27):

| case | dense | ours B | before V3T (2026-09-26 final) |
|---|---|---|---|
| R8 t1   | 9.8 | **8.2**  | 7.9 vs 10.3 |
| R8 t8   | 9.9 | **12.3** | 23.3 vs 10.4 |
| R8 t32  | 9.9 | **27.9** | 49.6 vs 10.6 |
| R16 t1  | 9.9 | 10.7     | 10.5 vs 10.4 |
| R16 t32 | 9.9 | **82.4** | 126.2 vs 10.3 |

t = 1 did not change (same V3 kernel); the ratios moved by instance-to-instance noise
(R8 t1 1.2× instead of 1.3× faster; R16 t1 1.08× slower instead of a tie).
