# The kernel: idea, evidence and plan

**Target:** a CUDA C++ kernel for the three-core tensor-ring operator, running on an
NVIDIA H100, called from `prepare_optimized` in `src/factorized_inference/submission.py`.

**Before implementing:** read `sources/READING_GUIDE.md`. It lists the exact sections of
each source and the question each one answers for this design.
`sources/RESEARCH_NOTES.md` has the prior art (nothing identical found) and the H100
behaviours to test before coding (section 3: seven small tests).

**Status:** design. Nothing here is measured on an H100 yet. Every number marked
*(laptop)* comes from the lab guides 06–08 on an RTX 3050 Ti Laptop. Every number marked
*(count)* is exact arithmetic from the shapes. Every number marked *(model)* is a
prediction.

---

## 1. The idea in one sentence

> **Split the ring into many small independent pieces, one per (token, link `a`, input
> digit `k`), whose intermediates fit on-chip. Run all of them in one launch. Write each
> intermediate in the order the next step reads it, so no permute is ever needed. Add the
> pieces' results into `y`.**

```
   REFERENCE (3 einsums)                         OUR KERNEL (1 launch)

   x ─► T1 ─copy─► T1p ─► T2 ─copy─► T2p ─► y     x ═══════════════════════════► y
        └──────── slow memory (HBM) ────────┘        │  many small pieces, each  │
        8 GPU ops, ~100 µs host per einsum           │  lives in shared memory   │
                                                     └───────────────────────────┘
```

In studio terms: the reference records every intermediate track to tape and re-spools
it between desks. Our kernel mixes many small **takes** live inside the desk, and only
the final mix goes to tape.

---

## 2. What the experiments told us

### 2.1 Where the reference's time goes

One reference call at R = 8, t = 1 *(laptop, guide 08 steps 2–3)*:

```
   HOST ┃████ einsum ████┃████ einsum ████┃████ einsum ████┃       ≈ 340–400 µs
   GPU    ▮ ▮▮  ·  ▮ ▮ ▮  ·  ▮ ▮                                   ≈ 80 µs
          3 GEMMs + 5 copies = 8 ops, only 27 µs of real work;
          the rest is gaps between kernels (~7–9 µs each)
```

Across the five cases *(laptop, guide 08 step 5)*:

```
                 t = 1          t = 8          t = 32
   R = 8     host-bound     host-bound      GPU-bound (0.58 ms)
   R = 16    host-bound         —           GPU-bound (2.43 ms)
```

At large t, most of the GPU time is **layout shuffling**, not arithmetic. In the Julia
version of the same chain, the permutes took 80–84% of the time at t = 32 *(laptop, guide
07 step 5)*. PyTorch's copies are faster, but they are still pure traffic.

### 2.2 The model that predicts it

```
   one kernel  =  max( FLOPs / (eff·F) ,  bytes / BW )  +  g
   one call    =  max( host side ,  Σ kernels )
```

It was tested three times *(laptop)*:

| Test | Result |
|---|---|
| 15 real GEMM shapes (guide 06) | ±11% at t = 32, 0.65–1.7× at t = 1 |
| whole Julia chain (guide 07) | 1.4–3.6× off → **one missing parameter** (permute bandwidth) → 0.88–1.03 |
| harness's own JSON (guide 08) | 0.85–1.17, **limiting side correct in 10/10 rows** |

This model is the tool that chooses between the design options in section 5.

### 2.3 What compression costs

*(count)*

```
                  stored (FP16)     FLOPs / token     T1 + T2 / token
   dense W        11.1 MB           11.1 M            —
   ring R = 8     87 KiB            39.8 M            0.68 MB
   ring R = 16    348 KiB          277.2 M            2.70 MB
```

The ring stores 30–120× fewer bytes and does 3.6–25× more arithmetic. The kernel's job
is to make sure the *bytes* saving is not lost again in T1/T2 traffic, copies and host
overhead.

---

## 3. The key observation: which indices are free until the end

```
   stage 1   S1[j,k,a,p,b]  =  Σ_i        x[i,j,k] · A[a,p,i,b]
   stage 2   S2[k,a,p,q,c]  =  Σ_{j,b}    S1       · B[b,q,j,c]
   stage 3   y[p,q,r]      +=  Σ_{k,a,c}  S2       · C[c,r,k,a]
                                  ▲  ▲
                                  │  └── a: the ring link, carried from stage 1 to 3
                                  └───── k: an input digit, carried from stage 1 to 3
```

`k` and `a` are only **carried along** through stages 1 and 2. They are summed only in
stage 3. So for fixed `(t, a, k)` the whole chain is an independent problem:

```
   one piece (t, a, k)                                        size, R = 8 (count)

   x[:, :, k, t] ──A[a,:,:,:]──► S1[j,p,b] ──B──► S2[p,q,c] ──C[:,:,k,a]──► partial y[p,q,r]
      96 values                  1,152 values     960 values                 2,880 values
                                 └──────── a few KB: fits on-chip ────────┘

   y[:, t] = Σ over all (a, k) pieces         R · nk = 160 pieces per token (R = 8)
                                                        320 pieces per token (R = 16)
```

This is guide 03's "cut the ring" (fix `a`), plus a second cut (fix `k`) that the index
structure gives for free.

Where the arithmetic is *(count)*:

```
   stage 1   7%  (R=8)   4%  (R=16)      K = ni = 8           tiny, bandwidth-like
   stage 2  74%  (R=8)  85%  (R=16)      K = nj·R = 96 / 192  the R³ term: Tensor Cores
   stage 3  19%  (R=8)  11%  (R=16)      K = R = 8 / 16       short, many outputs
```

---

## 4. Mapping the pieces onto the H100

### 4.1 Grid: who does what

The first design ("v1"): **one thread block (CTA) per (a, k, token tile)**. A token tile
is `tt` tokens handled together, so the block loads each core slice once and reuses it
`tt` times.

```
   grid = R  ×  nk  ×  ceil(t / tt)              R = 8, t = 1  →  8 × 20 × 1 = 160 blocks
                                                 R = 16, t = 32, tt = 8  →  16 × 20 × 4 = 1,280
   H100: 132 SMs  →  at t = 1, about 1–2 blocks per SM: enough to start everything at once

   ┌─────────────── block (a, k, tile) ───────────────┐
   │                                                   │
   │  load:  x[:, :, k, tile]   A[a,:,:,:]   B   C[:,:,k,a]      (HBM → shared)
   │                                                   │
   │  stage 1  ── Tensor Core MMA ──►  S1   (shared)   │
   │  stage 2  ── Tensor Core MMA ──►  S2   (shared)   │
   │  stage 3  ── Tensor Core MMA ──►  partial y (registers, FP32)
   │                                                   │
   │  add partial y into the output  (section 5)       │
   └───────────────────────────────────────────────────┘
```

### 4.2 Each stage is a small GEMM, and the layouts chain

The trick that removes the permutes: **each stage writes its result in exactly the
layout the next stage reads as a GEMM operand.**

```
   stage 1:  [ (j,t) × i ]  ·  [ i × (p,b) ]       →  S1 stored as  [ (p,t) × (j,b) ]
                                                                         │
   stage 2:  [ (p,t) × (j,b) ]  ·  [ (j,b) × (q,c) ]  →  S2 stored as  [ (p,q,t) × c ]
                                                                         │
   stage 3:  [ (p,q,t) × c ]  ·  [ c × r ]        →  partial y  [ (p,q,r), t ]
```

The reshuffle happens as a side effect of *where each result is written*, in shared
memory, which costs nothing extra. In the reference it is a separate copy through HBM.

GEMM shapes per block *(count)*:

```
                    M (rows)     K (summed)     N (cols)
   stage 1 R=8     12·tt           8             96          K = 8: pad to 16 or use k8 MMA
   stage 2 R=8     12·tt          96             80          the big one
   stage 3 R=8    120·tt           8             24
   stage 1 R=16    12·tt           8            192
   stage 2 R=16    12·tt         192            160
   stage 3 R=16   120·tt          16             24
```

Every K and N is a multiple of 8, and so is M whenever `tt` is even. They tile onto
`mma.sync.m16n8k16` (FP16 inputs, FP32 accumulation) with little padding: M = 12 at
tt = 1 pads to 16, and K = 8 pads to 16 (or uses the k8 shape).

### 4.3 Shared-memory budget

*(count, FP16, per block; the H100 allows up to 227 KiB per block)*

```
                     tt = 1      tt = 8      tt = 32
   R = 8              21 KiB      51 KiB     155 KiB    ✔ all fit
   R = 16             72 KiB     131 KiB     334 KiB    ✘ tt = 32 does not fit → tt ≤ 16
                      └── B is 15 KiB (R=8) / 60 KiB (R=16) of this: every block needs all of B
```

So `tt` is a tuning parameter with a hard ceiling: about 16 at R = 16.

### 4.4 Preparation (done once, allowed, reported)

`prepare_optimized(cores, spec)` repacks the cores into the operand layouts above:

```
   A[a,p,i,b]  →  per a:     [ i × (p,b) ]
   B[b,q,j,c]  →            [ (j,b) × (q,c) ]
   C[c,r,k,a]  →  per (k,a): [ c × r ]
```

This is the same number of values as the cores: **no expansion, no dense W**. It is
reported as "packed factors" (`IMPLEMENTATION.md`, interface requirements).

---

## 5. The one real design decision: how the pieces are added

Every output `y[t, (p,q,r)]` receives one partial result from each `(a, k)` piece. There
are three ways to add them:

```
   OPTION A — atomics                    OPTION B — loop k inside the block     OPTION C — partials + 2nd kernel
   each block atomicAdds into            block (a, tile) loops over all 20 k,   blocks write partials (FP32),
   an FP32 buffer, then a tiny           keeps y's partial in FP32 registers    a small kernel adds them
   kernel converts it to FP16            → 20× fewer atomics

   blocks: R·nk·tiles  (many)            blocks: R·tiles  (few at t = 1!)       blocks: many
   atomics/output: R·nk = 160–320        atomics/output: R = 8–16                atomics: none
   deterministic: no                     deterministic: no                       deterministic: yes
   launches: memset + 1 + convert        memset + 1 + convert                    1 + 1
```

The trade-off in numbers *(count)*:

```
                         atomics per call                     blocks at t = 1
                    one piece/block    block loops k      one piece/block   block loops k
   R = 8,  t = 1        0.46 M            0.02 M               160                8
   R = 16, t = 32      29.5 M             1.5 M              1,280               64
```

Option B has too few blocks at t = 1 (8 blocks on a 132-SM GPU). Option A has 29.5 M
atomics at the largest case. The real answer is probably **in between: each block loops
over a *chunk* of k (`kc` values)**. Then `kc` is one dial that trades parallelism for
atomics:

```
   blocks  = R · (nk / kc) · ceil(t / tt)
   atomics per output = R · nk / kc
```

Choose `kc` and `tt` by **measuring**, with the section 2.2 model to predict which way
each case moves.

All three launches (memset, main kernel, convert) are issued from **one C++ function**.
Host cost is paid per *Python* call, and a C++ launch costs a few µs. Three launches from
C++ are far cheaper than three einsums from Python.

---

## 6. What we expect on the H100 *(model, to be measured)*

```
   the costs that matter                      reference         dense           our kernel
   ─────────────────────────────────────────  ────────────────  ──────────────  ─────────────────
   host: Python ops per call                  3 einsums         1 F.linear      1 custom op
   GPU:  HBM bytes per call, t = 1, R = 8     T1/T2 + copies    11 MB (W)       ~0.1 MB
   GPU:  kernels per call                     8                 1               2–3
   GPU:  FLOPs per token, R = 16              277 M             11 M            277 M
```

Predicted outcome, following guide 05's regimes:

| Case | Prediction | Why |
|---|---|---|
| t = 1, 8 | **≫ faster than reference**; about a **tie with dense** | both one op: host-bound; the H100's 50 MB L2 may also hold dense's 11 MB W warm |
| R = 8, t = 32 | beats reference; close to dense | little work either way |
| R = 16, t = 32 | beats reference; **dense likely wins** | 8.9 GFLOP of real arithmetic vs dense's 11 MB read |

A well-explained tie with dense and a clear loss at R = 16, t = 32 are valid results
under the brief ("beating dense is an objective, not a passing requirement").

---

## 7. The plan

```
   step 0   measure the H100          guide 06/08 method on the H100: H, g, BW, F, eff,
                                      kernels per reference call, the harness table
                │
   step 1   prove the index algebra   the (a, k) piece decomposition in PyTorch (slow
            on the CPU                loops, FP64), checked against the dense oracle
                │
   step 2   CUDA v0: correct          one block per (a, k, tile), plain FP32 loops in
                                      shared memory, option A atomics. Wired into
                                      prepare_optimized; pytest passes
                │
   step 3   CUDA v1: fast stage 2     mma.sync Tensor Cores for stage 2 (74–85% of FLOPs),
                                      then stages 1 and 3
                │
   step 4   tune kc and tt            predict with the model, measure all five cases,
                                      pick per case (or one setting for all)
                │
   step 5   evidence                  harness JSON, token-1 profiler trace showing our kernel,
                                      Nsight Compute on the kernel, report
```

Each step ends with a **measured** result compared against the model's prediction.
Where they disagree, finding out why goes into the report.

### What each step must show

| Step | Pass condition |
|---|---|
| 0 | the five-case reference/dense table on the H100, with the limiting side per row |
| 1 | piece decomposition equals the oracle to FP64 precision |
| 2 | `pytest` passes; FP16 errors within `atol = rtol = 0.02`; kernel visible in the trace |
| 3 | stage 2 reaches a stated fraction of FP16 Tensor Core peak (Nsight Compute) |
| 4 | the table for all five cases, predicted vs measured |
| 5 | `REPORT_TEMPLATE.md` sections 1–5 filled with evidence |

---

## 8. Risks and open questions

- **Atomic throughput.** At R = 16, t = 32 with one piece per block: 29.5 M FP32 atomics
  per call. Unknown on the H100 until measured; `kc` is the fix.
- **Parallelism at t = 1.** 160–320 blocks for 132 SMs is fine; with a large `kc` it is
  not. The model says t = 1 is host-bound anyway, which may make this a non-issue.
- **Loading B.** Every block loads all of B (15–60 KiB). With 1,280 blocks that is
  ~77 MB of L2 → shared-memory traffic at R = 16, t = 32. L2 should serve it, but it is a
  number to watch.
- **Host cost of the custom op.** How a PyTorch C++ extension is called decides the host
  side at t = 1. Measure it exactly like guide 08, step 2.
- **Determinism.** Atomics make the order of additions vary, so results differ in the
  last bits between runs. Allowed by the tolerance; say so in the report.
- **mma.sync vs WGMMA.** `mma.sync` (Ampere-style) runs on the H100 but does not reach
  Hopper's full Tensor Core rate; WGMMA/TMA do, at a large jump in complexity. Only worth
  it if step 4 shows R = 16, t = 32 is limited by stage 2's FLOPs.

---

## 9. What we will not do

- Construct or keep a dense W in any layout. The pieces never form a W tile.
- Capture a CUDA graph in the required runs (optional, in a separate section).
- Change the harness, the tolerances or the workloads.
- Claim a number that was not measured on the H100.
