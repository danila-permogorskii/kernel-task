# G2: kernel walkthrough (`src/factorized_inference/csrc/tr_ring.cu`)

Read with the file open next to it. Line numbers refer to commit `8f5c392`. Every section
starts with the picture, then says which lines implement it and why they look that way.

---

## 0. Prepare, once per weight set (`tr_kernel.py:53`, `pack_cores`)

The kernel wants each core in the layout its stage reads as a matrix operand. Packing happens
once in `prepare_optimized`, so the call itself never permutes anything.

```
   core            packed as                      shape (R = 16)         used by
   A[a,p,i,b]  →  A1[a][i][(p,b)]                [16][8][192]           stage 1
   B[b,q,j,c]  →  B2[(j,b)][(q,c)]               [192][160]             stage 2
   C[c,r,k,a]  →  C3[k][a][c][r], padded to 16   [20][16][16][32]       stage 3
```

Same values as the cores plus zero padding to multiples of 16 (the Tensor Core tile edge).
No dense W, no expansion. The padding is the only extra storage: C3 grows from 24 to 32
columns for r.

---

## 1. Compile-time shapes (lines 44–63)

```
   Shape<ni, nj, P, Q, Rr, R>        RealR8  = <8, 12, 12, 10, 24,  8>
   0 = "read it at runtime"          RealR16 = <8, 12, 12, 10, 24, 16>
                                     Generic = <0, 0, 0, 0, 0, 0>
```

The kernel is a template. For the two real workloads every size is a constant, so
`e / 12` becomes a multiply-and-shift and loops over `i < 8` unroll. Every other shape (the
small test shapes) runs `Generic` with the same code and runtime sizes. `dispatch` (line 357)
picks the variant. G3 shows what this bought: 63 M → 40 M instructions per call.

`Dims` (line 44) carries the runtime sizes and the padded row strides (section 3).

---

## 2. The shared-memory map (lines 65–82)

```
   one block, R16 t32 tuning (kc = 20, tt = 4, qc = 10): 168 KiB
   ┌────────────────────────┬──────┬──────────┬──────────┬────────┬────────┬───────┐
   │ B slice  [192 × 168]   │ A[a] │ C × kc   │ x slice  │  S1    │  S2    │ stage │
   │ FP16  63 KiB           │ FP32 │ FP16     │ FP32     │ 48×200 │ 48×160 │ 8×16× │
   │                        │ 6 KiB│ 25 KiB   │ 30 KiB   │ 19 KiB │ 15 KiB │ 20 f32│
   │                        │      │          │          │        │        │ 10 KiB│
   └────────────────────────┴──────┴──────────┴──────────┴────────┴────────┴───────┘
     loaded once per block ────────────────────────────┘  reused every k ─────────┘
```

`smem_layout` is `__host__ __device__`: the host uses it to ask for the right amount of
dynamic shared memory, the kernel uses the same function to find each buffer. One source of
truth, so they cannot disagree.

---

## 3. Load, once per block (lines 127–173)

```
   global (L2)                              shared
   B2[:, q0*Rc : (q0+qc)*Rc]  ──int4──►    sB   (padded rows: +8 halves)
   C3[k0..k0+kc, a]           ──int4──►    sC   (padded rows: +8 halves)
   A1[a]                      ──FP16→FP32► sA
   x[t0.., :, :, k0..]        ──FP16→FP32► sX   (zero outside the valid tokens / k's)
   zero: S1 padding;  zero: Y accumulators (registers)
```

- **16-byte copies (`int4`, 8 halves).** All 256 threads issue their loads at once, so the
  ~0.5 µs memory latency is paid once, not once per step.
- **Padded rows.** Row strides are `+8` halves (16 bytes) wider than the data. Without this,
  a Tensor Core tile's 16 rows start at addresses 384 or 128 bytes apart, all in the same
  shared-memory bank, and are served one at a time. This single change took R16 t32 from
  430 µs to 176 µs (G3).
- **q slice.** A block needs only the columns of B for its q chunk (`col < valid8` guard
  zero-fills the ragged last chunk).

---

## 4. The k loop: three stages per k (lines 175–262)

```
   for kk in 0 .. kc-1:                       everything below is on-chip
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ stage 1  CUDA cores   sX, sA            ──► S1 [(t,p) × (j,b)]   FP16, shared   │
   │   __syncthreads                                                               │
   │ stage 2  Tensor Cores S1 @ sB           ──► S2 [(t,p) × (q,c)]   FP16, shared   │
   │   __syncthreads                                                               │
   │ stage 3  Tensor Cores S2 @ sC[kk]       ──► Y  [(t,p,q) × r]    FP32, REGISTERS │
   │   __syncthreads                          (Y is summed over all kk)            │
   └──────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1 (lines 177–214): register tile, written in stage 2's layout

```
   one thread = 4 j × 4 b for one (t, p):

   for i in 0..7:   xv = x[t, i, j0..j0+3]    (1 float4 load)
                    av = A[i, p, b0..b0+3]    (1 float4 load)
                    acc[4][4] += xv ⊗ av      (16 FMAs)
   write acc as FP16 into S1[row = t*P + p][col = j*R + b]
```

Two loads feed 16 multiply-adds; the naive version was one load per multiply-add. The result
goes **straight into the layout stage 2 reads** (rows `(t,p)`, columns `(j,b)`): this is the
permute that the reference does as a separate copy kernel, done for free by choosing where
to write. The `else` branch is the generic one-value-per-thread version for odd shapes.

### Stage 2 (lines 216–239): Tensor Cores, WMMA 16×16×16

```
   S2 tile [16 × 16] = Σ_kt  S1[16 × 16] @ B[16 × 16]         FP16 in, FP32 accumulate

   warp w takes output tiles w, w+8, w+16, ...
   then: FP32 tile ──store──► per-warp staging (16 × 20 floats) ──convert──► FP16 into S2
```

`wmma::load_matrix_sync / mma_sync / store_matrix_sync` is CUDA's portable Tensor Core API:
one warp cooperatively multiplies 16×16 tiles. The staging step exists because stage 3
needs S2 as an FP16 *input*, and a FP32 result fragment cannot be re-used as an input
fragment through the WMMA API. **This round trip is what v3 would remove.**

### Stage 3 (lines 241–262): Tensor Cores into register accumulators

```
   S2 rows (t,p) are Q blocks of Rc values:   [ q0: c0..c15 | q1: c0..c15 | ... ]
   read the same bytes as a matrix [(t,p,q) × c] with row stride Rc   ← second free re-layout

   Y[(t,p,q) × r] += S2[(t,p,q) × c] @ C[c × r]      Y stays in yacc[] registers for all kk
```

Each warp owns up to `kMaxYTiles = 8` Y tiles (`yacc[8]`, line 168) for the whole k loop. v1
kept Y in shared memory and loaded/stored it every k; moving it to registers freed 69 KiB of
shared memory and removed that traffic (G3).

---

## 5. Add into the workspace (lines 264–282)

```
   yacc tile ──store──► staging ──► atomicAdd(ws[t, (p, q0+ql, r)], value)
                                    one atomic per output value per block
```

Every block adds its partial sums into an FP32 workspace. The number of atomic adds per
output value is R · ceil(nk / kc): 16 to 80 in the tuned cases (G1 table), against 320 in
the one-piece-per-block v0. Atomics make the order of additions vary between runs, so the
last bits of y can differ run to run; the error stays ~1e-3 against a 2e-2 tolerance.

---

## 6. How the call ends: design A vs B

### A (lines 396–403): three launches, no state

```
   torch::zeros(ws)  →  tr_ring_fused_kernel<false>  →  tr_ring_convert_kernel
   launch 1             launch 2                       launch 3 (FP32 → FP16)
```

The workspace comes from PyTorch's allocator on every call: nothing survives between calls.

### B (lines 284–305): one launch, the last block finishes

```
   every block:   atomics ─► __threadfence ─► __syncthreads ─► thread 0: n = atomicAdd(counter, 1)
   the block that sees n == blocks_per_tile - 1:
                  __threadfence ─► read ws (L2, __ldcg) ─► write y FP16 ─► ws = 0 ─► counter = 0
```

- `__threadfence` makes this block's atomics visible to the whole GPU *before* it announces
  "I'm done". Without it, the last block could read a partial sum.
- `__ldcg` reads through L2, not a possibly stale L1 line.
- The last block leaves the workspace and counter at zero: that is what makes the next call
  correct, and it is also B's production risk (a call that dies midway leaves them dirty).
- The workspace and counters are allocated once and grown on demand
  (`tr_kernel.py`, `PreparedTRKernel.__call__`).

---

## 7. The host side (lines 341–411)

- `launch<kB, S>` opts in to the needed dynamic shared memory once per kernel variant
  (`cudaFuncSetAttribute`), then launches on PyTorch's current stream
  (`at::cuda::getCurrentCUDAStream()`): the output is usable on the caller's stream.
- `tr_ring_forward` checks inputs and picks A or B; `dispatch` picks the shape variant.
- The floors (`empty_launch`, `empty_call`, line 413) exist only for measurement (G4).

*Read later:* CUDA Programming Guide, "Warp Matrix Functions" (WMMA) and "Memory Fence
Functions"; the `threadFenceReduction` CUDA sample (the pattern design B uses).
