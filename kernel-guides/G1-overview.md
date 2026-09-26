# G1: the kernel in one picture

Read this first. It is the map; G2 walks through the code, G3 tells how it got fast, G4
shows how to read the evidence.

```
   G1  overview: what runs where                   ◄── you are here
   G2  kernel walkthrough: tr_ring.cu, section by section
   G3  the optimisation journey: v0 → v2, every step measured
   G4  evidence: harness, traces, A vs B, what to say about it
```

---

## 1. What one call computes

```
   x [T tokens × 1,920]  ──►  y [T × 2,880]          y[t,p,q,r] = Σ_{a,k} piece(a,k)

   one piece (a, k), from K0:
   x[t,:,:,k] ─stage 1─► S1 ─stage 2─► S2 ─stage 3─► part of y
               × A[a]          × B            × C[:,:,k,a]
               sum i           sum j,b        sum c
```

There are R · nk = 160 (R = 8) or 320 (R = 16) pieces per token. They are independent until
the final sum, so they can run anywhere, in any order.

In studio terms: every piece is one **take**; the kernel mixes the takes live inside the desk
and only the final mix goes to tape (HBM).

---

## 2. Who does what: the grid

One **thread block** (256 threads = 8 warps) owns a rectangle of the work:

```
   block (k-chunk, q-chunk, a, token tile)

      a        one ring link                          grid.y  = R
      k chunk  kc input digits, done one after another grid.x  = ceil(nk/kc) · ceil(Q/qc)
      q chunk  qc output digits  (stage 3 treats q separately, so q can be split)
      tokens   tt tokens                              grid.z  = ceil(T/tt)
```

The tiling (kc, tt, qc) was tuned on the H100 per case
(`H100_TUNED` in `src/factorized_inference/tr_kernel.py:72`):

| Case | kc | tt | qc | blocks | shared mem / block | atomics per output value |
|---|---|---|---|---|---|---|
| R8, t1 | 2 | 1 | 4 | 240 | 35 KiB | 80 |
| R8, t8 | 5 | 1 | 10 | 256 | 61 KiB | 32 |
| R8, t32 | 5 | 4 | 10 | 256 | 83 KiB | 32 |
| R16, t1 | 4 | 1 | 4 | 240 | 58 KiB | 80 |
| R16, t32 | 20 | 4 | 10 | 128 | 168 KiB | 16 |

The H100 has 132 SMs, so every case launches roughly one or two blocks per SM.

---

## 3. Where the data lives

```
   HBM (80 GB, 3.35 TB/s)          L2 (50 MB)          SM: shared memory (≤ 227 KB)     SM: registers
   ─────────────────────           ──────────          ────────────────────────────    ─────────────
   x, packed cores A1 B2 C3  ───►  cached       ───►   B slice, A[a], C[k,a], x slice
                                                        S1   (one k at a time)
                                                        S2   (one k at a time)          Y accumulators
   FP32 workspace  ◄──── atomicAdd, once per block ◄─────────────────────────────────── (whole k loop)
   y (FP16)        ◄──── convert (A: 2nd kernel / B: last block)
```

The only per-call HBM traffic is x, the cores (~0.1–0.35 MB, mostly served from L2), and y.
Everything in between stays on the chip. In the reference, S1 and S2 go to HBM *and* are
copied again for re-layout (5 extra copy kernels).

---

## 4. One call on the timeline (design A, t = 1, from the H100 trace)

```
   GPU  ┃fill 1.05 µs┃ gap ┃████ tr_ring_fused_kernel 8.4 µs ████┃ gap ┃convert 1.1 µs┃
        workspace = 0        all pieces, atomics into workspace        FP32 → FP16

   reference, same call: 3 einsums → 8 kernels, ~112 µs (host-bound)
```

Design B folds the fill and convert into the fused kernel (one launch); G4 shows what that
buys and costs.

---

## 5. Files

| File | Role |
|---|---|
| `src/factorized_inference/csrc/tr_ring.cu` | the CUDA kernels + the C++ launcher (G2) |
| `src/factorized_inference/tr_kernel.py` | packing, tiling choice, the prepared callable |
| `src/factorized_inference/submission.py` | `prepare_optimized`: kernel for CUDA FP16, reference otherwise |
| `kernel_work/pieces_cpu.py` | the CPU oracle from K0 (piece algebra) |
| `kernel_work/tr_ring_v0.cu` | v0, kept for the history in G3 |
| `tools/check_kernel.py` | correctness sweep, A and B, many shapes and tilings |
| `tools/measure_kernels.py` | floors, kernel-only times, % of peak, tiling sweep |
| `tools/h100_*.sh`, `tools/remote.sh` | the H100 session (see `H100-RUNBOOK.md`) |

## 6. How to run it (laptop or H100)

```bash
source .venv/bin/activate
bash tools/make_cuda_home.sh                 # once: compiler from pip packages
python tools/check_kernel.py                 # correctness: expect "ALL OK"
pytest -q                                    # the assignment's tests: 21 passed
TR_DESIGN=B python tools/check_kernel.py --quick
python benchmarks/benchmark.py --device cuda:0 --dtype float16 --rank 8 --tokens 1,8,32 \
    --output results/rank8.json              # TR_DESIGN=A (default) or B
```

*Read later:* PyTorch C++/CUDA extensions (`torch.utils.cpp_extension.load`), the CUDA
Programming Guide chapters on shared memory and warp matrix functions (WMMA).
