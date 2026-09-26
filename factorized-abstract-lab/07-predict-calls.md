# 07 — Predict whole calls, then measure them

## Purpose

**Problem.** Guide 06 measured the machine and checked guide 05's formula one GEMM at a
time. This guide tests the formula on **whole calls**: the dense baseline and the full
reference chain (three GEMMs plus the permutes between them), for the project's five
cases. First predict every row, then measure, then find out why the misses happened.

This is the hypothesis test from guide 05: *time per call = max(host, Σ kernels)*. Every
implementation here is written in Julia on the GPU, so predictions and measurements come
from one system and a miss points at the model, not at a second runtime. Guide 08 moves
the same test to PyTorch and the project's harness.

| This guide | Project |
|---|---|
| `prepare_chain` | `tr_forward_reference`, `reference.py:122-124` (same three contractions) |
| `prepare_dense` | `materialize_dense_weight` + `dense_forward` |
| `stream_time` | the harness's `cuda_event_stream_*` (`benchmark.py:164-173`) |
| `best_time` of call + sync | the harness's `host_synchronized_*` |

## Mental model

```
   PREDICT (guide 05's formula, guide 06's numbers)          MEASURE (this GPU)

   host = ops × H          gpu = Σ kernel_time(FLOPs, bytes)
            └──── max(host, gpu) ────┘  ────── compare ──────   stream time
                                                                    │
                                    ratio ≠ 1?  ── time each kernel alone ──┐
                                                                            ▼
                                                  which kernel broke the formula,
                                                  and which parameter fixes it
```

The goal is not to be right on the first try. It is to be wrong in a way you can
**localise**: one kernel type, one missing parameter.

### In the studio

Guide 06 timed single button presses. Now the engineer runs **whole sessions**: the
dense studio (one pass through the giant board) and the chain studio (three desk passes
with three re-spoolings of the tape between them). You predict each session's length
from the price list, then time it. When a session runs long, you time each pass on its
own to find the slow machine.

## Setup

As in guide 06: type the file into `workspace/` and run from there with the GPU
environment.

**COMMAND**

```
julia --project=../julia-gpu 07_predict_calls.jl
```

## Step 1 — the machine, as data

**FILE — create `07_predict_calls.jl`**

```julia
# 07_predict_calls.jl — Rung 7: predict whole calls, then measure them
include("common.jl")
using CUDA, LinearAlgebra, Random

#== STEP 1 ==#
# Guide 05's machine. Paste YOUR numbers from guide 06, step 5.
Base.@kwdef struct Machine
    BW; F; eff; H; g
end
const m = Machine(BW = 1.8e11, F = 1.98e13, eff = 0.67, H = 6.4e-6, g = 8.7e-6)

kernel_time(m, flops, bytes) = max(flops / (m.eff * m.F), bytes / m.BW) + m.g
function call_time(m, host_ops, kernels)
    gpu = sum(kernel_time(m, f, b) for (f, b) in kernels)
    return max(host_ops * m.H, gpu), host_ops * m.H, gpu
end

# The same FP32-accumulating GEMM as guide 06.
const ONE, ZERO = CuRef{Float32}(1f0), CuRef{Float32}(0f0)
function mul32!(C, A, B)
    m, k = size(A)
    n = size(B, 2)
    CUBLAS.cublasGemmEx(CUBLAS.handle(), 'N', 'N', m, n, k,
                        ONE, A, Float16, m, B, Float16, k,
                        ZERO, C, Float16, m,
                        CUBLAS.CUBLAS_COMPUTE_32F, CUBLAS.CUBLAS_GEMM_DEFAULT)
    return C
end

const ni, nj, nk = 8, 12, 20
const np, nq, nr = 12, 10, 24
const K, N = ni * nj * nk, np * nq * nr          # 1920 inputs, 2880 outputs
const e = 2                                      # bytes per FP16 value

println("STEP 1 — the machine from guide 06")
@printf "  BW = %.0f GB/s   F = %.1f TFLOP/s   eff = %.2f   H = %s   g = %s\n" m.BW / 1e9 m.F / 1e12 m.eff us(m.H) us(m.g)
@printf "  a do-nothing kernel costs %s; one 11 MB read costs %s\n" us(kernel_time(m, 0, 0)) us(kernel_time(m, 0, e * N * K))
```

Notes:

- **Paste your own numbers** into the `Machine(…)` line: the one guide 06, step 5
  printed on your GPU. The expected outputs below use the numbers shown.
- `Machine`, `kernel_time` and `call_time` are guide 05's model, unchanged. Fields have no
  types (`BW; F; eff; H; g`), which keeps the struct short; it is not timed.
- `mul32!` is guide 06's FP32-accumulating GEMM, retyped because every guide is its own
  process.
- Inside `mul32!`, `m, k = size(A)` makes a **local** `m` that hides the global machine
  `m`. Harmless here, because `mul32!` never uses the machine, but worth noticing.

**Expected output** (exact for these parameters)

```
STEP 1 — the machine from guide 06
  BW = 180 GB/s   F = 19.8 TFLOP/s   eff = 0.67   H = 6.4 µs   g = 8.7 µs
  a do-nothing kernel costs 8.7 µs; one 11 MB read costs 70.1 µs
```

Two numbers to remember: a kernel that does nothing costs `g`, and reading the dense W
once costs about 70 µs on this GPU. Dense can never be faster than that.

## Step 2 — the chain and dense on the GPU, checked against the definition

The chain does what `reference.py` does, letter for letter, but each contraction is
written as a reshape, a GEMM and an explicit permute:

```
  X[i,(j,k,t)] ──G1──► T1[(a,p,b),(j,k,t)] ──P1──► T1p[(j,b),(k,t,a,p)]
               ──G2──► T2[(q,c),(k,t,a,p)] ──P2──► T2p[(k,a,c),(p,q,t)]
               ──G3──► Yr[r,(p,q,t)]       ──P3──► Y[(p,q,r),t]
```

A GEMM can only sum over indices that sit **together at the front** of one operand and at
the back of the other. Each permute moves the next indices to be summed to that
position. That is what PyTorch's einsum does internally before each GEMM.

**FILE — append to `07_predict_calls.jl`**

```julia
#== STEP 2 ==#
"Cores A[a,p,i,b], B[b,q,j,c], C[c,r,k,a] in FP16, scaled so that y comes out near 1."
function make_cores(R; seed = 7)
    Random.seed!(seed)
    s = (K * R^3)^(-1 / 6)
    return (Float16.(s .* randn(R, np, ni, R)),
            Float16.(s .* randn(R, nq, nj, R)),
            Float16.(s .* randn(R, nr, nk, R)))
end

"One output straight from the definition (guide 04, step 1), in Float64 on the CPU."
function y_def(A, B, C, X, p, q, r, t)
    A, B, C, X = Float64.(A), Float64.(B), Float64.(C), Float64.(X)
    s = 0.0
    for k in 1:nk, j in 1:nj, i in 1:ni
        s += tr(A[:, p, i, :] * B[:, q, j, :] * C[:, r, k, :]) * X[i, j, k, t]
    end
    return s
end

"The reference, written as GEMMs. Cores are reordered once; buffers are allocated once."
function prepare_chain(A, B, C, t)
    R = size(A, 1)
    A1 = CuArray(reshape(permutedims(A, (1, 2, 4, 3)), R * np * R, ni))  # [(a,p,b), i]
    B2 = CuArray(reshape(permutedims(B, (2, 4, 3, 1)), nq * R, nj * R))  # [(q,c), (j,b)]
    C3 = CuArray(reshape(permutedims(C, (2, 3, 4, 1)), nr, nk * R * R))  # [r, (k,a,c)]
    T1  = CuArray{Float16}(undef, R, np, R, nj, nk, t)                     # (a,p,b,j,k,t)
    T1p = CuArray{Float16}(undef, nj, R, nk, t, R, np)                     # (j,b,k,t,a,p)
    T2  = CuArray{Float16}(undef, nq, R, nk, t, R, np)                     # (q,c,k,t,a,p)
    T2p = CuArray{Float16}(undef, nk, R, R, np, nq, t)                     # (k,a,c,p,q,t)
    Yr  = CuArray{Float16}(undef, nr, np, nq, t)                           # (r,p,q,t)
    Y   = CuArray{Float16}(undef, np, nq, nr, t)                           # (p,q,r,t)
    ops = (X -> mul32!(reshape(T1, R * np * R, nj * nk * t), A1, reshape(X, ni, nj * nk * t)),  # G1: Σ i
           X -> permutedims!(T1p, T1, (4, 3, 5, 6, 1, 2)),                                        # P1
           X -> mul32!(reshape(T2, nq * R, nk * t * R * np), B2, reshape(T1p, nj * R, nk * t * R * np)),  # G2: Σ j,b
           X -> permutedims!(T2p, T2, (3, 5, 2, 6, 1, 4)),                                        # P2
           X -> mul32!(reshape(Yr, nr, np * nq * t), C3, reshape(T2p, nk * R * R, np * nq * t)),  # G3: Σ k,a,c
           X -> permutedims!(Y, Yr, (2, 3, 1, 4)))                                                # P3
    function forward(X)
        for op in ops
            op(X)
        end
        return reshape(Y, N, t)
    end
    return forward, ops
end
chain_forward(A, B, C, t) = prepare_chain(A, B, C, t)[1]

"Dense baseline: build W once, by pushing identity columns through the chain, 64 at a time."
function prepare_dense(A, B, C, t)
    f = chain_forward(A, B, C, 64)
    W = CUDA.zeros(Float16, N, K)
    Id = CuArray(Matrix{Float16}(I, K, K))
    for c0 in 1:64:K
        W[:, c0:c0+63] .= f(Id[:, c0:c0+63])
    end
    Y = CuArray{Float16}(undef, N, t)
    return X -> mul32!(Y, W, X)
end

println("\nSTEP 2 — chain and dense on the GPU agree with the definition")
const R0, t0 = 8, 4
const A0, B0, C0 = make_cores(R0)
const X0 = Float16.(randn(K, t0))
const spots = [(rand(1:np), rand(1:nq), rand(1:nr), rand(1:t0)) for _ in 1:20]
const yref = [y_def(A0, B0, C0, reshape(X0, ni, nj, nk, t0), s...) for s in spots]
for (name, prep) in (("chain", chain_forward), ("dense", prepare_dense))
    Y = Array(prep(A0, B0, C0, t0)(CuArray(X0)))
    Y4 = reshape(Y, np, nq, nr, t0)
    err = maximum(abs(Y4[s...] - yr) for (s, yr) in zip(spots, yref)) / maximum(abs, yref)
    @printf "  %-5s  max error on 20 outputs, relative to the largest: %.1e\n" name err
end
```

Notes:

- **The cores are reordered once**, in `prepare_chain`, on the CPU. Only the
  intermediates are permuted on every call. The project allows this ("packing"), and
  the harness counts it as preparation, not per-call cost.
- **Buffers are allocated once.** `CuArray{Float16}(undef, …)` reserves memory without
  filling it. Each call overwrites the same buffers, so the timing measures the work,
  not the allocator.
- `reshape` of a `CuArray` is free: same memory, new shape. Only `permutedims!` moves
  data.
- `ops` is a tuple of six small functions, one per kernel. `forward` runs them in order.
  Step 5 times them one by one.
- `prepare_dense` builds W by pushing the identity through the chain, 64 columns at a
  time. That is the "bounded" way: at most 64 columns of intermediates exist at once.
  The project forbids this **only** for the factorized method. For the dense baseline it
  is exactly what `materialize_dense_weight` is for.
- `y_def` is guide 04's definition, in Float64 on the CPU. It checks 20 random outputs;
  checking all 11,520 would take minutes.
- The scale `(K·R³)^(-1/6)` keeps y near 1. Unscaled, y would reach the hundreds and the
  FP16 error would look larger than it is.

**Expected output** (approximate: anything below `1e-2` is fine for FP16)

```
STEP 2 — chain and dense on the GPU agree with the definition
  chain  max error on 20 outputs, relative to the largest: 3.7e-04
  dense  max error on 20 outputs, relative to the largest: 2.7e-04
```

FP16 keeps about 3 significant digits, so ~4e-4 relative error is what correct looks
like. **What failure looks like:** a wrong permutation tuple gives an error near 1 (the
numbers land in the wrong places); a wrong reshape size gives a `DimensionMismatch`.

## Step 3 — predict, before measuring anything

**FILE — append to `07_predict_calls.jl`**

```julia
#== STEP 3 ==#
# The model's view: each method is a number of host ops and a list of (FLOPs, bytes).
function chain_kernels(R, t)
    t1 = R * np * R * nj * nk * t                     # |T1| = |T1p|
    t2 = nq * R * nk * t * R * np                     # |T2| = |T2p|
    return [(2.0 * t1 * ni,          e * (R * np * R * ni + K * t + t1)),     # G1
            (0.0,                    2e * t1),                                # P1
            (2.0 * t2 * nj * R,      e * (nq * R * nj * R + t1 + t2)),        # G2
            (0.0,                    2e * t2),                                # P2
            (2.0 * N * t * nk * R^2, e * (nr * nk * R^2 + t2 + N * t)),       # G3
            (0.0,                    2e * N * t)]                             # P3
end
dense_kernels(R, t) = [(2.0 * N * K * t, e * (N * K + K * t + N * t))]

const cases = ((8, 1), (8, 8), (8, 32), (16, 1), (16, 32))
const methods = (("dense", 1, dense_kernels, prepare_dense), ("chain", 6, chain_kernels, chain_forward))

println("\nSTEP 3 — predicted time per call (host side, GPU side, the slower one)")
println("   R   t  method      host       gpu     total   limited by")
for (R, t) in cases, (name, ops, ks, _) in methods
    total, host, gpu = call_time(m, ops, ks(R, t))
    @printf "  %2d  %2d  %-6s %9s %9s %9s   %s\n" R t name us(host) us(gpu) us(total) host > gpu ? "host" : "GPU"
end
```

Notes:

- `chain_kernels` lists the six kernels with guide 05's counting rules: a GEMM reads its
  two inputs and writes its output (`e × (in1 + in2 + out)`), and a permute reads and
  writes everything once (`2e × size`), with no FLOPs.
- The chain has **6 host ops** here, not guide 05's 9: the cores are no longer copied on
  every call. Guide 08 counts what PyTorch actually does.

**Predict first:** write down which of the ten rows will miss by the most, and in which
direction.

**Expected output** (exact for these parameters)

```
STEP 3 — predicted time per call (host side, GPU side, the slower one)
   R   t  method      host       gpu     total   limited by
   8   1  dense     6.4 µs   70.2 µs   70.2 µs   GPU
   8   1  chain    38.4 µs   67.8 µs   67.8 µs   GPU
   8   8  dense     6.4 µs   70.6 µs   70.6 µs   GPU
   8   8  chain    38.4 µs  173.8 µs  173.8 µs   GPU
   8  32  dense     6.4 µs   71.8 µs   71.8 µs   GPU
   8  32  chain    38.4 µs  537.0 µs  537.0 µs   GPU
  16   1  dense     6.4 µs   70.2 µs   70.2 µs   GPU
  16   1  chain    38.4 µs  116.8 µs  116.8 µs   GPU
  16  32  dense     6.4 µs   71.8 µs   71.8 µs   GPU
  16  32  chain    38.4 µs   2.07 ms   2.07 ms   GPU
```

Every row is GPU-bound. On this GPU `g` is larger than `H`, and the 180 GB/s bus makes
even dense slow (70 µs). The model says the chain nearly ties dense at R = 8, t = 1.

## Step 4 — measure

**FILE — append to `07_predict_calls.jl`**

```julia
#== STEP 4 ==#
"Harness-style stream time: CUDA events around 20 back-to-back calls, per call; best of 5."
function stream_time(f; n = 20, reps = 5)
    f(); CUDA.synchronize()
    return minimum(CUDA.@elapsed(for _ in 1:n; f(); end) for _ in 1:reps) / n
end
const measured = Dict{Tuple{Int, Int, String}, Float64}()        # step 6 reuses these

println("\nSTEP 4 — measured against predicted")
println("   R   t  method  predicted   stream    ratio   host+sync")
for (R, t) in cases
    A, B, C = make_cores(R)
    X = CuArray(Float16.(randn(K, t)))
    for (name, ops, ks, prep) in methods
        f = prep(A, B, C, t)
        tp, _, _ = call_time(m, ops, ks(R, t))
        ts = stream_time(() -> f(X))
        th = best_time(() -> (f(X); CUDA.synchronize()))
        measured[(R, t, name)] = ts
        @printf "  %2d  %2d  %-6s %9s %9s   %5.2f   %9s\n" R t name us(tp) us(ts) ts / tp us(th)
    end
end
```

Notes:

- `stream_time` is what the harness reports as `cuda_event_stream_*`: events around 20
  back-to-back calls. If the host is the slower side, the GPU waits for it, and the
  events see that waiting too. That is why it measures `max(host, gpu)`, not just the GPU.
- `host+sync` is the harness's other number: one call, then wait for the GPU. It is
  always a bit larger, because the host also pays for the wait itself.
- `measured` keeps the stream times for step 6.

**Expected output** (machine-dependent)

```
STEP 4 — measured against predicted
   R   t  method  predicted   stream    ratio   host+sync
   8   1  dense    70.2 µs   88.0 µs    1.25    101.3 µs
   8   1  chain    67.8 µs   96.6 µs    1.42    108.7 µs
   8   8  dense    70.6 µs   70.7 µs    1.00     84.0 µs
   8   8  chain   173.8 µs  470.8 µs    2.71    516.1 µs
   8  32  dense    71.8 µs   95.9 µs    1.34    106.3 µs
   8  32  chain   537.0 µs   1.83 ms    3.40     1.89 ms
  16   1  dense    70.2 µs   81.2 µs    1.16     95.4 µs
  16   1  chain   116.8 µs  251.9 µs    2.16    261.4 µs
  16  32  dense    71.8 µs   95.1 µs    1.32    108.0 µs
  16  32  chain    2.07 ms   7.31 ms    3.53     7.38 ms
```

Measured over three runs: dense ratios 1.00–1.39; chain ratios 1.38–3.56. Each value
moves by under 7% between runs.

**Read it.** Dense lands within 1.0–1.4×. The chain is **1.4–3.6× slower than
predicted**, and the miss grows with t. A miss that grows with the size of the data is
not a fixed cost like `g` or `H`. It is something proportional to bytes or FLOPs that
the model gets wrong. Step 5 finds it.

## Step 5 — localise the miss: each kernel alone

**FILE — append to `07_predict_calls.jl`**

```julia
#== STEP 5 ==#
println("\nSTEP 5 — where the chain's time goes: each kernel alone")
const names = ("G1", "P1", "G2", "P2", "G3", "P3")
const perm_bw = Float64[]                                        # step 6 reuses these
for (R, t) in ((8, 1), (8, 32), (16, 32))
    A, B, C = make_cores(R)
    X = CuArray(Float16.(randn(K, t)))
    _, ops = prepare_chain(A, B, C, t)
    println("  R = $R, t = $t:   predicted   measured   ratio   bytes / measured")
    for (nm, op, (fl, by)) in zip(names, ops, chain_kernels(R, t))
        tp = kernel_time(m, fl, by)
        tm = stream_time(() -> op(X))
        @printf "    %s        %9s  %9s   %5.2f   %6.1f GB/s\n" nm us(tp) us(tm) tm / tp by / tm / 1e9
        t == 32 && nm in ("P1", "P2") && push!(perm_bw, by / tm)
    end
end
```

Notes:

- `zip(names, ops, chain_kernels(R, t))` walks the six names, the six functions and the
  six (FLOPs, bytes) pairs together.
- `bytes / measured` is the bandwidth each kernel **actually** reached. For a
  memory-bound kernel, compare it with `BW` = 180 GB/s.
- `perm_bw` keeps the permute bandwidths at t = 32 for step 6.

**Expected output** (machine-dependent)

```
STEP 5 — where the chain's time goes: each kernel alone
  R = 8, t = 1:   predicted   measured   ratio   bytes / measured
    G1          10.8 µs    12.9 µs    1.19     29.9 GB/s
    P1          12.8 µs    32.8 µs    2.56     22.5 GB/s
    G2          12.5 µs    16.0 µs    1.28     43.1 GB/s
    P2          12.1 µs    24.7 µs    2.04     24.9 GB/s
    G3          10.8 µs    15.0 µs    1.39     25.0 GB/s
    P3           8.8 µs    12.0 µs    1.37      1.0 GB/s
  R = 8, t = 32:   predicted   measured   ratio   bytes / measured
    G1          75.0 µs    72.5 µs    0.97    164.6 GB/s
    P1         139.8 µs   864.0 µs    6.18     27.3 GB/s
    G2         128.9 µs   135.3 µs    1.05    160.0 GB/s
    P2         117.9 µs   671.9 µs    5.70     29.3 GB/s
    G3          64.7 µs    62.2 µs    0.96    162.1 GB/s
    P3          10.7 µs    13.5 µs    1.25     27.4 GB/s
  R = 16, t = 32:   predicted   measured   ratio   bytes / measured
    G1         271.8 µs   261.0 µs    0.96    181.5 GB/s
    P1         533.0 µs    3.16 ms    5.94     29.8 GB/s
    G2         577.8 µs   918.2 µs    1.59     94.3 GB/s
    P2         445.6 µs    2.70 ms    6.07     29.1 GB/s
    G3         229.5 µs   223.4 µs    0.97    178.0 GB/s
    P3          10.7 µs    13.9 µs    1.29     26.6 GB/s
```

**Read it.**

- **G1 and G3 are within 5%, and G2 at R = 8.** They reach 160–180 GB/s. The GEMM
  formula works.
- **P1 and P2 are 6× slower than predicted.** They reach **27–30 GB/s, about 16% of the
  bus**, while guide 06's `copyto!` reached 180 GB/s. A permute is a copy with scattered
  reads or writes; CUDA.jl's generic `permutedims!` does not tile them, so most of each
  memory transaction is wasted. At t = 32 the two permutes take **80–84%** of the chain's
  time.
- **G2 at R = 16 is 1.6× off.** In guide 06, the same FLOPs as a 122880 × 192 × 160 GEMM
  took 569 µs. Here it is written the other way round, 160 × 192 × 122880, and cuBLAS
  takes 918 µs. **The model has no parameter for orientation.** Same work, different
  shape, 1.6× the time.
- **At t = 1 everything is 1.2–2.6× off.** These kernels are all near `g`, and ±5 µs of
  fixed-cost noise is a large ratio.

## Step 6 — repair the model with one measured parameter

**FILE — append to `07_predict_calls.jl`**

```julia
#== STEP 6 ==#
# Repair: a permute is a copy that reaches only BWp, not BW.
const BWp = sum(perm_bw) / length(perm_bw)
kernel_time_fixed(fl, by) = fl == 0 ? by / BWp + m.g : kernel_time(m, fl, by)

println("\nSTEP 6 — the model with permutes at their measured bandwidth")
@printf "  BWp = %.1f GB/s  (%.0f%% of BW)\n" BWp / 1e9 100BWp / m.BW
println("   R   t  method  predicted   measured   ratio")
for (R, t) in cases, (name, ops, ks, _) in methods
    gpu = sum(kernel_time_fixed(fl, by) for (fl, by) in ks(R, t))
    tp = max(ops * m.H, gpu)
    tm = measured[(R, t, name)]
    @printf "  %2d  %2d  %-6s %9s  %9s   %5.2f\n" R t name us(tp) us(tm) tm / tp
end
```

Notes:

- `fl == 0 ? … : …` picks the permute rule for kernels with no FLOPs and keeps guide
  05's rule for everything else. One new parameter, `BWp`, measured in step 5.
- `100BWp` is `100 * BWp`.

**Expected output** (machine-dependent)

```
STEP 6 — the model with permutes at their measured bandwidth
  BWp = 28.9 GB/s  (16% of BW)
   R   t  method  predicted   measured   ratio
   8   1  dense    70.2 µs    88.0 µs    1.25
   8   1  chain   107.5 µs    96.6 µs    0.90
   8   8  dense    70.6 µs    70.7 µs    1.00
   8   8  chain   491.0 µs   470.8 µs    0.96
   8  32  dense    71.8 µs    95.9 µs    1.34
   8  32  chain    1.81 ms    1.83 ms    1.01
  16   1  dense    70.2 µs    81.2 µs    1.16
  16   1  chain   274.4 µs   251.9 µs    0.92
  16  32  dense    71.8 µs    95.1 µs    1.32
  16  32  chain    7.11 ms    7.31 ms    1.03
```

Measured over three runs: `BWp` 28.8–29.1 GB/s; chain ratios 0.88–1.03; dense unchanged.

**Read it.** One measured number moves the chain from 1.4–3.6× off to **0.88–1.03**.
The model's structure (host ops, a list of kernels, roofs plus `g`) was right; one
parameter was wrong: permutes do not run at copy bandwidth. That is a finding you can
defend: **the reference's cost at large t is mostly layout shuffling, not arithmetic.**
It also names the first thing a custom kernel should remove.

Dense still misses by up to 1.39× at t = 32. That is exercise 2.

## What is *not* true

- *"The model was wrong, so it is useless."* It was wrong in **one** place, found in one
  step and fixed with one measured number. A model that can be wrong in a localised way
  is exactly what makes a measurement explainable.
- *"Permutes are always 6× slow."* This is CUDA.jl's generic `permutedims!` on this GPU.
  PyTorch's copy kernels are different code (guide 08 shows them). A hand-written
  permute with shared-memory tiling, the classic transpose kernel, reaches close to
  copy bandwidth. The lesson is the size of the gap, not the constant.
- *"The chain loses to dense, so factorization loses."* This chain materialises two big
  intermediates and shuffles them. The fused "cut the ring" kernel from guides 03–05
  does not write T1/T2 at all, so it has no permutes to lose on. That kernel is the
  assignment.
- *"FLOPs predict GEMM time."* G2 in two orientations: same FLOPs, same bytes, 1.6×
  different time. Shape matters to a library kernel in ways the roofs cannot see.

## Exercises

1. In `chain_kernels`, which kernel would you remove first if a custom kernel could fuse
   P1 into G2? Predict the new chain time at R = 8, t = 32 with the repaired model, before
   changing any code.
2. Dense at t = 32 is 1.33–1.39× off. Time `mul32!(Y, W, X)` alone with `stream_time` and
   compare it with the byte roof `e * (N*K + K*t + N*t) / m.BW`. Is the miss in the GEMM,
   or somewhere else? (Hint: compare with guide 06's G1 at R = 16, t = 32, a GEMM with a
   similar shape.)
3. Write G2 in guide 06's orientation: `mul32!(T2', T1p', B2')` does not work directly
   (`mul32!` only takes plain matrices). What would you have to permute differently to
   get the 122880 × 192 × 160 shape, and would that cost an extra permute?
4. Run step 3 with `Machine(…, H = 0.0)`. Which rows change? Why does removing the host
   cost change nothing on this GPU, when in guide 05 it changed almost every row?

## Checklist

- [ ] steps 1–6 run; step 2 errors are below `1e-2`; steps 1 and 3 match for my numbers
- [ ] I can say why a GEMM needs a permute before it, in terms of where the summed
      indices sit
- [ ] I can say what `stream_time` measures when the host is slower than the GPU
- [ ] I can name the kernel type that broke the formula, how I localised it, and the one
      parameter that repaired it
- [ ] I can explain why the same FLOPs in a different orientation took 1.6× the time,
      and why the model cannot see it
- [ ] in studio terms: which machine made the chain session run long
