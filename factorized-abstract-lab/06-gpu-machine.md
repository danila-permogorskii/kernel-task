# 06 — Measure the machine: replacing guide 05's guesses

## Purpose

**Problem.** Guide 05 predicted the benchmark table from five parameters, three of them
guesses: `eff`, `H` and `g`. This guide measures all five on a real GPU, then checks the
kernel half of guide 05's formula against the reference's three real GEMMs.

This is the measurement plan from guide 05 ("measure `H` and `g` from the trace, measure
your kernel's achieved FLOP/s for `eff`"), carried out. It uses the same cuBLAS library
PyTorch calls in `reference.py:122-124`, with the same FP32 accumulation the harness forces
(`benchmarks/benchmark.py:91`).

**Hardware.** The outputs below come from an RTX 3050 Ti Laptop GPU (20 SMs, 4 GB) under
Windows. Guide 08 checks how well these numbers carry over; the H100 will give different
values. The **method** carries over unchanged.

## Mental model

The whole guide rests on one trick: **park the GPU first**, so the host gets ahead.

```
  HOST    ─[spin]─[f][f][f][f]…[f]──────────────────────────────  enqueue n calls
                  └── th ──────┘      th / n  = HOST cost per call  (H)

  GPU     ═══════ spin 20 ms ═══════╪[f][f][f][f]…[f]╪              runs them back to back
                                    e1               e2
                                    └─── (e2 - e1) / n = GPU cost per call ───┘
```

Without the spin, the GPU would sit idle waiting for each call, and the GPU clock would
only measure the host again. With the spin, all `n` calls are already queued when the
GPU gets to them. The events then see nothing but GPU work.

### In the studio

The engineer loads a 20-minute song onto the hardware (the spin). While it plays, they
press the next 200 buttons as fast as they can. The **engineer's stopwatch** times the
button presses (`H`). The **hardware's clock** starts when the long song ends and times
the 200 short passes (`g`, or the kernel's time). Two clocks, one measurement.

## Setup

The GPU environment lives in `julia-gpu/` next to your `workspace/` folder. It holds
CUDA.jl, which installs the CUDA runtime and cuBLAS itself. Type the file into
`workspace/`, next to `common.jl`, and run from there:

**COMMAND**

```
julia --project=../julia-gpu 06_gpu_machine.jl
```

`--project` points Julia at the environment that has CUDA.jl. Without it you get
`Package CUDA not found`.

## Step 1 — the parking kernel and the two clocks

**FILE — create `06_gpu_machine.jl`**

```julia
# 06_gpu_machine.jl — Rung 6: measure the machine that rung 5 guessed
include("common.jl")
using CUDA, LinearAlgebra

#== STEP 1 ==#
# A kernel that does nothing but wait `cycles` GPU clock ticks: it parks the GPU.
function spin!(cycles)
    t0 = clock(UInt64)
    while clock(UInt64) - t0 < cycles
    end
    return
end
const SPIN = UInt64(30_000_000)             # about 20 ms at 1.5 GHz

"Per-call HOST time and GPU time of f(). The GPU is parked first, so the host runs ahead."
function split_time(f; n = 200, reps = 3)
    f(); CUDA.synchronize()                 # compile and warm up
    host, gpu = Inf, Inf
    e1, e2 = CuEvent(), CuEvent()
    for _ in 1:reps
        @cuda spin!(SPIN)                   # the GPU is busy for ~20 ms
        record(e1)                          # GPU clock starts when the spin ends
        th = @elapsed for _ in 1:n          # host clock: time to ENQUEUE n calls
            f()
        end
        record(e2)                          # GPU clock stops after the n-th call
        synchronize(e2)
        host = min(host, th / n)
        gpu  = min(gpu, CUDA.elapsed(e1, e2) / n)
    end
    return host, gpu
end

println("STEP 1 — the device and the parking kernel")
const dev = CUDA.device()
@printf "  %s: %d SMs, %.1f GiB\n" CUDA.name(dev) CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) CUDA.totalmem(dev) / 1024^3
@cuda spin!(UInt64(1)); CUDA.synchronize()
@printf "  one spin on the GPU clock: %s\n" us(CUDA.@elapsed @cuda spin!(SPIN))
```

Notes:

- `spin!` is your first hand-written GPU kernel. `clock(UInt64)` reads the SM's cycle
  counter, so the loop runs for a fixed number of GPU cycles. A loop doing arithmetic
  would not work: the compiler removed it and the "20 ms" kernel took 75 µs.
- `@cuda spin!(SPIN)` **launches** the kernel and returns immediately. The host does
  not wait. That asynchrony is what the whole guide measures.
- `record(e)` puts a timestamp *into the GPU's queue*. It fires when the GPU reaches
  it, not when the host calls it. `CUDA.elapsed(e1, e2)` is GPU time between the two.
- `CUDA.@elapsed expr` is the same idea in one macro: an event before, an event after,
  wait, subtract. The harness's `cuda_event_stream_*` numbers are measured this way.
- The `@cuda spin!(UInt64(1))` line warms up the kernel, so compilation is not timed.

**COMMAND**

```
julia --project=../julia-gpu 06_gpu_machine.jl
```

**Expected output** (the device line is exact for this laptop; the time is ~20 ms on any
GPU clocked near 1.5 GHz)

```
STEP 1 — the device and the parking kernel
  NVIDIA GeForce RTX 3050 Ti Laptop GPU: 20 SMs, 4.0 GiB
  one spin on the GPU clock: 20.26 ms
```

Measured over three runs: 20.26–20.28 ms. The first run after installing takes about a
minute longer, because CUDA.jl compiles itself.

**What failure looks like**

- `Package CUDA not found`: the `--project=../julia-gpu` flag is missing or points to
  the wrong folder.
- `UndefVarError: us`: `common.jl` is not in the folder you run from.

## Step 2 — the price of one button press

**FILE — append to `06_gpu_machine.jl`**

```julia
#== STEP 2 ==#
function add1!(y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(y)
        @inbounds y[i] += 1f0
    end
    return
end
const y = CUDA.zeros(Float32, 1024)

println("\nSTEP 2 — the price of one button press (almost no work)")
const H, g = split_time(() -> @cuda threads=256 blocks=4 add1!(y))
@printf "  hand-written kernel  host %8s   gpu %8s\n" us(H) us(g)
hb, gb = split_time(() -> (y .+= 1f0))
@printf "  broadcast y .+= 1    host %8s   gpu %8s\n" us(hb) us(gb)
```

Notes:

- `add1!` adds 1 to 1024 numbers: 4 KiB of traffic, which is nothing. Whatever time it
  takes is the **fixed cost of a kernel**, not the work.
- `threads=256 blocks=4`: 4 blocks of 256 threads, one thread per element.
  `(blockIdx().x - 1) * blockDim().x + threadIdx().x` is the thread's global number, the
  same formula as `blockIdx.x * blockDim.x + threadIdx.x` in HIP, shifted because Julia
  counts from 1.
- `const H, g = …` keeps both numbers. Step 5 puts them into the machine.

**Predict first:** guide 05 guessed `H = 5 µs` and `g = 1 µs`. Which one will be further
off?

**Expected output** (machine-dependent)

```
STEP 2 — the price of one button press (almost no work)
  hand-written kernel  host   6.4 µs   gpu   8.7 µs
  broadcast y .+= 1    host  11.3 µs   gpu   7.7 µs
```

Measured over three runs: kernel host 6.4–7.1 µs, gpu 8.3–9.7 µs; broadcast host
9.8–12.4 µs, gpu 7.7–8.8 µs.

**Read it.** `H` was guessed well (5 → ~7 µs). `g` was guessed **8–10× too small**. Even
when every call is already queued, this GPU needs about 8–9 µs per kernel. That is not
a property of the work: the kernel moves 4 KiB. It is the cost of the GPU starting and
finishing a kernel on this laptop. The broadcast costs **more on the host** (Julia builds
the kernel's arguments) but **the same on the GPU**: a kernel is a kernel. Different
buttons cost the engineer different amounts, while the hardware charges the same.

## Step 3 — the two roofs

**FILE — append to `06_gpu_machine.jl`**

```julia
#== STEP 3 ==#
# FP16 in, FP16 out, FP32 accumulation: the setting the project's harness uses.
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

println("\nSTEP 3 — the two roofs: bandwidth and matmul peak")
const src = CUDA.zeros(UInt8, 512 * 2^20)   # 512 MiB
const dst = similar(src)
_, t_copy = split_time(() -> copyto!(dst, src); n = 5)
const BW = 2 * length(src) / t_copy         # a copy reads and writes every byte
const BW_spec = 2 * CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) * 1e3 *
                CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH) / 8
@printf "  bandwidth          %6.1f GB/s      spec %6.1f GB/s\n" BW / 1e9 BW_spec / 1e9

const nb = 4096
const Ab, Bb = CUDA.randn(Float16, nb, nb), CUDA.randn(Float16, nb, nb)
const Cb = similar(Ab)
_, t16 = split_time(() -> mul!(Cb, Ab, Bb); n = 5)
_, t32 = split_time(() -> mul32!(Cb, Ab, Bb); n = 5)
const F = 2nb^3 / t32
@printf "  FP16, FP16 accum.  %6.1f TFLOP/s\n" 2nb^3 / t16 / 1e12
@printf "  FP16, FP32 accum.  %6.1f TFLOP/s   ← the harness's setting\n" F / 1e12
```

Notes:

- **Why `mul32!` exists.** Julia's `mul!` on `Float16` asks cuBLAS for **FP16
  accumulation**. PyTorch in the harness uses **FP32 accumulation**, because
  `benchmark.py:91` sets `allow_fp16_reduced_precision_reduction = False`. To measure
  the same thing, `mul32!` calls cuBLAS directly with `CUBLAS_COMPUTE_32F`. Inputs and
  output stay FP16; only the running sum inside is FP32.
- **Why `ONE` and `ZERO` are constants.** cuBLAS here reads `alpha` and `beta`
  (`C = alpha·A·B + beta·C`) from **GPU memory**. Creating `CuRef{Float32}(1f0)` inside
  `mul32!` allocates and copies to the GPU on every call: two hidden operations, which
  made each small GEMM cost 22 µs instead of 9 µs. Hidden operations inside one call are
  the reference chain's whole problem (guide 05, step 2); here you met one first-hand.
- The bandwidth spec is the bus: 6001 MHz memory clock × 2 (double data rate) × 128 bits
  ÷ 8 = 192 GB/s. There is no clean spec line for the matmul peak. Laptop GPUs boost
  their clock depending on power and temperature, so the measured value is the one used.
- `2nb^3` is `2 * nb^3`: Julia lets a number multiply a name directly.

**Expected output** (machine-dependent)

```
STEP 3 — the two roofs: bandwidth and matmul peak
  bandwidth           180.1 GB/s      spec  192.0 GB/s
  FP16, FP16 accum.    34.9 TFLOP/s
  FP16, FP32 accum.    19.8 TFLOP/s   ← the harness's setting
```

Measured over three runs: 180.1–180.3 GB/s, 34.5–34.9 and 19.8–20.0 TFLOP/s.

**Read it.** The copy reaches 94% of the bus spec, which is normal for a copy.
**Accumulation precision costs 43% of the matmul peak** on this GPU. A number quoted
"FP16 TFLOP/s" without saying which accumulation is ambiguous by almost 2×.

## Step 4 — the reference's three GEMMs at the real sizes

Every einsum in `reference.py` becomes one GEMM once the letters are grouped. Read the
shapes off the strings: letters kept from the first operand are the rows `M`, summed
letters are `K`, and letters kept from the second operand are the columns `N`.

**FILE — append to `06_gpu_machine.jl`**

```julia
#== STEP 4 ==#
const ni, nj, nk = 8, 12, 20
const np, nq, nr = 12, 10, 24
# (M, K, N) of the reference's three GEMMs, read off its einsum strings
gemm_shapes(R, t) = ((t * nj * nk,     ni,         R * np * R),   # "tijk,apib->tjkapb"
                     (t * nk * R * np, nj * R,     nq * R),       # "tjkapb,bqjc->tkapqc"
                     (t * np * nq,     nk * R * R, nr))           # "tkapqc,crka->tpqr"
const cases = ((8, 1), (8, 8), (8, 32), (16, 1), (16, 32))
const rows = Tuple{Float64, Float64, Float64}[]                   # (FLOPs, bytes, time)

println("\nSTEP 4 — the reference's three GEMMs at the real sizes (GPU clock)")
println("   R   t  GEMM       M      K     N    measured   FLOP roof   byte roof")
for (R, t) in cases, (gi, (M, K, N)) in enumerate(gemm_shapes(R, t))
    A = CUDA.randn(Float16, M, K)
    B = CUDA.randn(Float16, K, N)
    C = CUDA.zeros(Float16, M, N)
    _, tg = split_time(() -> mul32!(C, A, B))
    fl = 2.0 * M * K * N
    by = 2.0 * (M * K + K * N + M * N)
    push!(rows, (fl, by, tg))
    @printf "  %2d  %2d   G%d  %6d  %5d  %4d   %9s   %9s   %9s\n" R t gi M K N us(tg) us(fl / F) us(by / BW)
end
```

Notes:

- G1, `"tijk,apib->tjkapb"`: kept from x are `t, j, k` (M = t·12·20), summed is `i` (K = 8),
  kept from the core are `a, p, b` (N = R·12·R). The other two follow the same way.
- **FLOP roof** = the time at full `F`. **Byte roof** = the time to read A and B and
  write C once at full `BW`. A kernel cannot beat the larger of the two.
- This measures only the GEMMs. The permute copies the reference adds between them are
  not here; guide 07 adds them.
- `for (R, t) in cases, (gi, (M, K, N)) in …` is one loop over two variables: all 15
  (case, GEMM) pairs.

**Predict first:** at t = 1, which roof is closer to the measured time?

**Expected output** (shapes and roofs exact for these `F`, `BW`; times machine-dependent)

```
STEP 4 — the reference's three GEMMs at the real sizes (GPU clock)
   R   t  GEMM       M      K     N    measured   FLOP roof   byte roof
   8   1   G1     240      8   768      8.8 µs      0.1 µs      2.1 µs
   8   1   G2    1920     96    80     10.7 µs      1.5 µs      3.8 µs
   8   1   G3     120   1280    24     15.2 µs      0.4 µs      2.1 µs
   8   8   G1    1920      8   768     20.8 µs      1.2 µs     16.6 µs
   8   8   G2   15360     96    80     37.0 µs     11.9 µs     30.1 µs
   8   8   G3     960   1280    24     19.3 µs      3.0 µs     14.2 µs
   8  32   G1    7680      8   768     74.2 µs      4.8 µs     66.2 µs
   8  32   G2   61440     96    80    130.4 µs     47.6 µs    120.2 µs
   8  32   G3    3840   1280    24     66.6 µs     11.9 µs     55.9 µs
  16   1   G1     240      8  3072     11.1 µs      0.6 µs      8.5 µs
  16   1   G2    3840    192   160     29.7 µs     11.9 µs     15.3 µs
  16   1   G3     120   5120    24     17.7 µs      1.5 µs      8.2 µs
  16  32   G1    7680      8  3072    263.8 µs     19.0 µs    262.9 µs
  16  32   G2  122880    192   160    569.1 µs    380.6 µs    480.6 µs
  16  32   G3    3840   5120    24    253.5 µs     47.6 µs    220.7 µs
```

Measured over three runs: every time within about ±15% of these at t = 1, and within ±3%
at t = 32.

**Read it, row group by row group.**

- **t = 1: neither roof.** Both roofs are 0.1–4 µs; the measured times are 9–18 µs,
  close to step 2's `g`. The work is too small to matter. **Each GEMM costs about one
  kernel's fixed price.**
- **t = 32: the byte roof.** Most rows sit a few percent above it. G1 at R = 16 hits it
  almost exactly (263.8 vs 262.9 µs). These GEMMs are **memory-bound**, like your GEMV
  work: `K` is small (8 or 96), so each byte loaded is used for few FLOPs.
- **The one compute-heavy GEMM** is G2 at R = 16, t = 32. It is the R³ contraction of
  guide 04, and the only row where the FLOP roof is the same order as the byte roof.

## Step 5 — the measured machine, and a first test of the formula

Guide 05's kernel formula: `kernel = max(FLOPs / (eff·F), bytes / BW) + g`. Four of its
parameters are now measured. The fifth, `eff`, is **fitted**: the best fraction of `F`
that any of the 15 GEMMs reached. Then the formula predicts all 15.

**FILE — append to `06_gpu_machine.jl`**

```julia
#== STEP 5 ==#
const eff = maximum(fl / (tg * F) for (fl, by, tg) in rows)       # best fraction of F reached
kernel_time(fl, by) = max(fl / (eff * F), by / BW) + g            # guide 05's formula

println("\nSTEP 5 — the measured machine, and guide 05's kernel formula on 15 GEMMs")
@printf "  Machine(BW = %.3g, F = %.3g, eff = %.2f, H = %.2g, g = %.2g)\n" BW F eff H g
println("   R   t  GEMM   measured   predicted   measured / predicted")
for (row, ((R, t), gi)) in zip(rows, ((c, gi) for c in cases for gi in 1:3))
    fl, by, tg = row
    tp = kernel_time(fl, by)
    @printf "  %2d  %2d   G%d   %9s   %9s   %6.2f\n" R t gi us(tg) us(tp) tg / tp
end
```

Notes:

- `((c, gi) for c in cases for gi in 1:3)` generates `(case, GEMM number)` in the same
  order the rows were pushed in step 4. `zip` walks both together.
- The printed `Machine(…)` line is the output of this guide. Copy it: guide 07 uses it.

**Expected output** (machine-dependent)

```
STEP 5 — the measured machine, and guide 05's kernel formula on 15 GEMMs
  Machine(BW = 1.8e+11, F = 1.98e+13, eff = 0.67, H = 6.4e-06, g = 8.7e-06)
   R   t  GEMM   measured   predicted   measured / predicted
   8   1   G1      8.8 µs     10.8 µs     0.82
   8   1   G2     10.7 µs     12.5 µs     0.85
   8   1   G3     15.2 µs     10.8 µs     1.41
   8   8   G1     20.8 µs     25.3 µs     0.82
   8   8   G2     37.0 µs     38.8 µs     0.95
   8   8   G3     19.3 µs     22.9 µs     0.84
   8  32   G1     74.2 µs     74.9 µs     0.99
   8  32   G2    130.4 µs    128.9 µs     1.01
   8  32   G3     66.6 µs     64.6 µs     1.03
  16   1   G1     11.1 µs     17.2 µs     0.65
  16   1   G2     29.7 µs     26.5 µs     1.12
  16   1   G3     17.7 µs     16.9 µs     1.05
  16  32   G1    263.8 µs    271.6 µs     0.97
  16  32   G2    569.1 µs    577.8 µs     0.98
  16  32   G3    253.5 µs    229.4 µs     1.11
```

Measured over three runs: `eff` 0.67 every time, `H` 6.4–7.1 µs, `g` 8.3–9.7 µs. The
ratios move by up to 0.3 at t = 1, and by 0.03 or less at t = 32.

**The verdict on the kernel formula.**

- **Large kernels: within ±11%.** Every t = 32 row lands between 0.96 and 1.11. G2 at
  R = 16, t = 32 is ≈1 partly by construction, because `eff` was fitted on it. The other
  fourteen are genuine predictions.
- **Small kernels: within 0.65–1.5×.** The formula adds `g` on top of a roof. For tiny
  GEMMs the real cost behaves more like `max(roof, g)`, so the formula overshoots (0.65–0.85).
- **The outlier, G3 at R = 8, t = 1 (1.41–1.68).** K = 1280 is long while M and N are tiny.
  **Guess, not measured:** cuBLAS splits the long sum across blocks ("split-K") and
  launches a second kernel to add the pieces. That is two `g`s. A profiler trace would
  confirm or refute it; that is exercise 3.

## What is *not* true

- *"`g` is small, so launch cost is only a host problem."* On this machine the GPU side
  alone costs ~9 µs per kernel, more than the host side. A 9-kernel reference chain pays
  ~80 µs of GPU time at t = 1 before any work. On the H100 under Linux, `g` is expected to
  be smaller, but it must be measured there too, not assumed.
- *"`eff` is how good cuBLAS is."* The fitted 0.67 comes from one mid-sized,
  compute-heavy GEMM. Tiny GEMMs reach 2–3% of `F`, and not because the library is bad:
  they are bounded by `g`. `eff` only means something where FLOPs are the limit.
- *"These numbers predict the H100 table."* They describe this laptop. What transfers to
  the H100 is the procedure (park, two clocks, roofs, fit, check residuals) and the
  structure (t = 1 is fixed-cost-bound, t = 32 is byte-bound, G2 at R = 16 is the only
  compute-heavy contraction).
- *"The harness's PyTorch runs would give the same GEMM times."* They call the same
  cuBLAS, but PyTorch may pick different algorithms and layouts, and adds its own permute
  copies. Guide 08 measures that directly.

## Exercises

1. Set `SPIN = UInt64(300_000)` (about 0.2 ms) and run step 2. Why does the GPU time
   change? (Hint: is the host still ahead when the spin ends?)
2. Replace `mul32!` with `mul!` in step 4 only (FP16 accumulation). Which rows get
   faster, and by how much? Predict from step 3 before running.
3. Test the split-K guess for G3. After step 3, create that one shape
   (`A = CUDA.randn(Float16, 120, 1280)`, `B = CUDA.randn(Float16, 1280, 24)`,
   `C = CUDA.zeros(Float16, 120, 24)`) and run
   `display(CUDA.@profile split_time(() -> mul32!(C, A, B); n = 1, reps = 1))`.
   Read the **device-side** table: which kernel names appear, and how many per call?
   (`split_time` calls `f` twice here, once to warm up and once measured.) Then compare
   the kernels' own durations with G3's measured time in step 4. Where did the rest of
   the time go?
4. Change the formula to `max(FLOPs / (eff·F), bytes / BW, g)` (no `+ g`). Which rows
   improve, and which get worse?

## Checklist

- [ ] steps 1–5 run; the step 1 device line and the step 4 shapes match exactly
- [ ] I can explain why parking the GPU separates host time from GPU time
- [ ] I can say which of guide 05's guesses was furthest off, and by how much
- [ ] I can explain why `mul32!` exists and what `ONE`/`ZERO` being constants saved
- [ ] I can read an einsum string and give its GEMM's M, K, N
- [ ] I can say which roof bounds the reference GEMMs at t = 1, at t = 32, and the one
      exception
- [ ] in studio terms: what the long song is for, and what the two clocks measure
