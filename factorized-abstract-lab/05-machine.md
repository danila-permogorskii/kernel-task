# 05 — The abstract machine: predicting the benchmark table

## Purpose

**Problem.** Guides 01–04 counted **FLOPs** and **sizes**. They cannot say how long
something takes on a GPU, because time also depends on the machine: bandwidth, compute
throughput, and the fixed cost of every kernel launch. This guide builds the smallest
machine model that turns counts into time, and uses it to **predict the project's
results table before touching a GPU**.

The prediction is not the goal. The goal is a list of **assumptions you must measure**,
and a clear statement of which one decides each case. That list is what your report
defends in the 45-minute discussion.

## Mental model

```
    HOST (CPU, Python)                 GPU
    ──────────────────                 ──────────────────────────────────────────
    dispatch op 1  ─── H ───►          kernel 1: max(FLOPs / (eff·F), bytes / BW) + g
    dispatch op 2  ─── H ───►          kernel 2: …
    …                                  …
    host time = ops × H                gpu time = Σ kernel times

              time per call  =  max( host time , gpu time )
                                 └── the slower side sets the pace ──┘
```

Five parameters, all written down, all replaceable:

| Symbol | Meaning | Value used | Source |
|---|---|---|---|
| `BW` | HBM bandwidth | 3.35 TB/s | H100 SXM spec sheet |
| `F` | FP16 Tensor Core peak | 989 TFLOP/s | H100 SXM spec sheet (dense) |
| `eff` | fraction of peak a small kernel reaches | 0.30 | **guess** |
| `H` | host cost to dispatch one op | 5 µs | **guess** |
| `g` | GPU-side gap per kernel | 1 µs | **guess** |

Three of the five are guesses. That is the honest state before measuring, and your
`batch1-cdna` rule applies: *the denominator is measured, not quoted*. When you have
the H100, the profiler trace gives you `H`, `g` and per-kernel times, and every guess
in this table gets replaced.

### In the studio

Guides 01–04 designed the desks. This guide adds the **people and the equipment**
that run them:

```
  ENGINEER (host)                    STUDIO HARDWARE (GPU)
  ───────────────                    ─────────────────────────────────────────────────
  presses a button    ── H ──►       one PASS (a kernel):
  for every pass                       mixing work ÷ desk speed      (FLOPs / eff·F)
                                       or tape read+write ÷ tape speed (bytes / BW),
                                       whichever is slower, + rethreading the tape (g)

                                     TAPE MACHINE (HBM): where tracks go between passes
                                     INSIDE THE DESK (on-chip): tracks that never hit tape
```

The four methods of this guide, as ways to run a session for one song:

```
 dense    1 button:  one pass that reads the giant board's knobs (11 MB) for the song
 chain    9 buttons: 3 desk passes, each bouncing its tracks (T1, T2) to tape, plus
                     6 re-spooling passes that copy a tape into a new order (permutes)
 fused1   1 button:  all three desks in one live pass; the R takes are summed inside
                     the desk; T1/T2 never touch tape
 fused2   2 buttons: pass 1 records the R takes to tape, pass 2 mixes them down
```

`time = max(host, gpu)` in one sentence: **if the engineer presses buttons more slowly
than the hardware plays, the studio waits for the engineer.** More songs per session
(tokens `t`) share one reading of the knobs but multiply the mixing work.

## Step 1 — one kernel on the machine

**FILE — create `05_machine.jl`**

```julia
# 05_machine.jl — Rung 5: an abstract machine that predicts the benchmark
include("common.jl")

#== STEP 1 ==#
# The machine: every number here is an ASSUMPTION you will replace with a measurement.
Base.@kwdef struct Machine
    BW  = 3.35e12      # slow-memory bandwidth, bytes/s        (H100 SXM HBM3 spec)
    F   = 989e12       # peak FP16 tensor FLOP/s               (H100 SXM spec, dense)
    eff = 0.30         # fraction of peak a small kernel gets  (guess)
    H   = 5e-6         # host cost to dispatch one op, s       (guess)
    g   = 1e-6         # GPU-side gap per kernel, s            (guess)
end

# A kernel is just (FLOPs, bytes moved to/from slow memory).
kernel_time(m::Machine, flops, bytes) = max(flops / (m.eff * m.F), bytes / m.BW) + m.g

# The host dispatches while the GPU runs: the slower of the two sets the pace.
function call_time(m::Machine, host_ops, kernels)
    gpu = sum(kernel_time(m, f, b) for (f, b) in kernels)
    return max(host_ops * m.H, gpu), host_ops * m.H, gpu
end

println("STEP 1 — one kernel on the machine")
m = Machine()
@printf "  read 11.06 MB, tiny math : %s\n" us(kernel_time(m, 1e3, 11.06e6))
@printf "  40 MFLOP, tiny memory    : %s\n" us(kernel_time(m, 40e6, 1e3))
```

Notes:

- `Base.@kwdef` gives every field a default, so `Machine()` uses the table above and
  `Machine(H = 0.0)` changes just one parameter. Step 5 uses this.
- `kernel_time` is a **roofline** in one line: a kernel is limited either by math or by
  memory, whichever is slower, plus a fixed gap.
- `call_time` returns three numbers: total, host part and GPU part, so you can see
  **which side** is the bottleneck.

**COMMAND**

```
julia 05_machine.jl
```

**Expected output** (exact: pure arithmetic)

```
STEP 1 — one kernel on the machine
  read 11.06 MB, tiny math : 4.3 µs
  40 MFLOP, tiny memory    : 1.1 µs
```

The first line is the dense weight read at t = 1: 11.06 MB ÷ 3.35 TB/s = 3.3 µs, plus
the 1 µs gap. The second is the ring's R = 8 arithmetic for one token: 40 M ÷
(0.3 × 989 T) = 0.13 µs, plus the gap. **On paper the ring's math is 25× faster than
dense's memory read.** Whether that survives depends on everything else.

## Step 2 — dense and the reference as lists of kernels

**FILE — append to `05_machine.jl`**

```julia
#== STEP 2 ==#
# Problem sizes (project defaults), 2 bytes per FP16 value.
const ni, nj, nk = 8, 12, 20
const np, nq, nr = 12, 10, 24
const K = ni * nj * nk                      # 1920 inputs
const N = np * nq * nr                      # 2880 outputs
const e = 2                                 # bytes per value

function sizes(R)
    A  = R * np * ni * R
    B  = R * nq * nj * R
    C  = R * nr * nk * R
    t1 = nj * nk * R * np * R               # per token
    t2 = nk * R * np * nq * R               # per token
    f1 = 2 * t1 * ni
    f2 = 2 * t2 * nj * R
    f3 = 2 * N * nk * R * R
    return (; A, B, C, t1, t2, f1, f2, f3)
end

dense(m, R, t) = call_time(m, 1, [(2.0 * N * K * t, e * (N * K + K * t + N * t))])

function chain(m, R, t)                     # what the reference launches
    s = sizes(R)
    ks = [
        (0.0,       2e * K * t),                           # copy x
        (0.0,       2e * s.A),                             # copy A
        (s.f1 * t,  e * (K * t + s.A + s.t1 * t)),         # GEMM 1 → T1
        (0.0,       2e * s.t1 * t),                        # copy T1
        (0.0,       2e * s.B),                             # copy B
        (s.f2 * t,  e * (s.t1 * t + s.B + s.t2 * t)),      # GEMM 2 → T2
        (0.0,       2e * s.t2 * t),                        # copy T2
        (0.0,       2e * s.C),                             # copy C
        (s.f3 * t,  e * (s.t2 * t + s.C + N * t)),         # GEMM 3 → y
    ]
    return call_time(m, 9, ks)
end

println("\nSTEP 2 — dense vs reference chain, R = 8, t = 1")
for (name, f) in (("dense", dense), ("chain", chain))
    total, host, gpu = f(m, 8, 1)
    @printf "  %-6s total %8s   (host %8s, gpu %8s)\n" name us(total) us(host) us(gpu)
end
```

How each kernel's bytes are counted:

- A **copy** reads everything once and writes everything once: `2e × size`.
- A **GEMM** reads its two inputs and writes its output once: `e × (in1 + in2 + out)`.
- The 9-kernel list is the "copy, copy, GEMM" pattern from each einsum. **It is an
  assumption.** The real list comes from the project's token-1 profiler trace, and may
  have fewer copies if PyTorch finds a layout that avoids one.

**COMMAND**

```
julia 05_machine.jl
```

**Expected output** (exact)

```
STEP 2 — dense vs reference chain, R = 8, t = 1
  dense  total   5.0 µs   (host   5.0 µs, gpu   4.3 µs)
  chain  total  45.0 µs   (host  45.0 µs, gpu   9.9 µs)
```

**Read the bracket, not the total.** For the chain, the GPU needs 9.9 µs but the host
needs 45 µs to dispatch nine ops. **The GPU sits idle most of the time, waiting for
Python.** In the studio: the hardware finishes each pass and waits for the engineer's
next button press. This is what "explain the token-1 profiler evidence" in the project's README
is about: in the trace you would see short kernels separated by gaps.

## Step 3 — the fused kernel, two ways to finish the Σ over `a`

**FILE — append to `05_machine.jl`**

```julia
#== STEP 3 ==#
# Fused "cut the ring": T1/T2 never touch slow memory.
function fused_1k(m, R, t)                  # one kernel, partial sums combined in-kernel
    s = sizes(R)
    flops = (s.f1 + s.f2 + s.f3) * t
    bytes = e * (K * t + s.A + s.B + s.C + N * t)
    return call_time(m, 1, [(flops, bytes)])
end

function fused_2k(m, R, t)                  # kernel 1 writes R partials (FP32), kernel 2 sums
    s = sizes(R)
    flops = (s.f1 + s.f2 + s.f3) * t
    k1 = (flops, e * (K * t + s.A + s.B + s.C) + 4 * R * N * t)
    k2 = (1.0 * R * N * t, 4 * R * N * t + e * N * t)
    return call_time(m, 2, [k1, k2])
end
```

These are the two options from guide 03, step 4, for adding the R chains together:

```
 fused_1k:  one kernel; the R pieces are combined inside it (e.g. atomics into FP32)
            → 1 host op, 1 kernel;  arithmetic order may vary run to run
 fused_2k:  kernel 1 writes R partial outputs in FP32, kernel 2 adds them
            → 2 host ops, 2 kernels;  deterministic; R·N·t·4 extra bytes of workspace
```

Both keep `T1`/`T2` on-chip. The only slow-memory traffic is x, the cores, y and (for
`fused_2k`) the partials. This step prints nothing; step 4 uses it.

**COMMAND**

```
julia 05_machine.jl
```

**Expected output**: the same as after step 2. If you see an error instead, the most
likely cause is a typo in a field name (`s.f1`, `s.A`) that matches nothing in `sizes`.

## Step 4 — the predicted table

**FILE — append to `05_machine.jl`**

```julia
#== STEP 4 ==#
const methods = (("dense", dense), ("chain", chain), ("fused1", fused_1k), ("fused2", fused_2k))
const cases = ((8, 1), (8, 8), (8, 32), (16, 1), (16, 32))

function table(m)
    println("   R   t |    dense     chain    fused1    fused2 | fused1 vs dense")
    for (R, t) in cases
        ts = [f(m, R, t)[1] for (_, f) in methods]
        @printf "  %2d  %2d | %8s  %8s  %8s  %8s | %5.2f×\n" R t us(ts[1]) us(ts[2]) us(ts[3]) us(ts[4]) ts[1] / ts[3]
    end
end

println("\nSTEP 4 — predicted table, default machine  (>1× means fused1 is faster)")
table(m)
```

`cases` is exactly the project's five required GPU cases.

**Predict first:** in which of the five rows will `fused1` beat dense?

**COMMAND**

```
julia 05_machine.jl
```

**Expected output** (exact)

```
STEP 4 — predicted table, default machine  (>1× means fused1 is faster)
   R   t |    dense     chain    fused1    fused2 | fused1 vs dense
   8   1 |   5.0 µs   45.0 µs    5.0 µs   10.0 µs |  1.00×
   8   8 |   5.0 µs   45.0 µs    5.0 µs   10.0 µs |  1.00×
   8  32 |   5.0 µs   45.0 µs    5.3 µs   10.0 µs |  0.94×
  16   1 |   5.0 µs   45.0 µs    5.0 µs   10.0 µs |  1.00×
  16  32 |   5.0 µs  112.8 µs   30.9 µs   33.7 µs |  0.16×
```

**Probably not what you predicted.** Under these assumptions:

- The fused kernel is **9× faster than the reference** in the small cases (45 → 5 µs),
  which is a real result worth reporting.
- It **ties** dense in four of five cases, and ties at exactly 5.0 µs. That is the host
  dispatch cost `H`. Both are **host-bound**: one Python op each, and the GPU finishes
  before the next op arrives.
- `fused2` is 2× slower than `fused1` at small t for no GPU reason at all: it simply
  dispatches two ops.
- At R = 16, t = 32 the extra arithmetic becomes real. `fused1` is GPU-bound at 30.9 µs
  and loses to dense by 6×.

## Step 5 — which assumption decides the answer?

**FILE — append to `05_machine.jl`**

```julia
#== STEP 5 ==#
println("\nSTEP 5a — no host overhead (H = 0, like CUDA-graph replay)")
table(Machine(H = 0.0))
println("\nSTEP 5b — a worse kernel (eff = 0.10)")
table(Machine(eff = 0.10))
```

**COMMAND**

```
julia 05_machine.jl
```

**Expected output** (exact)

```
STEP 5a — no host overhead (H = 0, like CUDA-graph replay)
   R   t |    dense     chain    fused1    fused2 | fused1 vs dense
   8   1 |   4.3 µs    9.9 µs    1.1 µs    2.2 µs |  3.79×
   8   8 |   4.3 µs   15.6 µs    2.1 µs    3.3 µs |  2.09×
   8  32 |   4.4 µs   35.1 µs    5.3 µs    7.2 µs |  0.83×
  16   1 |   4.3 µs   12.6 µs    1.9 µs    3.0 µs |  2.23×
  16  32 |   4.4 µs  112.8 µs   30.9 µs   33.7 µs |  0.14×

STEP 5b — a worse kernel (eff = 0.10)
   R   t |    dense     chain    fused1    fused2 | fused1 vs dense
   8   1 |   5.0 µs   45.0 µs    5.0 µs   10.0 µs |  1.00×
   8   8 |   5.0 µs   45.0 µs    5.0 µs   10.0 µs |  1.00×
   8  32 |   5.0 µs   45.0 µs   13.9 µs   15.8 µs |  0.36×
  16   1 |   5.0 µs   45.0 µs    5.0 µs   10.0 µs |  1.00×
  16  32 |   5.0 µs  163.3 µs   90.7 µs   93.5 µs |  0.06×
```

**5a: remove host overhead, and the compression pays off.** Once the host is not the
limit, the GPU-side picture from guide 04 appears: `fused1` beats dense 2–4× at small t,
because it reads ~0.1 MB instead of 11 MB. The crossover sits between t = 8 and t = 32
at R = 8, and dense wins at R = 16, t = 32.

In the studio, `H = 0` is a **recorded macro**: the engineer records the whole button
sequence once, and later one press replays all of it. That is what a CUDA graph is.

This is why the project treats CUDA graphs as **optional and separate**. The required
runs are uncaptured, so host cost is part of the answer, and a fused kernel's GPU-side
advantage can be hidden behind it.

**5b: a worse kernel only matters where the GPU is the bottleneck.** The small cases do
not move (still host-bound); the large ones get much worse. Kernel quality (tiling,
Tensor Core use) matters mostly at large R × t.

## The one-page summary this guide produces

```
 regime                         limited by           who wins (this model)
 ─────────────────────────────  ───────────────────  ─────────────────────────────────
 small t, uncaptured            host dispatch (H)    fused ≈ dense ≫ reference
 small t, host cost removed     dense: bytes         fused ≫ dense   (compression pays)
                                fused: gaps (g)
 large R × t                    fused: FLOPs         dense           (extra math is real)
```

Your **measurement plan** on the H100 follows from this: measure `H` and `g` from the
token-1 trace, measure your kernel's achieved FLOP/s for `eff`, and check whether the
real table lands where the model says. Where it does not, **the model is wrong
somewhere**, and finding where is the report.

## What is *not* true

- *"The model says the fused kernel ties dense, so it's not worth writing."* The model
  also says it beats the reference 9×, that its GPU-side time is 4× below dense at t = 1,
  and that the tie comes from the host, which the report can show with the trace. A
  well-explained tie is a valid result under the brief.
- *"These are predictions of the H100 numbers."* They are predictions **under five
  stated assumptions**, three of them guesses. The model leaves out L2 caching (the dense
  W is 11 MB, the H100 L2 is 50 MB, so a warm W may never touch HBM), kernel-internal
  inefficiency beyond `eff`, and Triton's own Python launch overhead, which can exceed a
  plain PyTorch op.

## Exercises

1. Add L2 to the model: if a kernel's total bytes are below 50 MB, use a faster
   bandwidth for it (try 3× BW). Which rows change? Does dense get better or worse
   relative to `fused1`? (In the studio: a shelf of tapes next to the desk, faster to
   reach than the tape store. Whose tapes fit on the shelf?)
2. Set `H = 20e-6` to model a heavy Python launch path. What happens to `fused2`?
3. Find the smallest t at which `fused1` loses to dense at R = 8 in the `H = 0` machine.
   Compare with the break-even formula `t* ≈ dense_time ÷ (FLOPs_per_token / (eff·F))`.
4. The model's `chain` has 9 kernels. If the real trace shows 6, which three would you
   remove from `ks`, and how much does the chain row change?

## Checklist

- [ ] steps 1–5 match exactly
- [ ] I can explain `max(host, gpu)` and read which side is the bottleneck
- [ ] I can say why fused1 and dense tie at 5.0 µs, and what removes the tie
- [ ] I can say why fused2 costs 2× at small t, and what it buys (determinism)
- [ ] I can list the three guessed parameters and how the H100 trace would measure each
- [ ] I can tell the whole project as one studio story: mics, grid desks on a loop, songs,
      takes, the tape machine, and the engineer's button presses
