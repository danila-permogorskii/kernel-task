# 01 — Low rank: the order of operations

## Purpose

**Problem.** A matrix `M` (n × n) is given as a product of two thin matrices,
`M = U · V`, with `U` n × r and `V` r × n, and r ≪ n. Compute `y = M · x`.

There are two ways to bracket it, and they give the same answer at very different cost.
This is the simplest form of the project's central rule, *"do not construct the complete
dense matrix, even temporarily"* (README.md of the project).

## Mental model

```
  BAD ORDER:  (U · V) · x                       GOOD ORDER:  U · (V · x)

   U        V              M                     V       x        z        U       z      y
  ┌─┐   ┌───────┐      ┌───────┐                ┌───────┐ ┌─┐     ┌─┐     ┌─┐    ┌─┐   ┌─┐
  │ │ · └───────┘  =   │       │  · x           └───────┘·│ │  =  └─┘     │ │  · └─┘ = │ │
  │ │                  │ n × n │                 r × n    │ │    r × 1    │ │  r×1     │ │
  └─┘                  └───────┘                          └─┘            └─┘          └─┘
  n×r                  built!                                           n×r
                    n²·r + n² work                               2·r·n + 2·n·r work
                    n² numbers stored                            r numbers stored
```

The lesson in one line: **a recipe is only cheap if you never cook the whole dish.**

### In the studio

`x` is n microphones, `y` is n speakers, and between them sit **two mixing desks** with
only r **audio groups** in the middle.

```
 MICROPHONES        MIXER V          GROUPS          MIXER U          SPEAKERS
      x                               z = V·x                          y = U·z

  Mic 1 ───┐                                                     ┌──► Speaker 1
  Mic 2 ───┼──►  [ V: r × n ]  ──►  Group 1  ──►  [ U: n × r ] ──┼──► Speaker 2
  Mic 3 ───┤                        Group 2                      ├──► Speaker 3
  Mic 4 ───┘                                                     └──► Speaker 4

  V collects many inputs into a few groups; U distributes the groups to many outputs.
```

Every number is a **volume knob**. Row `g` of `V` says how much of each microphone
goes into group `g`; row `s` of `U` says how much of each group goes to speaker `s`.

```
                 Mic 1  Mic 2  Mic 3  Mic 4                    Group 1  Group 2
   V:  Group 1    0.8    0.5    0.2    0.7        U:  Spk 1      1.0      0.2
       Group 2    0.1    0.6    0.9    0.3            Spk 2      0.7      0.5
                                                      Spk 3      0.1      1.0
   (2 × 4)·(4 × 1) = (2 × 1)                          Spk 4      0.4      0.8
                                                      (4 × 2)·(2 × 1) = (4 × 1)
```

The three functions in this guide are three ways to run the studio:

```
 good   play the song through V, then U           y = U·(V·x)     2·n·r knobs
        (the closest desk to x works first)

 bad    before the song, work out a knob for      y = (U·V)·x     builds n × n knobs,
        EVERY mic → speaker pair, then play                        every call

 dense  buy a giant board with one knob per       y = M·x         n × n knobs,
        mic → speaker pair, set it once, then                     read on every call
        read all of it on every song
```

At the sizes of this guide (n = 2000, r = 8): the giant board has 4,000,000 knobs, the
two desks 32,000. That ratio is the whole point of step 4.

## Step 1 — same answer, two orders

**FILE — create `01_lowrank.jl`**

```julia
# 01_lowrank.jl — Rung 1: the order of operations
include("common.jl")
using LinearAlgebra, Random

#== STEP 1 ==#
Random.seed!(1)
const n = 2000
const r = 8
const U = randn(n, r)        # tall and thin
const V = randn(r, n)        # short and wide
const x = randn(n)

bad(U, V, x)  = (U * V) * x  # cooks the whole n×n dish first
good(U, V, x) = U * (V * x)  # never builds the n×n matrix

println("STEP 1 — do both orders give the same y?")
@printf "  max |bad - good| = %.2e\n" maximum(abs.(bad(U, V, x) .- good(U, V, x)))
```

Notes:

- `const` tells Julia these globals never change type, so functions using them compile
  to fast code. Without it, timings in step 3 would measure Julia's dynamic dispatch.
- `bad` and `good` are one-line functions. Parentheses decide the order; Julia does
  **not** reorder a matrix product for you. Neither does PyTorch.
- `abs.(…)` with a dot applies `abs` to each element.

**COMMAND**

```
julia 01_lowrank.jl
```

**Expected output**

```
STEP 1 — do both orders give the same y?
  max |bad - good| = 9.09e-13
```

The number is approximate. Anything below `1e-10` means "equal up to rounding". The
two orders add the same products in a different sequence, and floating-point addition
is not associative, so a tiny difference is correct. **Exactly `0.0` would be suspicious.**

## Step 2 — predict the cost by counting

**FILE — append to `01_lowrank.jl`**

```julia
#== STEP 2 ==#
flops_bad(n, r)  = 2n * n * r + 2n * n   # build M, then M*x
flops_good(n, r) = 2r * n + 2n * r       # V*x, then U*(V*x)

println("\nSTEP 2 — predicted cost (count by formula)")
@printf "  bad : %12d FLOPs, intermediate M   = %s\n" flops_bad(n, r)  human(8n * n)
@printf "  good: %12d FLOPs, intermediate V*x = %s\n" flops_good(n, r) human(8r)
@printf "  ratio: %.0f×\n" flops_bad(n, r) / flops_good(n, r)
```

How the counts come from the shapes. A matrix product `(a × b)·(b × c)` costs
`2·a·b·c` FLOPs: one multiply and one add for each of the `b` terms in each of the `a·c`
outputs.

```
 bad:   U·V   = (n×r)·(r×n) → 2·n·r·n      M·x   = (n×n)·(n×1) → 2·n·n
 good:  V·x   = (r×n)·(n×1) → 2·r·n        U·v   = (n×r)·(r×1) → 2·n·r
```

`8n * n` is the size of M in bytes: n² values × 8 bytes per `Float64`.

**COMMAND**

```
julia 01_lowrank.jl
```

**Expected output** (after the step 1 lines; these values are exact)

```
STEP 2 — predicted cost (count by formula)
  bad :     72000000 FLOPs, intermediate M   = 30.5 MiB
  good:        64000 FLOPs, intermediate V*x = 64 B
  ratio: 1125×
```

## Step 3 — measure it

**FILE — append to `01_lowrank.jl`**

```julia
#== STEP 3 ==#
println("\nSTEP 3 — measured cost")
tb = best_time(() -> bad(U, V, x))
tg = best_time(() -> good(U, V, x))
ab = alloc_bytes(() -> bad(U, V, x))
ag = alloc_bytes(() -> good(U, V, x))
@printf "  bad : %10s   allocated %s\n" us(tb) human(ab)
@printf "  good: %10s   allocated %s\n" us(tg) human(ag)
```

**Predict first:** the FLOP ratio was 1125×. Will the time ratio be larger, smaller or
the same? Write your guess down.

**COMMAND**

```
julia 01_lowrank.jl
```

**Expected output** (times are machine-dependent; allocations are nearly exact)

```
STEP 3 — measured cost
  bad :    7.78 ms   allocated 30.5 MiB
  good:     5.2 µs   allocated 15.8 KiB
```

Measured on the reference laptop over three runs: bad 7.56–7.78 ms, good 4.9–5.2 µs,
so about 1500×. That is close to the FLOP ratio, but not equal to it, because the two
computations are limited by different things.

**Why is `good` allocating 15.8 KiB, not 64 B?** The output `y` itself is n = 2000 values
× 8 bytes = 15.6 KiB. Allocations count *everything created*, including the result. The
intermediate `V*x` adds only 64 bytes on top. This is the same accounting the project
uses in `incremental_workspace_and_output_bytes`: the output is included.

## Step 4 — the dense baseline: M built once and stored

**FILE — append to `01_lowrank.jl`**

```julia
#== STEP 4 ==#
const M = U * V                 # the "dense baseline": built ONCE, stored
dense(M, x) = M * x

println("\nSTEP 4 — dense (prebuilt M) vs factored")
@printf "  stored: dense M = %s   factors U,V = %s\n" human(sizeof(M)) human(sizeof(U) + sizeof(V))
@printf "  dense : %10s\n" us(best_time(() -> dense(M, x)))
@printf "  good  : %10s\n" us(best_time(() -> good(U, V, x)))
```

This mirrors the project's `dense` worker exactly: it pays `materialize_dense_weight`
once (at "preparation"), then every call is a single matrix-vector product reading all
of M.

**COMMAND**

```
julia 01_lowrank.jl
```

**Expected output**

```
STEP 4 — dense (prebuilt M) vs factored
  stored: dense M = 30.5 MiB   factors U,V = 250.0 KiB
  dense :   760.3 µs
  good  :     5.2 µs
```

Measured range over three runs: dense 749–943 µs, good 4.7–10.2 µs.

**Read this like your GEMV work.** `dense` does only 2n² = 8 M FLOPs, but it has to
stream 30.5 MiB through the memory system, which is larger than the L3 cache of this CPU
(16 MB on a 5800H). It is **memory-bound**: bytes ÷ bandwidth. `good` touches 250 KiB, which
stays in cache.

**In the studio:** the giant board needs no thinking per song, only a walk past four
million knobs. The two desks need a little arithmetic, but the engineer only reads 32,000
knobs. Reading knobs is the expensive part, so the desks win.

## What is *not* true

- *"The factored form is always faster."* Not in general. Here r = 8 is tiny compared
  with n = 2000. The factored form stores `2·n·r` numbers and does `4·n·r` FLOPs, which
  beats dense only while r < n/2. For low rank, that condition almost always holds. For
  the **ring** in guides 03–05 it does not, because the rank enters **squared**. Keep
  this contrast in mind: it is the whole reason the project is interesting.
- *"bad is slow because of FLOPs alone."* It also allocates and writes 30.5 MiB, which
  costs memory bandwidth on top of the math.
- *"The groups are a smaller copy of the song."* They are not. A group is whatever mix
  the knobs of `V` make; with random knobs it has no musical meaning. What is true is that
  the speakers can only ever play mixes of those r groups. That is what "rank r" means:
  `M` can produce at most r independent outputs, however many speakers there are.

## Exercises (predict, then change the code and run)

1. Set `r = 64`. Predict the new FLOP ratio in step 2 before running.
2. Make x a matrix with 32 columns: `const x = randn(n, 32)`. Does `good` still beat
   `dense` by the same factor? Why do both costs change in the way they do? (Hint: how
   many times is M read now?) In the studio: 32 **songs** go through the same desk. The
   engineer reads the knobs once and uses them 32 times. Which studio gains more from
   that, the giant board or the two desks?
3. Find the r at which `flops_good` equals `2n*n`, the cost of `dense`. Is the factored
   form still worth storing there? In the studio: how many groups does it take before the
   two desks have as many knobs as the giant board?

## Checklist

- [ ] steps 1–4 print the expected lines; the exact ones match exactly
- [ ] I can compute `2·a·b·c` for any matrix product from its shapes
- [ ] I can explain why `good` allocates 15.8 KiB and not 64 B
- [ ] I can say which of `dense` / `good` is memory-bound and why
- [ ] I know the one condition under which "factored beats dense" can fail
- [ ] I can tell `bad`, `good` and `dense` as three ways to run the studio, and say which
      desk touches `x` first in `y = U·(V·x)`
