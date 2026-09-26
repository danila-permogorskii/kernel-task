# 03 — The ring: links, the R² intermediate, and cutting the ring

## Purpose

**Problem.** Guide 02's Kronecker product is too rigid: one small matrix per digit, with
nothing connecting them. A **tensor ring** connects the per-digit pieces through
**links** of width R (the rank), joined in a circle. This lets it express far more
matrices, at a price: **the intermediate grows as R²**.

This guide uses **two** cores, which is the smallest ring. It contains every idea of the
project's three-core ring: the trace definition, the chain of contractions, the R² blow-up
of `T1`/`T2`, and the kernel idea ("cut the ring").

## Mental model

```
  Guide 02 (Kronecker):             This guide (ring, rank R):

    A[p,i]   B[q,j]                          a
    (numbers)                          ┌───────────┐
                                       A ────b──── B
    W = A[p,i] · B[q,j]                └───────────┘

                                    A[:,p,i,:] and B[:,q,j,:] are R×R matrices
                                    W = trace( A[:,p,i,:] · B[:,q,j,:] )
```

Core layout, with the letters used everywhere below:

```
   A[a, p, i, b]         B[b, q, j, a]
     │  │  │  │            │  │  │  │
     │  │  │  └ right link │  │  │  └ right link = A's left link: the ring closes
     │  │  └ input digit   │  │  └ input digit
     │  └ output digit     │  └ output digit
     └ left link           └ left link = A's right link
```

### In the studio

Take guide 02's two grid desks (A for rows, B for columns) and connect them with guide
01's idea: a **group bus**. The bus is a cable of R lines. Each desk takes the bus in
on one side and sends it out on the other, and the bus from B **loops back** into A.

```
                 bus line a  (the loop back)
        ┌────────────────────────────────────────┐
        ▼                                        │
   ┌─────────┐      bus line b            ┌─────────┐
   │ desk A  │ ─────────────────────────► │ desk B  │
   │ row i→p │                            │ col j→q │
   └─────────┘                            └─────────┘

   For every (p, i), desk A has a small R × R PATCH BAY  A[:, p, i, :]:
   "sound arriving on line a leaves on line b at volume A[a, p, i, b]".
```

The knob from mic `(i, j)` to speaker `(p, q)` is the sum over every **round trip**:
start on some line `a`, go through A to line `b`, through B back to line `a`. A trip
that comes back on a different line than it started on does not count. "Sum over the
trips that come home" is exactly `tr(A[:, p, i, :] · B[:, q, j, :])`.

Why the intermediate is R² (step 2). Halfway round the loop, after desk A, the
engineer must label every track with **two** things: the line it is on now (`b`), and
the line it started on (`a`), because only trips that return to their starting line
will count. R current lines × R starting lines = R² tracks for every signal.

Why cutting the ring works (step 4). Record the song in **R separate takes**. In take
`a`, sound may only enter the loop on line `a`, and only sound arriving back on line `a`
is kept. Inside one take the start label is fixed, so the tape only needs the current
line: R tracks, not R². The final mix-down adds the R takes (`Y += s`).

## Step 1 — the definition: every entry is a trace

**FILE — create `03_ring2.jl`**

```julia
# 03_ring2.jl — Rung 3: two cores joined in a ring
include("common.jl")
using LinearAlgebra, Random

#== STEP 1 ==#
Random.seed!(3)
const ni, nj = 4, 4                 # input digits
const np, nq = 4, 4                 # output digits
const R = 2                         # rank: width of each link

const A = randn(R, np, ni, R)       # A[a, p, i, b]   links a (left), b (right)
const B = randn(R, nq, nj, R)       # B[b, q, j, a]   links b (left), a (right)
const X = randn(ni, nj)             # input as a digit grid X[i, j]

# The definition: every entry of W is the trace of two small R×R matrices.
function dense_W(A, B)
    R, np, ni, _ = size(A)
    _, nq, nj, _ = size(B)
    W = zeros(np, nq, ni, nj)
    for j in 1:nj, i in 1:ni, q in 1:nq, p in 1:np
        W[p, q, i, j] = tr(A[:, p, i, :] * B[:, q, j, :])
    end
    return reshape(W, np * nq, ni * nj)
end

println("STEP 1 — the ring as a dense matrix")
const W = dense_W(A, B)
const y_dense = W * vec(X)
@printf "  size(W) = %s,  W[1,1] = %.4f\n" string(size(W)) W[1, 1]
@printf "  check W[1,1] = tr(A[:,1,1,:]·B[:,1,1,:]) = %.4f\n" tr(A[:, 1, 1, :] * B[:, 1, 1, :])
```

Notes:

- `A[:, p, i, :]` slices out the R×R matrix for one (output digit, input digit) pair:
  one cell of the "grid of little matrices".
- `for j in 1:nj, i in 1:ni, q in 1:nq, p in 1:np` is one loop nest; the **last** name
  is the innermost loop. `p` innermost matches column-major order (first index fastest).
- `dense_W` is this lab's `materialize_dense_weight`: allowed **only** as an oracle.

**COMMAND**

```
julia 03_ring2.jl
```

**Expected output** (the two numbers must be equal to each other; with seed 3 they are exactly these)

```
STEP 1 — the ring as a dense matrix
  size(W) = (16, 16),  W[1,1] = -0.6404
  check W[1,1] = tr(A[:,1,1,:]·B[:,1,1,:]) = -0.6404
```

## Step 2 — the chain: how the reference computes it

Written as sums, the ring gives:

```
   y[p,q] = Σ_{i,j,a,b}  A[a,p,i,b] · B[b,q,j,a] · X[i,j]
```

The reference-style way: contract one digit at a time, **storing** the intermediate.

```
   contraction 1:  T[a,p,j,b] = Σ_i        A[a,p,i,b] · X[i,j]        (i → p)
   contraction 2:  Y[p,q]     = Σ_{j,a,b}  T[a,p,j,b] · B[b,q,j,a]    (j → q, close a and b)
```

**FILE — append to `03_ring2.jl`**

```julia
#== STEP 2 ==#
# The reference way: two contractions, intermediate stored in between.
function chain(A, B, X)
    R, np, ni, _ = size(A)
    _, nq, nj, _ = size(B)
    T = zeros(R, np, nj, R)                      # T[a, p, j, b]: BOTH links open
    for b in 1:R, j in 1:nj, p in 1:np, a in 1:R
        s = 0.0
        for i in 1:ni                            # contraction 1: sum over i
            s += A[a, p, i, b] * X[i, j]
        end
        T[a, p, j, b] = s
    end
    Y = zeros(np, nq)
    for q in 1:nq, p in 1:np
        s = 0.0
        for a in 1:R, b in 1:R, j in 1:nj        # contraction 2: sum over j, b, a
            s += T[a, p, j, b] * B[b, q, j, a]
        end
        Y[p, q] = s
    end
    return vec(Y), T
end

println("\nSTEP 2 — the chain (reference style)")
y_chain, T = chain(A, B, X)
@printf "  max |dense - chain| = %.2e\n" maximum(abs.(y_dense .- y_chain))
@printf "  size(T) = %s  →  %d numbers  (X has %d)\n" string(size(T)) length(T) length(X)
```

This is an einsum written out by hand. Contraction 1 is
`einsum("ij,apib->apjb", X, A)`: every letter kept in the output is an outer loop, every
letter summed away is the inner loop with `s +=`.

**COMMAND**

```
julia 03_ring2.jl
```

**Expected output**

```
STEP 2 — the chain (reference style)
  max |dense - chain| = 2.66e-15
  size(T) = (2, 4, 4, 2)  →  64 numbers  (X has 16)
```

**The key observation.** In guide 02 the intermediate was the same size as x. Here it is
**4× larger**, and 4 = R². `T` has to carry **both** links, `a` and `b`, because neither
can be closed yet: `b` closes when B arrives, and `a` only when the ring comes all the
way round.

## Step 3 — how bad does it get?

**FILE — append to `03_ring2.jl`**

```julia
#== STEP 3 ==#
println("\nSTEP 3 — how the intermediate grows with R  (np = nj = 4)")
println("      R   |T| = R²·np·nj   |T| / |X|")
for Rt in (1, 2, 4, 8, 16)
    @printf "  %5d   %13d   %8.0f×\n" Rt Rt^2 * np * nj Rt^2 * np * nj / (ni * nj)
end
```

**COMMAND**

```
julia 03_ring2.jl
```

**Expected output** (exact)

```
STEP 3 — how the intermediate grows with R  (np = nj = 4)
      R   |T| = R²·np·nj   |T| / |X|
      1              16          1×
      2              64          4×
      4             256         16×
      8            1024         64×
     16            4096        256×
```

At the project's R = 16, the intermediate is **256×** the size of the input. On a GPU
that intermediate goes to slow memory (HBM) and comes back. This table is the project's
`T1`/`T2` problem in miniature.

## Step 4 — cut the ring

**The idea.** Fix the link `a` to one value. Then `a` is just a constant, and the ring
becomes a **chain** with only one open link (`b`). Do that for every `a` and add the results.

```
   RING: two open links in T              CHAIN for one fixed a: one open link

         ┌── A ──b── B ──┐                   A[a,·,·,:] ──b── B[:,·,·,a]
         └───────a───────┘                   (a is a constant here)

   y = Σ_a ( chain for that a )           ← R independent, small pieces of work
```

**FILE — append to `03_ring2.jl`**

```julia
#== STEP 4 ==#
# Cut the ring: fix a. Each slice is a plain chain with ONE open link (b).
function cut_ring(A, B, X)
    R, np, ni, _ = size(A)
    _, nq, nj, _ = size(B)
    Y  = zeros(np, nq)
    Ta = zeros(np, nj, R)                        # Ta[p, j, b]: only b open
    for a in 1:R                                 # ← the cut
        for b in 1:R, j in 1:nj, p in 1:np
            s = 0.0
            for i in 1:ni
                s += A[a, p, i, b] * X[i, j]
            end
            Ta[p, j, b] = s
        end
        for q in 1:nq, p in 1:np
            s = 0.0
            for b in 1:R, j in 1:nj
                s += Ta[p, j, b] * B[b, q, j, a]
            end
            Y[p, q] += s                         # sum of the R chains
        end
    end
    return vec(Y), Ta
end

println("\nSTEP 4 — cut the ring")
y_cut, Ta = cut_ring(A, B, X)
@printf "  max |dense - cut_ring| = %.2e\n" maximum(abs.(y_dense .- y_cut))
@printf "  slice size(Ta) = %s → %d numbers  (full T had %d)\n" string(size(Ta)) length(Ta) length(T)
```

Compare with `chain` line by line. The arithmetic is identical; only the loop order
changed, with `a` moved to the outside. That single change means the buffer `Ta` is
**reused** for every `a`, so it only ever needs to hold one slice.

**COMMAND**

```
julia 03_ring2.jl
```

**Expected output**

```
STEP 4 — cut the ring
  max |dense - cut_ring| = 1.78e-15
  slice size(Ta) = (4, 4, 2) → 32 numbers  (full T had 64)
```

The slice is **R times smaller** than the full intermediate: 32 vs 64 here. At R = 16
it would be 16× smaller. **This is the kernel idea:** on a GPU, each `a` becomes an
independent piece of work, its slice fits in fast on-chip memory, and nothing goes to
HBM. The price is the final `Y[p, q] += s`: the R pieces have to be **added together**.
On a GPU, parallel pieces adding into the same output is a real design problem (atomics,
or a second small kernel). Guide 05 puts a number on that choice.

## Step 5 — the ring contains the Kronecker product

**FILE — append to `03_ring2.jl`**

```julia
#== STEP 5 ==#
println("\nSTEP 5 — with R = 1 the ring is the Kronecker product from rung 2")
A1 = randn(1, np, ni, 1)
B1 = randn(1, nq, nj, 1)
K  = kron(B1[1, :, :, 1], A1[1, :, :, 1])
@printf "  max |dense_W(R=1) - kron| = %.2e\n" maximum(abs.(dense_W(A1, B1) .- K))
```

**COMMAND**

```
julia 03_ring2.jl
```

**Expected output** (exact)

```
STEP 5 — with R = 1 the ring is the Kronecker product from rung 2
  max |dense_W(R=1) - kron| = 0.00e+00
```

Exactly zero, and correctly so: with R = 1 the "trace of 1×1 matrices" is a single
multiplication, the same one `kron` does. In the studio: a bus with one line has no
routing choices, so each patch bay is a single knob and the studio is guide 02's
two grid desks again. **R is a dial:** R = 1 gives guide 02's rigid
Kronecker product; larger R gives more expressive matrices, bigger intermediates and more
FLOPs.

## What is *not* true

- *"Cutting the ring saves FLOPs."* It does not. `chain` and `cut_ring` do exactly the
  same multiplications. It saves **intermediate storage** (÷R) and makes the work split
  into R independent pieces. On a GPU, that is what lets the intermediate stay on-chip.
  In the studio: R takes of R tracks record the same sound as one take of R² tracks.
  The engineer does the same amount of mixing; only the tape used at any moment shrinks.
- *"You could cut at b instead of a and get the same benefit."* Try it in exercise 2 and
  look at which intermediate still has two open links.

## Exercises

1. Set `R = 8` in step 1. Predict `size(T)` and `size(Ta)` before running.
2. Write `cut_ring_b` that fixes `b` instead of `a`. Which intermediate would you need, and
   how many links does it carry? Why does the ring's closing link make `a` the natural cut?
3. Count FLOPs for `chain` by hand: contraction 1 is `2·(R·np·nj·R)·ni`. Write contraction
   2 the same way. With two cores both grow as R². In guide 04 (three cores) one
   contraction grows as **R³**. Predict which one, and why a third core makes it possible.

## Checklist

- [ ] steps 1–5 match; the step 1 pair are equal to each other; step 5 is exactly 0
- [ ] I can say why `T` carries R² and not R
- [ ] I can explain "cut the ring" in one sentence, and what it saves and does not save
- [ ] I can name the price of the cut: the final Σ over `a`
- [ ] I can say what R = 1 turns the ring into
- [ ] in studio terms: why a track halfway round the loop carries two labels, and what
      one "take" of the cut ring records
