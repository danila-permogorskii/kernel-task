# 04 — The project in miniature: three cores and tokens

## Purpose

**Problem.** The same operator as the project, `y = W·x` with W given as a
**three-core tensor ring**, at sizes small enough to print. Then the same formulas at
the project's real sizes, to reproduce its numbers.

Everything here maps one-to-one onto `reference.py`:

| This guide | Project |
|---|---|
| `dense_W` | `materialize_dense_weight` (lines 128-143) |
| `chain` | `tr_forward_reference` (lines 122-124: three einsums) |
| `cut_ring` | the kernel you would write in `submission.py` |
| `costs(8, 12, 20, 12, 10, 24, R)` | the default `TRSpec` |

## Mental model

```
               a                               x[i, j, k, t]
         ┌──────────────────────┐                 │
         A ───b─── B ───c─── C  │        contraction 1 (A):  i → p     T1 [j,k,a,p,b, t]
         └──────────────────────┘                 │
                                         contraction 2 (B):  j → q     T2 [k,a,p,q,c, t]
   A[a,p,i,b]  B[b,q,j,c]  C[c,r,k,a]             │
                                         contraction 3 (C):  k → r     Y  [p,q,r, t]
   W = tr( A[:,p,i,:] · B[:,q,j,:] · C[:,r,k,:] )           closes c and a

   Open links at each stage:    T1: a, b     T2: a, c     Y: none
                                    └─ a stays open from the first core to the last
```

**Tokens.** `t` is a batch of independent inputs, the project's `tokens`. Nothing mixes
across `t`; it is simply an extra outer loop. The weights are reused for every token,
which is why dense gets relatively cheaper as `t` grows (guide 05).

### In the studio

Three grid desks now sit on the loop, one per digit, joined by three group buses:

```
                          bus a  (the loop back)
      ┌─────────────────────────────────────────────────────────┐
      ▼                                                         │
  ┌────────┐     bus b      ┌────────┐     bus c      ┌────────┐
  │ desk A │ ─────────────► │ desk B │ ─────────────► │ desk C │
  │ i → p  │                │ j → q  │                │ k → r  │
  └────────┘                └────────┘                └────────┘

  tape after A (T1):  labels  start a, now b      R² tracks per signal
  tape after B (T2):  labels  start a, now c      R² tracks per signal
  after C:            only trips that arrive back on their start line a are kept
```

The start label `a` is written at desk A and cannot be dropped until desk C closes the
loop. That is the checklist question "which link stays open from the first core to the
last".

**Songs** are the tokens `t`. The same knobs are used for every song, and no song leaks
into another. **One take** of the cut ring is one `(song t, start line a)` pair: that is
the outer loop of `cut_ring` in step 3.

Desk B is the busy one. For every start label `a` it takes R incoming lines `b` and
sends them to R outgoing lines `c`: R · R · R patch cords. That is the **R³** contraction
of step 4.

## Step 1 — the dense oracle

**FILE — create `04_ring3.jl`**

```julia
# 04_ring3.jl — Rung 4: the project in miniature (three cores, t tokens)
include("common.jl")
using LinearAlgebra, Random

#== STEP 1 ==#
Random.seed!(4)
const ni, nj, nk = 2, 3, 2          # input digits   (project: 8, 12, 20)
const np, nq, nr = 2, 2, 3          # output digits  (project: 12, 10, 24)
const R  = 2                        # rank           (project: 8 or 16)
const nt = 3                        # tokens         (project: 1, 8, 32)

const A = randn(R, np, ni, R)       # A[a, p, i, b]
const B = randn(R, nq, nj, R)       # B[b, q, j, c]
const C = randn(R, nr, nk, R)       # C[c, r, k, a]   ← a closes the ring
const X = randn(ni, nj, nk, nt)     # X[i, j, k, t]

function dense_W(A, B, C)
    W = zeros(np, nq, nr, ni, nj, nk)
    for k in 1:nk, j in 1:nj, i in 1:ni, r in 1:nr, q in 1:nq, p in 1:np
        W[p, q, r, i, j, k] = tr(A[:, p, i, :] * B[:, q, j, :] * C[:, r, k, :])
    end
    return reshape(W, np * nq * nr, ni * nj * nk)
end

println("STEP 1 — dense oracle")
const W = dense_W(A, B, C)
const Y_dense = W * reshape(X, ni * nj * nk, nt)          # [features_out, tokens]
@printf "  size(W) = %s   size(Y) = %s\n" string(size(W)) string(size(Y_dense))
```

The digit sizes are deliberately **all different where possible** (2, 3, 2 / 2, 2, 3).
If you mix up two loop bounds, equal sizes would hide the bug; unequal sizes give an
out-of-bounds error instead.

Tokens go **last** (`X[i, j, k, t]`) because Julia is column-major: each token's
features then sit next to each other in memory. The project puts `t` first because
PyTorch is row-major. It is the same layout seen through the two conventions.

**COMMAND**

```
julia 04_ring3.jl
```

**Expected output** (exact)

```
STEP 1 — dense oracle
  size(W) = (12, 12)   size(Y) = (12, 3)
```

## Step 2 — the chain: `reference.py:122-124` written out

**FILE — append to `04_ring3.jl`**

```julia
#== STEP 2 ==#
# The reference: three contractions, T1 and T2 stored.
function chain(A, B, C, X)
    T1 = zeros(nj, nk, R, np, R, nt)                       # T1[j,k,a,p,b,t]
    for t in 1:nt, b in 1:R, p in 1:np, a in 1:R, k in 1:nk, j in 1:nj
        s = 0.0
        for i in 1:ni
            s += X[i, j, k, t] * A[a, p, i, b]
        end
        T1[j, k, a, p, b, t] = s
    end
    T2 = zeros(nk, R, np, nq, R, nt)                       # T2[k,a,p,q,c,t]
    for t in 1:nt, c in 1:R, q in 1:nq, p in 1:np, a in 1:R, k in 1:nk
        s = 0.0
        for b in 1:R, j in 1:nj
            s += T1[j, k, a, p, b, t] * B[b, q, j, c]
        end
        T2[k, a, p, q, c, t] = s
    end
    Y = zeros(np, nq, nr, nt)                              # Y[p,q,r,t]
    for t in 1:nt, r in 1:nr, q in 1:nq, p in 1:np
        s = 0.0
        for c in 1:R, a in 1:R, k in 1:nk
            s += T2[k, a, p, q, c, t] * C[c, r, k, a]
        end
        Y[p, q, r, t] = s
    end
    return reshape(Y, np * nq * nr, nt), T1, T2
end

println("\nSTEP 2 — chain (reference style)")
Y_chain, T1, T2 = chain(A, B, C, X)
@printf "  max |dense - chain| = %.2e\n" maximum(abs.(Y_dense .- Y_chain))
@printf "  |X| = %d   |T1| = %d   |T2| = %d   |Y| = %d\n" length(X) length(T1) length(T2) length(Y_chain)
```

Read each block against the project's einsum strings. The letters are the same; only
the position of `t` differs:

```
 project  "tijk,apib->tjkapb"      here  T1[j,k,a,p,b,t] = Σ_i     X · A
 project  "tjkapb,bqjc->tkapqc"    here  T2[k,a,p,q,c,t] = Σ_{j,b} T1 · B
 project  "tkapqc,crka->tpqr"      here  Y[p,q,r,t]      = Σ_{k,a,c} T2 · C
```

On a GPU, each block is one GEMM, plus the permute-copies PyTorch adds to bring the
summed letters together. Here the loops read the arrays in any order, so no copies are
needed. That is the one thing this CPU model leaves out.

**COMMAND**

```
julia 04_ring3.jl
```

**Expected output**

```
STEP 2 — chain (reference style)
  max |dense - chain| = 1.42e-14
  |X| = 36   |T1| = 144   |T2| = 96   |Y| = 36
```

`|T1|` is 4× `|X|`, and 4 = R², exactly as in guide 03: two open links, `a` and `b`.

## Step 3 — cut the ring at `a`

**FILE — append to `04_ring3.jl`**

```julia
#== STEP 3 ==#
# Cut the ring at a. Per token and per a, only ONE link is open at a time.
function cut_ring(A, B, C, X)
    Y  = zeros(np, nq, nr, nt)
    S1 = zeros(nj, nk, np, R)                              # S1[j,k,p,b]
    S2 = zeros(nk, np, nq, R)                              # S2[k,p,q,c]
    for t in 1:nt, a in 1:R
        for b in 1:R, p in 1:np, k in 1:nk, j in 1:nj
            s = 0.0
            for i in 1:ni
                s += X[i, j, k, t] * A[a, p, i, b]
            end
            S1[j, k, p, b] = s
        end
        for c in 1:R, q in 1:nq, p in 1:np, k in 1:nk
            s = 0.0
            for b in 1:R, j in 1:nj
                s += S1[j, k, p, b] * B[b, q, j, c]
            end
            S2[k, p, q, c] = s
        end
        for r in 1:nr, q in 1:nq, p in 1:np
            s = 0.0
            for c in 1:R, k in 1:nk
                s += S2[k, p, q, c] * C[c, r, k, a]
            end
            Y[p, q, r, t] += s                             # Σ over a
        end
    end
    return reshape(Y, np * nq * nr, nt), S1, S2
end

println("\nSTEP 3 — cut the ring")
Y_cut, S1, S2 = cut_ring(A, B, C, X)
@printf "  max |dense - cut_ring| = %.2e\n" maximum(abs.(Y_dense .- Y_cut))
@printf "  per token: T1 %d → S1 %d,   T2 %d → S2 %d   (÷R)\n" length(T1) ÷ nt length(S1) length(T2) ÷ nt length(S2)
```

The picture of one pass of the outer loop, a single `(t, a)` pair:

```
   X[:,:,:,t] ──A[a,…]──► S1[j,k,p,b] ──B──► S2[k,p,q,c] ──C[…,a]──► += Y[:,:,:,t]
                          one link (b)        one link (c)           closes c; a was fixed
```

`S1` and `S2` are reused on every pass. In a GPU kernel, one program would handle one
`(t, a)` pair (or a tile of them) and keep `S1`/`S2` in shared memory or registers.

**COMMAND**

```
julia 04_ring3.jl
```

**Expected output**

```
STEP 3 — cut the ring
  max |dense - cut_ring| = 7.11e-15
  per token: T1 48 → S1 24,   T2 32 → S2 16   (÷R)
```

## Step 4 — the same formulas at the project's sizes

**FILE — append to `04_ring3.jl`**

```julia
#== STEP 4 ==#
# The same formulas at the real project sizes.
function costs(ni, nj, nk, np, nq, nr, R)
    t1 = nj * nk * R * np * R                    # T1 per token
    t2 = nk * R * np * nq * R                    # T2 per token
    f1 = 2 * t1 * ni                             # sum over i
    f2 = 2 * t2 * nj * R                         # sum over j, b
    f3 = 2 * np * nq * nr * nk * R * R           # sum over k, c, a
    cores = R * R * (np * ni + nq * nj + nr * nk)
    dense = (np * nq * nr) * (ni * nj * nk)
    return (; t1, t2, flops = f1 + f2 + f3, f2, cores, dense)
end

println("\nSTEP 4 — the project's numbers from the same formulas")
println("     R   cores     T1/token   T2/token   FLOPs/token   dense FLOPs/token")
for Rp in (8, 16)
    c = costs(8, 12, 20, 12, 10, 24, Rp)
    @printf "  %4d  %7d   %9d  %9d   %11d   %11d\n" Rp c.cores c.t1 c.t2 c.flops 2c.dense
end
```

Each FLOP formula is "size of the output of that contraction × number of terms summed × 2":

```
 f1:  output T1 (t1 values)     × sums over i      (ni terms)     × 2
 f2:  output T2 (t2 values)     × sums over j, b   (nj·R terms)   × 2    ← R² · R = R³
 f3:  output Y  (np·nq·nr)      × sums over k,c,a  (nk·R·R terms) × 2
```

`(; t1, t2, …)` builds a *named tuple*, so the caller can write `c.t1`, `c.flops`.

**COMMAND**

```
julia 04_ring3.jl
```

**Expected output** (exact)

```
STEP 4 — the project's numbers from the same formulas
     R   cores     T1/token   T2/token   FLOPs/token   dense FLOPs/token
     8    44544      184320     153600      39813120      11059200
    16   178176      737280     614400     277217280      11059200
```

Check these against the project yourself:

- `cores` at R = 8 is `TRSpec(rank=8).factor_parameters`. It must be 44,544.
- FLOPs/token: 39.8 M at R = 8 and 277 M at R = 16, against dense's 11.1 M. The ring does
  **3.6×** and **25×** more arithmetic than dense.
- Doubling R multiplies FLOPs by about 7, not 4: contraction 2 grows as **R³**.

## What is *not* true

- *"The ring is compressed, so it is cheaper to compute."* Step 4 says the opposite: it
  is cheaper to **store** (44,544 vs 5,529,600 numbers) but dearer to **compute**
  (39.8 M vs 11.1 M FLOPs per token). Compression moved the cost from bytes to arithmetic.
  In the studio: the three desks have far fewer knobs than the giant board, but every
  signal goes the long way round the loop, through R² patch-bay routes at each desk.
- *"This CPU code shows the GPU bottleneck."* It shows the **sizes** and **counts**. It
  does not show kernel launches, host dispatch or HBM traffic. Those need a model of the
  machine, which is guide 05.

## Exercises

1. Set `nt = 1`. Which printed numbers change, and which stay the same? Why?
2. In `costs`, which of `f1`, `f2`, `f3` is largest at R = 8? What fraction of the total
   is it? This is the contraction the project's brief calls "the main bottleneck"
   candidate on FLOPs alone.
3. Compute, by hand, the size of `T1` in **bytes** for R = 16 and t = 32 in FP16
   (2 bytes per value). Compare it with the dense W in FP16 (5,529,600 × 2 bytes).

## Checklist

- [ ] steps 1–4 match; step 4 exactly
- [ ] I can match each loop block to its einsum string in `reference.py`
- [ ] I can say which link stays open from the first core to the last, and why
- [ ] I can say why FLOPs grow as ~R³, and which contraction causes it
- [ ] I can state the compression trade in one line: fewer bytes, more FLOPs
- [ ] I can draw the three-desk loop from memory, with the two labels on T1 and on T2
