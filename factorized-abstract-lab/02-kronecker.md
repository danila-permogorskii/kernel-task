# 02 — Kronecker: an index made of digits

## Purpose

**Problem.** The feature index `f` of a vector can be read as a number made of several
**digits**. Once you see the digits, a big matrix acting on `f` can sometimes be replaced
by small matrices acting on **one digit each**.

This is what the project does on its first line of real work:

```python
x_modes = x.reshape(x.shape[0], *spec.input_modes)   # [t, 1920] → [t, 8, 12, 20]
```

It is free (no data moves), and it is what makes every later contraction possible.

## Mental model

```
  a phone number   8-495-123   = one number, read as three digit groups

  a feature index  f ∈ 1:12    = one number, read as two digits (i, j)
                                  i ∈ 1:3 changes fastest, j ∈ 1:4 slowest

   x as a line:    [ x1 x2 x3 | x4 x5 x6 | x7 x8 x9 | x10 x11 x12 ]
                     └ j=1 ─┘   └ j=2 ─┘   └ j=3 ─┘   └─ j=4 ───┘

   x as a grid:          j=1  j=2  j=3  j=4
                   i=1 [  x1   x4   x7  x10 ]
                   i=2 [  x2   x5   x8  x11 ]      same memory, new glasses
                   i=3 [  x3   x6   x9  x12 ]
```

When `M = B ⊗ A` (a **Kronecker product**), `A` acts only on the first digit and `B`
only on the second:

```
   y = M · x          (one big product, 1024 × 1024)
       ═════
   Y = A · X · Bᵀ     (two small products on the 32 × 32 grid)
       └─┬─┘
         T = A·X  ← the first INTERMEDIATE of this ladder
```

**Julia vs Python ordering.** Julia stores arrays column-major: the **first** index
changes fastest. PyTorch is row-major: the **last** index changes fastest. The project's
`reshape(t, 8, 12, 20)` therefore has `k` fastest; here `i` is fastest. The idea is
identical; only the naming of "fast digit" flips. Keep this in mind in guide 04.

### In the studio

The microphones now stand on a **stage grid**: 3 rows (`i`) × 4 columns (`j`). A mic's
number `f` is its seat number; `(i, j)` is its row and column. The speakers stand on a
grid too, rows `p` and columns `q`.

```
            column j=1  j=2  j=3  j=4
   row i=1     Mic1  Mic4  Mic7  Mic10        seat f = 1…12  ⇔  (row i, column j)
   row i=2     Mic2  Mic5  Mic8  Mic11        same microphones, read as a grid
   row i=3     Mic3  Mic6  Mic9  Mic12
```

Instead of one giant board (one knob per mic → speaker pair), the studio uses two
**small desks, one per grid direction**:

```
   X[i, j]  ──[ desk A: row i → row p ]──►  T[p, j]  ──[ desk B: column j → column q ]──►  Y[p, q]
            same knobs for every column               same knobs for every row
```

The knob from mic `(i, j)` to speaker `(p, q)` is never stored; it is always
`A[p, i] · B[q, j]`, a product of one knob from each desk. That is what `kron` builds.

Compare with guide 01. There, desk V **narrowed** n mics into r groups. Here desk A only
**re-mixes the rows**: `T` has one track per (row p, column j), as many tracks as there
were microphones. Nothing narrows, so nothing grows either.

## Step 1 — see the digits

**FILE — create `02_kronecker.jl`**

```julia
# 02_kronecker.jl — Rung 2: splitting an index into digits
include("common.jl")
using LinearAlgebra, Random

#== STEP 1 ==#
println("STEP 1 — a feature number is a 2-digit number")
const ni, nj = 3, 4                 # input digits: i ∈ 1:3 (fast), j ∈ 1:4 (slow)
xs = collect(1.0:ni*nj)             # x = [1, 2, …, 12], value = its own position
Xs = reshape(xs, ni, nj)            # the SAME memory, viewed as a grid Xs[i, j]
display(Xs)
for f in (1, 2, 4, 12)
    i = (f - 1) % ni + 1
    j = (f - 1) ÷ ni + 1
    @printf "  f = %2d  →  (i=%d, j=%d)   x[f] = %4.1f   X[i,j] = %4.1f\n" f i j xs[f] Xs[i, j]
end
```

Notes:

- The value stored at each position **is** the position, so you can read the mapping
  straight off the printout.
- `(f - 1) % ni + 1` and `(f - 1) ÷ ni + 1` are the digit formulas. The `-1`/`+1`
  appear because Julia counts from 1. Type `÷` as `\div` then Tab in the REPL, or use
  `div(f - 1, ni) + 1` in a plain editor.

**COMMAND**

```
julia 02_kronecker.jl
```

**Expected output** (exact)

```
STEP 1 — a feature number is a 2-digit number
3×4 Matrix{Float64}:
 1.0  4.0  7.0  10.0
 2.0  5.0  8.0  11.0
 3.0  6.0  9.0  12.0
  f =  1  →  (i=1, j=1)   x[f] =  1.0   X[i,j] =  1.0
  f =  2  →  (i=2, j=1)   x[f] =  2.0   X[i,j] =  2.0
  f =  4  →  (i=1, j=2)   x[f] =  4.0   X[i,j] =  4.0
  f = 12  →  (i=3, j=4)   x[f] = 12.0   X[i,j] = 12.0
```

## Step 2 — two small products equal one big product

**FILE — append to `02_kronecker.jl`**

```julia
#== STEP 2 ==#
Random.seed!(2)
const np, nq = 32, 32               # output digits
const Ni, Nj = 32, 32               # input digits
const A = randn(np, Ni)             # acts on digit 1:  p ← i
const B = randn(nq, Nj)             # acts on digit 2:  q ← j
const x = randn(Ni * Nj)

dense_M(A, B) = kron(B, A)          # full matrix: (np·nq) × (Ni·Nj)

function factored(A, B, x)
    X = reshape(x, Ni, Nj)          # digits: X[i, j]      (free, no copy)
    T = A * X                       # step 1: i → p        T[p, j]
    Y = T * transpose(B)            # step 2: j → q        Y[p, q]
    return vec(Y)                   # back to one index    (free, no copy)
end

println("\nSTEP 2 — two small matmuls on the grid = one big matmul")
const M = dense_M(A, B)
@printf "  size(M) = %s\n" string(size(M))
@printf "  max |dense - factored| = %.2e\n" maximum(abs.(M * x .- factored(A, B, x)))
```

Why `kron(B, A)` and not `kron(A, B)`: in Julia's column-major digits, `i` (the digit
`A` acts on) is the *fast* digit. `kron(B, A)` puts `A` on the fast digit. Swap them and
the printed difference becomes large, which is exercise 1.

The digit swap, one letter at a time:

```
   X[i, j]  ──A acts on i──►  T[p, j]  ──B acts on j──►  Y[p, q]
   input digits               one digit swapped         both digits swapped
```

**COMMAND**

```
julia 02_kronecker.jl
```

**Expected output**

```
STEP 2 — two small matmuls on the grid = one big matmul
  size(M) = (1024, 1024)
  max |dense - factored| = 9.24e-14
```

## Step 3 — the cost

**FILE — append to `02_kronecker.jl`**

```julia
#== STEP 3 ==#
println("\nSTEP 3 — cost: predicted and measured")
fl_dense = 2 * (np * nq) * (Ni * Nj)
fl_fact  = 2 * np * Ni * Nj + 2 * np * Nj * nq
@printf "  dense   : %9d FLOPs  stored %9s  %9s\n" fl_dense human(sizeof(M)) us(best_time(() -> M * x))
@printf "  factored: %9d FLOPs  stored %9s  %9s\n" fl_fact human(sizeof(A) + sizeof(B)) us(best_time(() -> factored(A, B, x)))
@printf "  intermediate T = A*X holds %d numbers (x holds %d)\n" np * Nj Ni * Nj
```

**COMMAND**

```
julia 02_kronecker.jl
```

**Expected output** (FLOPs, bytes and counts exact; times machine-dependent)

```
STEP 3 — cost: predicted and measured
  dense   :   2097152 FLOPs  stored   8.0 MiB   115.0 µs
  factored:    131072 FLOPs  stored  16.0 KiB     5.3 µs
  intermediate T = A*X holds 1024 numbers (x holds 1024)
```

Measured range over three runs: dense 77–129 µs, factored 4.9–5.3 µs.

**What to notice.** The intermediate `T` is **the same size as x**. With one factor per
digit and nothing linking the factors, the chain never grows. Hold on to this: in guide 03
the factors get *linked*, and the intermediate stops being the same size as x.

## What is *not* true

- *"reshape copies the data."* It does not; `X` and `x` share memory. The project's
  `x.reshape(...)` is also free. What costs something in the project is the
  **permute** that einsum does afterwards to reorder digits (guide 04).
- *"Any matrix can be split like this."* Only matrices that really are a Kronecker
  product. A general 1024 × 1024 matrix has 1,048,576 free numbers; `A` and `B` hold
  2,048. The ring in guide 03 is a way to express **more** matrices than a single
  Kronecker product, at a price. In the studio: with two desks of 1,024 knobs each you
  cannot give mic (1, 1) → speaker (1, 1) its own volume; changing `A[1, 1]` also changes
  every other mic in row 1, in every column.

## Exercises

1. Change `kron(B, A)` to `kron(A, B)` and run. The code still runs, but `max |dense -
   factored|` is no longer tiny. What does that tell you about which digit is fast?
2. Add a third digit: `x` of length 8·8·8, three matrices, and apply them one digit at a
   time. How many intermediates does the chain have now? Are they bigger than x?
3. Count by hand: for `np = Ni = nq = Nj = 32`, why is the factored cost 16× smaller than
   dense, and not 1024×?

## Checklist

- [ ] step 1 output matches exactly; I can map any `f` to `(i, j)` and back
- [ ] step 2 error is below `1e-10`
- [ ] I can say why `reshape` is free and what the project's einsum pays for instead
- [ ] I can say why the intermediate `T` is the same size as `x` here
- [ ] I can say why a knob of `kron(B, A)` is a product of one row-desk knob and one
      column-desk knob, and what the studio cannot do because of that
