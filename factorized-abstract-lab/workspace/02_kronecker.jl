include("common.jl")
using LinearAlgebra, Random

println("a feature number is a 2-digit number")
const ni, nj = 3, 4
xs = collect(1.0:ni*nj)
Xs = reshape(xs, ni, nj)
display(Xs)

for f in (1, 2, 4, 12)
    i = (f - 1) % ni + 1
    j = (f - 1) ÷ ni + 1
     @printf "  f = %2d  →  (i=%d, j=%d)   x[f] = %4.1f   X[i,j] = %4.1f\n" f i j xs[f] Xs[i, j]
end


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

println("\ntwo small matmuls on the grid = one big matmul")
const M = dense_M(A, B)
@printf "  size(M) = %s\n" string(size(M))
@printf "  max |dense - factored| = %.2e\n" maximum(abs.(M * x .- factored(A, B, x)))

println("\ncost: predicted and measured")
fl_dense = 2 * (np * nq) * (Ni * Nj)
fl_fact  = 2 * np * Ni * Nj + 2 * np * Nj * nq
@printf "  dense   : %9d FLOPs  stored %9s  %9s\n" fl_dense human(sizeof(M)) us(best_time(() -> M * x))
@printf "  factored: %9d FLOPs  stored %9s  %9s\n" fl_fact human(sizeof(A) + sizeof(B)) us(best_time(() -> factored(A, B, x)))
@printf "  intermediate T = A*X holds %d numbers (x holds %d)\n" np * Nj Ni * Nj
