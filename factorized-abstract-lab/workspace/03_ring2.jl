include("common.jl")
using LinearAlgebra, Random

Random.seed!(3)
const ni, nj = 4, 4
const np, nq = 4, 4
const R = 2

const A = randn(R, np, ni, R)
const B = randn(R, nq, nj, R)
const X = randn(ni, nj)

function dense_W(A, B)
    R, np, ni, _ = size(A)
    _, nq, nj, _ = size(B)
    W = zeros(np, nq, ni, nj)
    for j in 1:nj, i in 1:ni, q in 1:nq, p in 1:np
        W[p, q, i, j] = tr(A[:, p, i, :] * B[:, q, j, :])
    end
    return reshape(W, np * nq, ni * nj)
end

println("the ring as a dense matrix")
const W = dense_W(A, B)
const y_dense = W * vec(X)
@printf "  size(W) = %s,  W[1,1] = %.4f\n" string(size(W)) W[1, 1]
@printf "  check W[1,1] = tr(A[:,1,1,:]·B[:,1,1,:]) = %.4f\n" tr(A[:, 1, 1, :] * B[:, 1, 1, :])

function chain(A, B, X)
    R, np, ni, _ = size(A)
    _, nq, nj, _ = size(B)
    T = zeros(R, np, nj, R)
    for b in 1:R, j in 1:nj, p in 1:np, a in 1:R
        s = 0.0
        for i in 1:ni
            s += A[a, p, i, b] * X[i, j]
        end
        T[a, p, j, b] = s
    end
    Y = zeros(np, nq)
    for q in 1:nq, p in 1:np
        s = 0.0
        for a in 1:R, b in 1:R, j in 1:nj
            s += T[a, p, j, b] * B[b, q, j, a]
        end
        Y[p, q] = s
    end
    return vec(Y), T
end

println("\n the chain (reference style)")
y_chain, T = chain(A, B, X)
@printf "  max |dense - chain| = %.2e\n" maximum(abs.(y_dense .- y_chain))
@printf "  size(T) = %s  →  %d numbers  (X has %d)\n" string(size(T)) length(T) length(X)

println("\nhow the intermediate grows with R (np = nj = 4)")
println("      R   |T| = R²·np·nj   |T| / |X|")
for Rt in (1, 2, 4, 8, 16)
    @printf "  %5d   %13d   %8.0f×\n" Rt Rt^2 * np * nj Rt^2 * np * nj / (ni * nj)
end

function cut_ring(A, B, X)
    R, np, ni, _ = size(A)
    _, nq, nj, _ = size(B)
    Y = zeros(np, nq)
    Ta = zeros(np, nj, R)
    for a in 1:R
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
                s += Ta[p, j ,b] * B[b, q, j, a]
            end
            Y[p, q] += s
        end
    end
    return vec(Y), Ta
end

println("\ncut the ring")
y_cut, Ta = cut_ring(A, B, X)
@printf "  max |dense - cut_ring| = %.2e\n" maximum(abs.(y_dense .- y_cut))
@printf "  slice size(Ta) = %s → %d numbers  (full T had %d)\n" string(size(Ta)) length(Ta) length(T)

println("\nwith R = 1 the ring is the Kronecker product from rung 2")
A1 = randn(1, np, ni, 1)
B1 = randn(1, nq, nj, 1)
K = kron(B1[1, :, :, 1], A1[1, :, :, 1])
@printf "  max |dense_W(R=1) - kron| = %.2e\n" maximum(abs.(dense_W(A1, B1) .- K))
