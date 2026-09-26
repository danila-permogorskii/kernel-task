include("common.jl")
using LinearAlgebra, Random

Random.seed!(4)
const ni, nj, nk = 2, 3, 2 # input digits
const np, nq, nr = 2, 2, 3 # output digits
const R = 2
const nt = 3

const A = randn(R, np, ni, R)
const B = randn(R, nq, nj, R)
const C = randn(R, nr, nk, R)
const X = randn(ni, nj, nk, nt)

function dense_W(A, B, C)
    W = zeros(np, nq, nr, ni, nj, nk)
    for k in 1:nk, j in 1:nj, i in 1:ni, r in 1:nr, q in 1:nq, p in 1:np
        W[p, q, r, i, j, k] = tr(A[:, p, i, :] * B[:, q, j, :] * C[:, r, k, :])
    end
    return reshape(W, np * nq * nr, ni * nj * nk)
end

println("dense oracle")
const W = dense_W(A, B, C)
const Y_dense = W * reshape(X, ni * nj * nk, nt)
@printf "   size(W) = %s    size(Y) = %s\n" string(size(W)) string(size(Y_dense))

function chain(A, B, C, X)
    T1 = zeros(nj, nk, R, np, R, nt)
    for t in 1:nt, b in 1:R, p in 1:np, a in 1:R, k in 1:nk, j in 1:nj
        s = 0.0
        for i in 1:ni
            s += X[i, j, k, t] * A[a, p, i, b]
        end
        T1[j, k, a, p, b, t] = s
    end
    T2 = zeros(nk, R, np, nq, R, nt)
    for t in 1:nt, c in 1:R, q in 1:nq, p in 1:np, a in 1:R, k in 1:nk
        s = 0.0
        for b in 1:R, j in 1:nj
            s += T1[j, k, a, p, b, t] * B[b, q, j, c]
        end
        T2[k, a, p, q, c, t] = s
    end
    Y = zeros(np, nq, nr, nt)
    for t in 1:nt, r in 2:nr, q in 1:nq, p in 1:np
        s = 0.0
        for c in 1:R, a in 1:R, k in 1:nk
            s += T2[k, a, p, q, c, t] * C[c, r, k, a]
        end
        Y[p, q, r, t] = s
    end
    return reshape(Y, np * nq * nr, nt), T1, T2
end

println("\nchain (reference style)")
Y_chain, T1, T2 = chain(A, B, C, X)
@printf "  max |dense - chain| = %.2e\n" maximum(abs.(Y_dense .- Y_chain))
@printf "  |X| = %d   |T1| = %d   |T2| = %d   |Y| = %d\n" length(X) length(T1) length(T2) length(Y_chain)

function cut_ring(A, B, C, X)
    Y = zeros(np, nq, nr, nt)
    S1 = zeros(nj, nk, np, R)
    S2 = zeros(nk, np, nq, R)
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
            Y[p, q, r, t] += s
        end
    end
    return reshape(Y, np * nq * nr, nt), S1, S2
end

println("\ncut the ring")
Y_cut, S1, S2 = cut_ring(A, B, C, X)
@printf "  max |dense - cut_ring| = %.2e\n" maximum(abs.(Y_dense .- Y_cut))
@printf "  per token: T1 %d → S1 %d,   T2 %d → S2 %d   (÷R)\n" length(T1) ÷ nt length(S1) length(T2) ÷ nt length(S2)

function costs(ni, nj, nk, np, nq, nr, R)
    t1 = nj * nk * R * np * R
    t2 = nk * R * np * nq * R
    f1 = 2 * t1 * ni
    f2 = 2 * t2 * nj * R
    f3 = 2 * np * nq * nr * nk * R * R
    cores = R * R * (np * ni + nq * nj + nr * nk)
    dense = (np * nq * nr) * (ni * nj * nk)
    return (; t1, t2, flops = f1 + f2 + f3, f2, cores, dense)
end

println("\nthe project's numbers from the same formulas")
println("     R   cores     T1/token   T2/token   FLOPs/token   dense FLOPs/token")
for Rp in (8, 16)
    c = costs(8, 12, 20, 12, 10, 24, Rp)
    @printf "  %4d  %7d   %9d  %9d   %11d   %11d\n" Rp c.cores c.t1 c.t2 c.flops 2c.dense
end
