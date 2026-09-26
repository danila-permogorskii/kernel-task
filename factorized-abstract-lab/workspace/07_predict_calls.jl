include("common.jl")
using CUDA, LinearAlgebra, Random

Base.@kwdef struct Machine
    BW; F; eff; H; g
end

const m = Machine(BW = 1.8e11, F = 1.98e13, eff = 0.67, H = 6.4e-6, g = 8.7e-6)

kernel_time(m, flops, bytes) = max(flops / (m.eff * m.F), bytes / m.BW) + m.g
function call_time(m, host_ops, kernels)
    gpu = sum(kernel_time(m, f, b) for (f, b) in kernels)
    return max(host_ops * m.H, gpu), host_ops * m.H, gpu
end

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
const K, N = ni * nj * nk, np * nq * nr
const e = 2

println("the machine from guide 06S")
@printf "  BW = %.0f GB/s   F = %.1f TFLOP/s   eff = %.2f   H = %s   g = %s\n" m.BW / 1e9 m.F / 1e12 m.eff us(m.H) us(m.g)
@printf "  a do-nothing kernel costs %s; one 11 MB read costs %s\n" us(kernel_time(m, 0, 0)) us(kernel_time(m, 0, e * N * K))

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
