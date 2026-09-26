include("common.jl")

# The machine: every number here is an ASSUMPTION you will replace with a measurement.
Base.@kwdef struct Machine
    BW  = 3.35e12      # slow-memory bandwidth, bytes/s        (H100 SXM HBM3 spec)
    F   = 989e12       # peak FP16 tensor FLOP/s               (H100 SXM spec, dense)
    eff = 0.30         # fraction of peak a small kernel gets  (guess)
    H   = 5e-6         # host cost to dispatch one op, s       (guess)
    g   = 1e-6         # GPU-side gap per kernel, s            (guess)
end

kernel_time(m::Machine, flops, bytes) = max(flops / (m.eff * m.F), bytes / m.BW) + m.g

function call_time(m::Machine, host_ops, kernels)
    gpu = sum(kernel_time(m, f, b) for (f, b) in kernels)
    return max(host_ops * m.H, gpu), host_ops * m.H, gpu
end

m = Machine()
@printf "  read 11.06 MB, tiny math : %s\n" us(kernel_time(m, 1e3, 11.06e6))
@printf "  40 MFLOP, tiny memory    : %s\n" us(kernel_time(m, 40e6, 1e3))

const ni, nj, nk = 8, 12, 20
const np, nq, nr = 12, 10, 24
const K = ni * nj * nk
const N = np * nq * nr
const e = 2

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

function chain(m, R, t)
    s = sizes(R)
    ks = [
        (0.0, 2e * K * t), # copy X
        (0.0, 2e * s.A), # copy A
        (s.f1 * t, e * (K * t + s.A + s.t1 * t)), # GEMM 1 → T1
        (0.0, 2e * s.t1 * t), # copy T1
        (0.0, 2e * s.B), # copy B
        (s.f2 * t, e * (s.t1 * t + s.B + s.t2 * t)), # GEMM 2 → T2
        (0.0, 2e * s.t2 * t), # copy T2
        (0.0, 2e * s.C), # copy C
        (s.f3 * t, e * (s.t2 * t + s.C + N * t)), # GEMM 3 → y
        ]
    return call_time(m, 9, ks)
end

println("\ndense vs reference chain, R = 8, t = 1")
for (name, f) in (("dense", dense), ("chain", chain))
    total, host, gpu = f(m, 8, 1)
    @printf "  %-6s total %8s   (host %8s, gpu %8s)\n" name us(total) us(host) us(gpu)
end

function fused_1k(m, R, t)
    s = sizes(R)
    flops = (s.f1 + s.f2 + s.f3) * t
    bytes = e * (K * t + s.A + s.B + s.C + N * t)
    return call_time(m, 1, [(flops, bytes)])
end

function fused_2k(m, R, t)
    s = sizes(R)
    flops = (s.f1 + s.f2 + s.f3) * t
    k1 = (flops, e * (K * t + s.A + s.B + s.C) + 4 * R * N * t)
    k2 = (1.0 * R * N * t, 4 * R * N * t + e * N * t)
    return call_time(m, 2, [k1, k2])
end

const methods = (("dense", dense), ("chain", chain), ("fused1", fused_1k), ("fused2", fused_2k))
const cases = ((8, 1), (8, 8), (8, 32), (16, 1), (16, 32))

function table(m)
    println("   R   t |    dense     chain    fused1    fused2 | fused1 vs dense")
    for (R, t) in cases
        ts = [f(m, R, t)[1] for (_, f) in methods]
        @printf "  %2d  %2d | %8s  %8s  %8s  %8s | %5.2f×\n" R t us(ts[1]) us(ts[2]) us(ts[3]) us(ts[4]) ts[1] / ts[3]
    end
end

println("\npredicted table, default machine  (>1× means fused1 is faster)")
table(m)

println("\nno host overhead (H = 0, like CUDA-graph replay)")
table(Machine(H = 0.0))
println("\na worse kernel (eff = 0.10)")
table(Machine(eff = 0.10))
