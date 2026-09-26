include("common.jl")
using LinearAlgebra, Random

Random.seed!(1)
const n = 2000
const r = 64
const U = randn(n, r) # tall and thing
const V = randn(r, n) # short and wide
const x = randn(n, 32)

bad(U, V, x) = (U * V) * x # cooks the whole nxn dish first
good(U,V, x) = U * (V * x) # never builds the nxn matrix

println("Do both orders give the same y?")
@printf "   max | bad - good | = %.2e\n" maximum(abs.(bad(U, V, x) .- good(U, V, x)))

flops_bad(n, r) = 2n * n * r + 2n * n # build M, then M*x
flops_good(n, r) = 2r * n + 2n * r # V*x, then U*(V*x)

println("predicted cost (count by formula)")
@printf "  bad : %12d FLOPs, intermediate M   = %s\n" flops_bad(n, r)  human(8n * n)
@printf "  good: %12d FLOPs, intermediate V*x = %s\n" flops_good(n, r) human(8r)
@printf "  ratio: %.0f×\n" flops_bad(n, r) / flops_good(n, r)

println("\nSTEP 3 — measured cost")
tb = best_time(() -> bad(U, V, x))
tg = best_time(() -> good(U, V, x))
ab = alloc_bytes(() -> bad(U, V, x))
ag = alloc_bytes(() -> good(U, V, x))
@printf "  bad : %10s   allocated %s\n" us(tb) human(ab)
@printf "  good: %10s   allocated %s\n" us(tg) human(ag)

const M = U * V # the "dense baseline": build once, stored
dense(M, x) = M * x

println("\ndense (prebuilt M) vs factored")
@printf "  stored: dense M = %s   factors U,V = %s\n" human(sizeof(M)) human(sizeof(U) + sizeof(V))
@printf "  dense : %10s\n" us(best_time(() -> dense(M, x)))
@printf "  good  : %10s\n" us(best_time(() -> good(U, V, x)))
