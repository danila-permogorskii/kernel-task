include("common.jl")
using CUDA, LinearAlgebra

function spin!(cycles)
    t0 = clock(UInt64)
    while clock(UInt64) - t0 < cycles
    end
    return
end
const SPIN = UInt64(30_000_000) # about 20 ms at 1.5 GHz

function split_time(f; n = 200, reps = 3)
    f(); CUDA.synchronize() # compile and warm up
    host, gpu = Inf, Inf
    e1, e2 = CuEvent(), CuEvent()
    for _ in 1:reps
        @cuda spin!(SPIN)
        record(e1)
        th = @elapsed for _ in 1:n
            f()
        end
        record(e2)
        synchronize(e2)
        host = min(host, th / n)
        gpu = min(gpu, CUDA.elapsed(e1, e2) / n)
    end
    return host, gpu
end

println("the device and the parking kernel")
const dev = CUDA.device()
@printf "   %s: %d SMs, %.1f GiB\n" CUDA.name(dev) CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) CUDA.totalmem(dev)
@cuda spin!(UInt64(1)); CUDA.synchronize()
@printf "   one spin on the GPU clock: %s\n" us(CUDA.@elapsed @cuda spin!(SPIN))

function add1!(y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(y)
        @inbounds y[i] += 1f0
    end
    return
end
const y = CUDA.zeros(Float32, 1024)

println("\nthe price of one button press (almost no work)")
const H, g = split_time(() -> @cuda threads=256 blocks=4 add1!(y))
@printf "   hand-written kernel host %8s gpu %8s\n" us(H) us(g)
hb, gb = split_time(() -> (y .+= 1f0))
@printf "   broadcast y .+= 1   host %8s gpu %8s\n" us(hb) us(gb)

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

println("\nthe two roofs: bandwidth and matmul peak")
const src = CUDA.zeros(UInt8, 512 * 2^20) # 512 MiB
const dst = similar(src)
_, t_copy = split_time(() -> copyto!(dst, src); n = 5)
const BW = 2 * length(src) / t_copy # a copy reads and writes every byte
const BW_spec = 2 * CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) *
    1e3 * CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH) / 8
@printf "   bandwidth   %6.1f GB/s  spec %6.1f GB/s\n" BW / 1e9 BW_spec / 1e9

const nb = 4096
const Ab, Bb = CUDA.randn(Float16, nb, nb), CUDA.randn(Float16, nb, nb)
const Cb = similar(Ab)
_, t16 = split_time(() -> mul!(Cb, Ab, Bb); n = 5)
_, t32 = split_time(() -> mul32!(Cb, Ab, Bb); n = 5)
const F = 2nb^3 / t32
@printf "   FP16, FP16 accum.   %6.1f TFLOPS/s\n" 2nb^3 / t16 / 1e12
@printf "   FP16, FP32 accum. %6.1f TFLOPS/s <- the harness's setting\n" F / 1e12

const ni, nj, nk = 8, 12, 20
const np, nq, nr = 12, 10, 24
# (M, K, N) of the reference's three GEMMs, read off its einsum strings
gemm_shapes(R, t) = ((t * nj * nk,     ni,         R * np * R),   # "tijk,apib->tjkapb"
                     (t * nk * R * np, nj * R,     nq * R),       # "tjkapb,bqjc->tkapqc"
                     (t * np * nq,     nk * R * R, nr))           # "tkapqc,crka->tpqr"
const cases = ((8, 1), (8, 8), (8, 32), (16, 1), (16, 32))
const rows = Tuple{Float64, Float64, Float64}[]                   # (FLOPs, bytes, time)

println("\nthe reference's three GEMMs at the real sizes (GPU clock)")
println("   R   t  GEMM       M      K     N    measured   FLOP roof   byte roof")
for (R, t) in cases, (gi, (M, K, N)) in enumerate(gemm_shapes(R, t))
    A = CUDA.randn(Float16, M, K)
    B = CUDA.randn(Float16, K, N)
    C = CUDA.zeros(Float16, M, N)
    _, tg = split_time(() -> mul32!(C, A, B))
    fl = 2.0 * M * K * N
    by = 2.0 * (M * K + K * N + M * N)
    push!(rows, (fl, by, tg))
    @printf "  %2d  %2d   G%d  %6d  %5d  %4d   %9s   %9s   %9s\n" R t gi M K N us(tg) us(fl / F) us(by / BW)
end

const eff = maximum(fl / (tg * F) for (fl, by, tg) in rows)       # best fraction of F reached
kernel_time(fl, by) = max(fl / (eff * F), by / BW) + g            # guide 05's formula

println("\nthe measured machine, and guide 05's kernel formula on 15 GEMMs")
@printf "  Machine(BW = %.3g, F = %.3g, eff = %.2f, H = %.2g, g = %.2g)\n" BW F eff H g
println("   R   t  GEMM   measured   predicted   measured / predicted")
for (row, ((R, t), gi)) in zip(rows, ((c, gi) for c in cases for gi in 1:3))
    fl, by, tg = row
    tp = kernel_time(fl, by)
    @printf "  %2d  %2d   G%d   %9s   %9s   %6.2f\n" R t gi us(tg) us(tp) tg / tp
end
