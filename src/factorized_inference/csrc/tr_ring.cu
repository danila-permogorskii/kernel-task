// Fused tensor-ring forward (v2):  y[t,(p,q,r)] = sum_{a,k} piece(a,k)[t,p,q,r]
//
// One thread block owns one ring link `a`, a chunk of `kc` input digits `k`, a tile of `tt`
// tokens and a chunk of `qc` output digits `q` (stage 3 treats every q separately, so q can
// be split across blocks; each block then needs only its slice of B). It loads its B slice,
// A[a] and C[:,:,k,a] into shared memory once, then loops over its k values; every piece
// stays on-chip:
//
//   for each k in the chunk:
//     stage 1 (CUDA cores)   S1[(t,p),(j,b)]   = sum_i  x[t,i,j,k] * A[a,p,i,b]   shared mem
//     stage 2 (Tensor Cores) S2[(t,p),(q,c)]   = sum_jb S1 * B2[(j,b),(q,c)]      shared mem
//                            (only the block's q chunk)
//     stage 3 (Tensor Cores) Y[(t,p,q), r]    += sum_c  S2[(t,p,q),c] * C[c,r,k,a] registers
//
// After the loop, Y is added into an FP32 workspace with atomicAdd (once per block).
//   design A: a separate kernel converts the workspace to FP16          (3 launches per call)
//   design B: the last block of each token tile converts and clears it  (1 launch per call)
//
// The two real workloads (R = 8, 16 on modes (8,12,20) -> (12,10,24)) are compiled as
// fixed-shape variants: sizes are compile-time constants, so index arithmetic becomes shifts
// and multiplies and loops unroll. Every other shape runs the generic variant.
//
// History: kernel_work/tr_ring_v0.cu (v0). Walkthrough: kernel-guides/G2-kernel-walkthrough.md

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#ifndef TR_THREADS
#define TR_THREADS 256  // experiments build other sizes: -DTR_THREADS=512
#endif
constexpr int kThreads = TR_THREADS;  // 8 warps per block by default
constexpr int kWarps = kThreads / 32;
constexpr int kMaxYTiles = 8;         // Y accumulator tiles held in registers per warp
constexpr int kStageLd = 20;          // per-warp FP32 staging tile: 16 rows x 20 floats

__host__ __device__ constexpr int round16(int v) { return (v + 15) / 16 * 16; }
__host__ __device__ inline size_t align128(size_t v) { return (v + 127) & ~size_t(127); }

struct Dims {
  int T, ni, nj, nk, P, Q, Rr, R;  // problem: tokens, input modes, output modes, ring rank
  int kc, tt, qc;                  // tiling: k values, tokens and q values per block
  int Rc, Rrp;                     // R and Rr padded to 16 (Tensor Core tile edge)
  int K2p, N2c, Mp;                // stage 2: K = nj*R -> K2p, N = Q*Rc (whole B2), M = tt*P -> Mp
  int Nb;                          // stage 2 N for one block: qc*Rc
  // shared-memory row strides (leading dimensions), padded so that consecutive rows do not
  // start in the same memory bank: a stride that is a multiple of 128 bytes makes all 16
  // rows of a Tensor Core tile hit the same bank and be served one after another
  int K2s, N2s, Rrs;               // S1: K2p+8 halves, B slice: Nb+8, C: Rrp+8
};

// Compile-time shape. 0 = "not fixed, read it from Dims at runtime".
template <int NI, int NJ, int P_, int Q_, int RR, int R_>
struct Shape {
  static constexpr int ni = NI, nj = NJ, P = P_, Q = Q_, Rr = RR, R = R_;
};
using Generic = Shape<0, 0, 0, 0, 0, 0>;
using RealR8  = Shape<8, 12, 12, 10, 24, 8>;
using RealR16 = Shape<8, 12, 12, 10, 24, 16>;

// Shared-memory map of one block. Host and device use the same function.
struct SmemLayout { size_t b, a, c, x, s1, s2, stage, total; };

__host__ __device__ inline SmemLayout smem_layout(const Dims& d) {
  SmemLayout s;
  size_t o = 0;
  s.b  = o; o = align128(o + sizeof(__half) * d.K2p * d.N2s);             // B2: the block's q slice
  s.a  = o; o = align128(o + sizeof(float)  * d.ni * d.P * d.R);          // A[a]
  s.c  = o; o = align128(o + sizeof(__half) * d.kc * d.Rc * d.Rrs);       // C[:,:,k,a], kc of them
  s.x  = o; o = align128(o + sizeof(float)  * d.kc * d.tt * d.ni * d.nj); // x slice
  s.s1 = o; o = align128(o + sizeof(__half) * d.Mp * d.K2s);              // S1, one k at a time
  s.s2 = o; o = align128(o + sizeof(__half) * d.Mp * d.Nb);               // S2, one k at a time
  s.stage = o; o = align128(o + sizeof(float) * kWarps * 16 * kStageLd);  // per-warp 16x16 FP32
  s.total = o;
  return s;
}

using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

template <bool kLastBlockFinishes, class S>
__global__ void __launch_bounds__(kThreads)
tr_ring_fused_kernel(const __half* __restrict__ x,    // [T, ni*nj*nk]
                     const __half* __restrict__ A1,   // [R(a)][ni][P*R (p,b)]
                     const __half* __restrict__ B2,   // [K2p (j,b)][N2c (q,c)], zero padded
                     const __half* __restrict__ C3,   // [nk][R(a)][Rc (c)][Rrp (r)], zero padded
                     float* __restrict__ ws,          // [T, P*Q*Rr] FP32 accumulator
                     __half* __restrict__ y,          // [T, P*Q*Rr]   (design B only)
                     unsigned int* __restrict__ tile_done,  // [token tiles] (design B only)
                     const Dims d) {
  // sizes: compile-time constants for the fixed variants, runtime values for Generic
  const int ni = S::ni ? S::ni : d.ni, nj = S::nj ? S::nj : d.nj;
  const int P = S::P ? S::P : d.P, Q = S::Q ? S::Q : d.Q;
  const int Rr = S::Rr ? S::Rr : d.Rr, R = S::R ? S::R : d.R;
  const int Rc = S::R ? round16(S::R) : d.Rc, Rrp = S::Rr ? round16(S::Rr) : d.Rrp;
  const int K2p = (S::nj && S::R) ? round16(S::nj * S::R) : d.K2p;
  const int N2c = Q * Rc, K2s = K2p + 8, Rrs = Rrp + 8;
  const int nk = d.nk, Mp = d.Mp, tt = d.tt, kc = d.kc, qc = d.qc;
  const int Nb = qc * Rc, N2s = Nb + 8;

  extern __shared__ __align__(128) unsigned char smem[];
  const SmemLayout L = smem_layout(d);
  __half* sB  = reinterpret_cast<__half*>(smem + L.b);
  float*  sA  = reinterpret_cast<float*>(smem + L.a);
  __half* sC  = reinterpret_cast<__half*>(smem + L.c);
  float*  sX  = reinterpret_cast<float*>(smem + L.x);
  __half* sS1 = reinterpret_cast<__half*>(smem + L.s1);
  __half* sS2 = reinterpret_cast<__half*>(smem + L.s2);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  float* stage = reinterpret_cast<float*>(smem + L.stage) + warp * 16 * kStageLd;

  const int a = blockIdx.y;
  const int nkc = (nk + kc - 1) / kc;
  const int k0 = (blockIdx.x % nkc) * kc, q0 = (blockIdx.x / nkc) * qc, t0 = blockIdx.z * tt;
  const int qc_valid = min(qc, Q - q0);
  const int kc_valid = min(kc, nk - k0);
  const int tt_valid = min(tt, d.T - t0);
  const int in_features = ni * nj * nk;
  const int out_features = P * Q * Rr;
  const int PR = P * R;

  // ---- load: global -> shared, once per block --------------------------------------------
  {  // B2 and C: 16-byte vector copies (8 halves), all threads at once, into padded rows
    const int4* src = reinterpret_cast<const int4*>(B2);
    int4* dst = reinterpret_cast<int4*>(sB);
    // columns [q0*Rc, (q0+qc)*Rc) of B2; zero past the last q
    const int grow = N2c / 8, brow = Nb / 8, bstride = N2s / 8;  // int4 per row
    const int valid8 = qc_valid * Rc / 8, col0 = q0 * Rc / 8;
    const int4 zero = make_int4(0, 0, 0, 0);
    for (int e = tid; e < K2p * brow; e += kThreads) {
      const int row = e / brow, col = e % brow;
      dst[row * bstride + col] = col < valid8 ? src[row * grow + col0 + col] : zero;
    }
    const int crow = Rrp / 8, cstride = Rrs / 8;
    const int cvec = Rc * crow;  // one C[:,:,k,a] slab in global memory, in int4 units
    for (int e = tid; e < kc * cvec; e += kThreads) {
      const int kk = e / cvec, c = (e % cvec) / crow, col = e % crow;
      const int4 zero = make_int4(0, 0, 0, 0);
      reinterpret_cast<int4*>(sC)[(kk * Rc + c) * cstride + col] = kk < kc_valid
          ? reinterpret_cast<const int4*>(C3)[((size_t)(k0 + kk) * R + a) * cvec + e % cvec]
          : zero;
    }
  }
  const __half* A1a = A1 + (size_t)a * ni * PR;  // sA[i][p*R+b] = A[a,p,i,b]
  for (int e = tid; e < ni * PR; e += kThreads) sA[e] = __half2float(A1a[e]);
  // sX[kk][t][i][j] = x[t0+t, i, j, k0+kk], zero outside the valid k / token range
  for (int e = tid; e < kc * tt * ni * nj; e += kThreads) {
    const int j = e % nj;
    int rest = e / nj;
    const int i = rest % ni;  rest /= ni;
    const int t = rest % tt;
    const int kk = rest / tt;
    float v = 0.f;
    if (kk < kc_valid && t < tt_valid)
      v = __half2float(x[(size_t)(t0 + t) * in_features + (i * nj + j) * nk + (k0 + kk)]);
    sX[e] = v;
  }
  // S1's padding (rows >= tt*P, columns >= nj*R) must be zero; stage 1 never writes it
  for (int e = tid; e < Mp * K2s; e += kThreads) sS1[e] = __float2half(0.f);

  // Y accumulators live in registers for the whole k loop: warp w owns tiles w, w+8, ...
  const int y_nt_n = Rrp / 16, y_tiles = (Mp * qc / 16) * y_nt_n;
  FragC yacc[kMaxYTiles];
#pragma unroll
  for (int i = 0; i < kMaxYTiles; ++i) wmma::fill_fragment(yacc[i], 0.f);
  __syncthreads();

  const int M = tt * P;
  const bool vec4 = (nj % 4 == 0) && (R % 4 == 0);  // true for the real workloads

  for (int kk = 0; kk < kc_valid; ++kk) {
    // ---- stage 1: S1[(t,p)][j*R+b] = sum_i x[t,i,j] * A[i][p*R+b] ------------------------
    // Written directly in the layout stage 2 reads (rows (t,p), columns (j,b)): no permute.
    const float* xk = sX + (size_t)kk * tt * ni * nj;
    if (vec4) {
      // register tile: one thread = 4 j x 4 b for one (t,p); per i: 2 float4 loads, 16 FMAs
      const int J4 = nj / 4, B4 = R / 4;
      for (int e = tid; e < M * J4 * B4; e += kThreads) {
        const int b4 = e % B4, j4 = (e / B4) % J4, row = e / (B4 * J4);
        const int t = row / P, p = row % P;
        float acc[4][4] = {};
#pragma unroll 8
        for (int i = 0; i < ni; ++i) {
          const float4 xv = *reinterpret_cast<const float4*>(xk + (t * ni + i) * nj + j4 * 4);
          const float4 av = *reinterpret_cast<const float4*>(sA + i * PR + p * R + b4 * 4);
          const float xs[4] = {xv.x, xv.y, xv.z, xv.w}, as[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
          for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[u][v] += xs[u] * as[v];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          __half2* dst = reinterpret_cast<__half2*>(sS1 + row * K2s + (j4 * 4 + u) * R + b4 * 4);
          dst[0] = __floats2half2_rn(acc[u][0], acc[u][1]);
          dst[1] = __floats2half2_rn(acc[u][2], acc[u][3]);
        }
      }
    } else {  // any shape: one thread = one S1 value
      for (int e = tid; e < M * nj * R; e += kThreads) {
        const int row = e / (nj * R), col = e % (nj * R);
        const int t = row / P, p = row % P, j = col / R, b = col % R;
        float acc = 0.f;
        for (int i = 0; i < ni; ++i)
          acc += xk[(t * ni + i) * nj + j] * sA[i * PR + p * R + b];
        sS1[row * K2s + col] = __float2half(acc);
      }
    }
    __syncthreads();

    // ---- stage 2: S2 = S1 @ B2 on Tensor Cores (FP16 in, FP32 accumulate, FP16 out) --------
    {
      const int mt_n = Mp / 16, nt_n = Nb / 16, kt_n = K2p / 16;
      for (int tile = warp; tile < mt_n * nt_n; tile += kWarps) {
        const int mt = tile % mt_n, nt = tile / mt_n;
        FragC acc;
        wmma::fill_fragment(acc, 0.f);
        for (int kt = 0; kt < kt_n; ++kt) {
          FragA fa;
          FragB fb;
          wmma::load_matrix_sync(fa, sS1 + mt * 16 * K2s + kt * 16, K2s);
          wmma::load_matrix_sync(fb, sB + kt * 16 * N2s + nt * 16, N2s);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        // FP32 tile -> per-warp staging -> FP16 into S2 (stage 3 needs FP16 operands)
        wmma::store_matrix_sync(stage, acc, kStageLd, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32)
          sS2[(mt * 16 + e / 16) * Nb + nt * 16 + e % 16] =
              __float2half(stage[(e / 16) * kStageLd + e % 16]);
        __syncwarp();
      }
    }
    __syncthreads();

    // ---- stage 3: Y[(t,p,q), r] += S2[(t,p,q), c] @ C[c, r]  on Tensor Cores ---------------
    // S2 rows (t,p) hold qc blocks of Rc values, so the same bytes read as a matrix
    // [(t,p,q) x c] with row stride Rc: the second "free re-layout".
    {
      const __half* ck = sC + kk * Rc * Rrs;
#pragma unroll
      for (int i = 0; i < kMaxYTiles; ++i) {
        const int tile = warp + i * kWarps;
        if (tile < y_tiles) {
          const int mt = tile / y_nt_n, nt = tile % y_nt_n;
          for (int kt = 0; kt < Rc / 16; ++kt) {
            FragA fa;
            FragB fb;
            wmma::load_matrix_sync(fa, sS2 + mt * 16 * Rc + kt * 16, Rc);
            wmma::load_matrix_sync(fb, ck + kt * 16 * Rrs + nt * 16, Rrs);
            wmma::mma_sync(yacc[i], fa, fb, yacc[i]);
          }
        }
      }
    }
    __syncthreads();
  }

  // ---- add this block's Y into the FP32 workspace (one atomic per output per block) --------
  // each warp stages its Y tiles through shared memory; a row of a tile = consecutive r
#pragma unroll
  for (int i = 0; i < kMaxYTiles; ++i) {
    const int tile = warp + i * kWarps;
    if (tile < y_tiles) {
      const int mt = tile / y_nt_n, nt = tile % y_nt_n;
      wmma::store_matrix_sync(stage, yacc[i], kStageLd, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        const int m = mt * 16 + e / 16, r = nt * 16 + e % 16;  // m = (t*P + p)*qc + ql
        const int ql = m % qc, tp = m / qc, t = tp / P, p = tp % P;
        if (r < Rr && t < tt_valid && ql < qc_valid)
          atomicAdd(ws + (size_t)(t0 + t) * out_features + (p * Q + q0 + ql) * Rr + r,
                    stage[(e / 16) * kStageLd + e % 16]);
      }
      __syncwarp();
    }
  }

  // ---- design B: the last block of this token tile converts FP32 -> FP16 and clears -------
  if constexpr (kLastBlockFinishes) {
    __shared__ bool is_last;
    __threadfence();   // this thread's atomics are visible device-wide before we signal
    __syncthreads();   // ... for every thread of the block
    if (tid == 0) {
      const unsigned int blocks_per_tile = gridDim.x * gridDim.y;
      is_last = atomicAdd(tile_done + blockIdx.z, 1u) == blocks_per_tile - 1;
    }
    __syncthreads();
    if (is_last) {
      __threadfence();
      const size_t base = (size_t)t0 * out_features;
      const int n = tt_valid * out_features;
      // kInFlight loads issued before any store: one L2 round trip per batch, not per value
      // (any out_features, so scalar loads: float4 would need a multiple of 4)
      constexpr int kInFlight = 8;
      for (int e0 = tid; e0 < n; e0 += kInFlight * kThreads) {
        float v[kInFlight];
#pragma unroll
        for (int s = 0; s < kInFlight; ++s) {
          const int e = e0 + s * kThreads;
          if (e < n) v[s] = __ldcg(ws + base + e);  // read from L2, not a stale L1 line
        }
#pragma unroll
        for (int s = 0; s < kInFlight; ++s) {
          const int e = e0 + s * kThreads;
          if (e < n) {
            y[base + e] = __float2half(v[s]);
            ws[base + e] = 0.f;                     // read-and-clear: ready for the next call
          }
        }
      }
      if (tid == 0) tile_done[blockIdx.z] = 0;
    }
  }
}

// Design A, launch 3: FP32 workspace -> FP16 output.
__global__ void tr_ring_convert_kernel(const float* __restrict__ ws, __half* __restrict__ y,
                                       size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2half(ws[i]);
}

static Dims make_dims(int64_t T, const std::vector<int64_t>& m, int64_t kc, int64_t tt,
                      int64_t qc) {
  TORCH_CHECK(m.size() == 7, "modes = [ni, nj, nk, P, Q, Rr, R]");
  Dims d;
  d.T = (int)T;
  d.ni = (int)m[0]; d.nj = (int)m[1]; d.nk = (int)m[2];
  d.P = (int)m[3];  d.Q = (int)m[4];  d.Rr = (int)m[5]; d.R = (int)m[6];
  d.kc = (int)kc; d.tt = (int)tt; d.qc = (int)qc;
  d.Rc = round16(d.R); d.Rrp = round16(d.Rr);
  d.K2p = round16(d.nj * d.R); d.N2c = d.Q * d.Rc; d.Mp = round16(d.tt * d.P);
  d.Nb = d.qc * d.Rc;
  d.K2s = d.K2p + 8; d.N2s = d.Nb + 8; d.Rrs = d.Rrp + 8;
  return d;
}

int64_t smem_bytes(std::vector<int64_t> modes, int64_t kc, int64_t tt, int64_t qc) {
  return (int64_t)smem_layout(make_dims(1, modes, kc, tt, qc)).total;
}

int64_t y_tiles(std::vector<int64_t> modes, int64_t tt, int64_t qc) {  // <= max_y_tiles()
  const Dims d = make_dims(1, modes, 1, tt, qc);
  return (int64_t)(d.Mp * d.qc / 16) * (d.Rrp / 16);
}

int64_t max_y_tiles() { return (int64_t)kMaxYTiles * kWarps; }

template <bool kB, class S>
static void launch(const Dims& d, dim3 grid, size_t smem, cudaStream_t stream,
                   const __half* x, const __half* a1, const __half* b2, const __half* c3,
                   float* ws, __half* y, unsigned int* tile_done) {
  static size_t current = 0;  // opt-in limit set so far, one per kernel variant (default 48 KB
                              // is not enough once static smem is added: always opt in)
  if (smem > current) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tr_ring_fused_kernel<kB, S>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    current = smem;
  }
  tr_ring_fused_kernel<kB, S><<<grid, kThreads, smem, stream>>>(x, a1, b2, c3, ws, y,
                                                                tile_done, d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kB>
static void dispatch(const Dims& d, dim3 grid, size_t smem, cudaStream_t stream,
                     const __half* x, const __half* a1, const __half* b2, const __half* c3,
                     float* ws, __half* y, unsigned int* tile_done) {
  const bool real = d.ni == 8 && d.nj == 12 && d.P == 12 && d.Q == 10 && d.Rr == 24;
  if (real && d.R == 8)
    launch<kB, RealR8>(d, grid, smem, stream, x, a1, b2, c3, ws, y, tile_done);
  else if (real && d.R == 16)
    launch<kB, RealR16>(d, grid, smem, stream, x, a1, b2, c3, ws, y, tile_done);
  else
    launch<kB, Generic>(d, grid, smem, stream, x, a1, b2, c3, ws, y, tile_done);
}

// ws_b / tile_done_b: persistent buffers for design B (all zero between calls); ignored for A.
torch::Tensor tr_ring_forward(torch::Tensor x, torch::Tensor A1, torch::Tensor B2,
                              torch::Tensor C3, std::vector<int64_t> modes, int64_t kc,
                              int64_t tt, int64_t qc, bool last_block_finishes, torch::Tensor ws_b,
                              torch::Tensor tile_done_b) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf && x.dim() == 2 && x.is_contiguous(),
              "x must be a contiguous 2D CUDA float16 tensor");
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const Dims d = make_dims(x.size(0), modes, kc, tt, qc);
  const int64_t out_features = (int64_t)d.P * d.Q * d.Rr;
  TORCH_CHECK(x.size(1) == (int64_t)d.ni * d.nj * d.nk, "x has the wrong number of features");
  TORCH_CHECK(B2.size(0) == d.K2p && B2.size(1) == d.N2c, "B2 packing does not match");
  TORCH_CHECK(y_tiles(modes, tt, qc) <= max_y_tiles(), "token tile too large for register Y");
  auto y = torch::empty({x.size(0), out_features}, x.options());
  if (d.T == 0) return y;

  const size_t smem = smem_layout(d).total;
  const dim3 grid(((d.nk + d.kc - 1) / d.kc) * ((d.Q + d.qc - 1) / d.qc), d.R,
                  (d.T + d.tt - 1) / d.tt);
  auto xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  auto a1 = reinterpret_cast<const __half*>(A1.data_ptr<at::Half>());
  auto b2 = reinterpret_cast<const __half*>(B2.data_ptr<at::Half>());
  auto c3 = reinterpret_cast<const __half*>(C3.data_ptr<at::Half>());
  auto yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

  if (!last_block_finishes) {  // ---------------------------- design A: 3 launches
    auto ws = torch::zeros({x.size(0), out_features}, x.options().dtype(torch::kFloat));  // 1
    dispatch<false>(d, grid, smem, stream, xp, a1, b2, c3, ws.data_ptr<float>(),          // 2
                    nullptr, nullptr);
    const size_t n = (size_t)d.T * out_features;
    tr_ring_convert_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(              // 3
        ws.data_ptr<float>(), yp, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {  // ------------------------------------------------ design B: 1 launch
    TORCH_CHECK(ws_b.numel() >= (int64_t)d.T * out_features && tile_done_b.numel() >= grid.z,
                "design B workspace too small");
    dispatch<true>(d, grid, smem, stream, xp, a1, b2, c3, ws_b.data_ptr<float>(), yp,
                   reinterpret_cast<unsigned int*>(tile_done_b.data_ptr<int32_t>()));
  }
  return y;
}

// =============================================================================================
// V3 (t = 1, the two real workloads): stages 2 -> 3 in registers on PTX mma.sync.
// Developed and measured in kernel_work/stack/tr_stack.cu (kernel-design/WEIGHT_STATIONARY.md
// §8): per layer 9.3 -> 7.35 µs against the WMMA path at R = 8. One block = one unit (link a,
// k chunk KC, q chunk QC), all sizes compile-time:
//   cores slice   cp.async (16-byte async global -> shared, zero-fill for padding), A in FP16
//   x slice       FP16 -> FP32 in shared memory
//   stage 1       S1[k][(p)][(j,b)] for every k of the unit, then ONE __syncthreads
//   stage 2       warp w owns q = q0 + w: mma.m16n8k16 over (j,b) -> accumulator 16 x 8c
//   stage 3       the accumulator IS the A operand of the next mma (m16n8 accumulator layout =
//                 m16n8k8 A layout; two tiles = m16n8k16 A for R = 16): packed to FP16 in
//                 registers, mma over c into Y (3 tiles of 8 r), Y stays in registers
//   output        red.global.add.v2.f32 straight from the Y fragment into the FP32 workspace
// No S2 buffer, no FP32 staging tile, no barrier between stages 2 and 3.
// =============================================================================================
namespace v3 {
constexpr int ni = 8, nj = 12, nk = 20, P = 12, Q = 10, Rr = 24;  // the real modes
__host__ __device__ constexpr size_t a128(size_t v) { return (v + 127) & ~size_t(127); }

template <int R, int KC, int QC>
struct Cfg {
  static constexpr int Rc = round16(R), Rrp = round16(Rr), Rrs = Rrp + 8;
  static constexpr int K2p = round16(nj * R), K2s = K2p + 8, N2c = Q * Rc;
  static constexpr int Nb = QC * Rc, N2s = Nb + 8, PR = P * R;
  static constexpr int nkc = (nk + KC - 1) / KC, nqc = (Q + QC - 1) / QC;
  static constexpr int units = R * nkc * nqc;
  static constexpr int kt_n = K2p / 16, NC8 = R / 8, RT = (Rr + 7) / 8;
  static constexpr size_t b_off = 0;
  static constexpr size_t a_off = a128(b_off + sizeof(__half) * K2p * N2s);
  static constexpr size_t c_off = a128(a_off + sizeof(__half) * ni * PR);
  static constexpr size_t x_off = a128(c_off + sizeof(__half) * KC * Rc * Rrs);
  static constexpr size_t s1_off = a128(x_off + sizeof(float) * KC * ni * nj);
  static constexpr size_t smem = a128(s1_off + sizeof(__half) * KC * 16 * K2s);
  static_assert(R == 8 || R == 16, "V3: R = 8 or 16");
  static_assert(QC <= kWarps, "V3: one warp per q of the chunk");
};

__device__ __forceinline__ void mma16816(float (&d)[4], unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma1688(float (&d)[4], unsigned a0, unsigned a1, unsigned b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ unsigned pack_f2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&h);
}
__device__ __forceinline__ unsigned pack_h2(__half lo, __half hi) {
  __half2 h = __halves2half2(lo, hi);
  return *reinterpret_cast<unsigned*>(&h);
}
__device__ __forceinline__ unsigned lds_u32(const __half* p) {
  return *reinterpret_cast<const unsigned*>(p);
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(
                   (unsigned)__cvta_generic_to_shared(dst)),
               "l"(src), "r"(src_bytes)
               : "memory");
}
__device__ __forceinline__ void red_add_v2(float* p, float a, float b) {
#if __CUDA_ARCH__ >= 900
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(p), "f"(a), "f"(b) : "memory");
#else
  atomicAdd(p, a);
  atomicAdd(p + 1, b);
#endif
}

template <bool kLastBlockFinishes, int kTail, int R, int KC, int QC>
__global__ void __launch_bounds__(kThreads)
tr_ring_fused_v3_kernel(const __half* __restrict__ x,    // [1, ni*nj*nk]
                        const __half* __restrict__ A1,   // [R(a)][ni][P*R (p,b)]
                        const __half* __restrict__ B2,   // [K2p (j,b)][N2c (q,c)]
                        const __half* __restrict__ C3,   // [nk][R(a)][Rc (c)][Rrp (r)]
                        float* __restrict__ ws,          // [P*Q*Rr] FP32 accumulator
                        __half* __restrict__ y,          // [P*Q*Rr]   (design B only)
                        unsigned int* __restrict__ tile_done) {
  using C = Cfg<R, KC, QC>;
  constexpr int Rc = C::Rc, Rrs = C::Rrs, K2s = C::K2s, N2s = C::N2s, PR = C::PR;
  constexpr int kt_n = C::kt_n, RT = C::RT, out_f = P * Q * Rr;
  extern __shared__ __align__(128) unsigned char smem[];
  __half* sB = reinterpret_cast<__half*>(smem + C::b_off);
  __half* sA = reinterpret_cast<__half*>(smem + C::a_off);
  __half* sC = reinterpret_cast<__half*>(smem + C::c_off);
  float* sX = reinterpret_cast<float*>(smem + C::x_off);
  __half* sS1 = reinterpret_cast<__half*>(smem + C::s1_off);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane >> 2, t4 = lane & 3;
  const int u = blockIdx.x, per_a = C::nkc * C::nqc;
  const int a = u / per_a, rem = u % per_a;
  const int k0 = (rem % C::nkc) * KC, q0 = (rem / C::nkc) * QC;
  const int qc_valid = min(QC, Q - q0), kc_valid = min(KC, nk - k0);

  {  // ---- cores slice: cp.async into shared memory, zero-fill for padding
    constexpr int brow = C::Nb / 8, bstride = N2s / 8, grow = C::N2c / 8;
    const int valid8 = qc_valid * Rc / 8, col0 = q0 * Rc / 8;
    const int4* gB = reinterpret_cast<const int4*>(B2);
    for (int e = tid; e < C::K2p * brow; e += kThreads) {
      const int row = e / brow, col = e % brow;
      const bool ok = col < valid8;
      cp_async16(reinterpret_cast<int4*>(sB) + row * bstride + col,
                 gB + row * grow + col0 + (ok ? col : 0), ok ? 16 : 0);
    }
    constexpr int crow = C::Rrp / 8, cstride = Rrs / 8, cvec = Rc * crow;
    const int4* gC = reinterpret_cast<const int4*>(C3);
    for (int e = tid; e < KC * cvec; e += kThreads) {
      const int kk = e / cvec, c = (e % cvec) / crow, col = e % crow;
      const bool ok = kk < kc_valid;
      cp_async16(reinterpret_cast<int4*>(sC) + (kk * Rc + c) * cstride + col,
                 gC + ((size_t)(k0 + (ok ? kk : 0)) * R + a) * cvec + e % cvec, ok ? 16 : 0);
    }
    const int4* gA = reinterpret_cast<const int4*>(A1 + (size_t)a * ni * PR);
    for (int e = tid; e < ni * PR / 8; e += kThreads)
      cp_async16(reinterpret_cast<int4*>(sA) + e, gA + e, 16);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  // ---- x slice: sX[kk][i][j] = x[i, j, k0 + kk] (FP32), while the copies are in flight
  for (int e = tid; e < KC * ni * nj; e += kThreads) {
    const int jj = e % nj, i = (e / nj) % ni, kk = e / (ni * nj);
    sX[e] = kk < kc_valid ? __half2float(x[(i * nj + jj) * nk + k0 + kk]) : 0.f;
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();

  // ---- stage 1, every k at once: S1[kk][p][j*R+b] = sum_i x[i,j,kk] * A[i][p*R+b]
  {
    constexpr int J4 = nj / 4, B4 = R / 4, tasks = P * J4 * B4;
    for (int e = tid; e < kc_valid * tasks; e += kThreads) {
      const int kk = e / tasks, f = e % tasks;
      const int b4 = f % B4, j4 = (f / B4) % J4, p = f / (B4 * J4);
      const float* xk = sX + kk * ni * nj;
      float acc[4][4] = {};
#pragma unroll
      for (int i = 0; i < ni; ++i) {
        const float4 xv = *reinterpret_cast<const float4*>(xk + i * nj + j4 * 4);
        const uint2 au = *reinterpret_cast<const uint2*>(sA + i * PR + p * R + b4 * 4);
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&au.x));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&au.y));
        const float xs[4] = {xv.x, xv.y, xv.z, xv.w}, as[4] = {lo.x, lo.y, hi.x, hi.y};
#pragma unroll
        for (int uu = 0; uu < 4; ++uu)
#pragma unroll
          for (int v = 0; v < 4; ++v) acc[uu][v] += xs[uu] * as[v];
      }
      __half* s1 = sS1 + kk * 16 * K2s;
#pragma unroll
      for (int uu = 0; uu < 4; ++uu) {
        __half2* dst = reinterpret_cast<__half2*>(s1 + p * K2s + (j4 * 4 + uu) * R + b4 * 4);
        dst[0] = __floats2half2_rn(acc[uu][0], acc[uu][1]);
        dst[1] = __floats2half2_rn(acc[uu][2], acc[uu][3]);
      }
    }
  }
  __syncthreads();  // rows P..15 of S1 are not written: their mma outputs are dropped

  // ---- stages 2 -> 3 in registers, one warp per q
  const int ql = warp;
  if (ql < qc_valid) {
    float yv[RT][4];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt) yv[rt][0] = yv[rt][1] = yv[rt][2] = yv[rt][3] = 0.f;
    const int colq = ql * Rc;
    constexpr bool kHoistB = (R == 8);  // R = 8: the unit's B fragments stay in registers
    unsigned bh[kHoistB ? kt_n : 1][2];
    if constexpr (kHoistB) {
#pragma unroll
      for (int kt = 0; kt < kt_n; ++kt) {
        const int kb = kt * 16 + 2 * t4, n = colq + g;
        bh[kt][0] = pack_h2(sB[kb * N2s + n], sB[(kb + 1) * N2s + n]);
        bh[kt][1] = pack_h2(sB[(kb + 8) * N2s + n], sB[(kb + 9) * N2s + n]);
      }
    }
    for (int kk = 0; kk < kc_valid; ++kk) {
      const __half* s1 = sS1 + kk * 16 * K2s;
      float acc[C::NC8][4];
#pragma unroll
      for (int nc = 0; nc < C::NC8; ++nc) acc[nc][0] = acc[nc][1] = acc[nc][2] = acc[nc][3] = 0.f;
#pragma unroll
      for (int kt = 0; kt < kt_n; ++kt) {  // stage 2: 16 x 8c tiles over K = (j,b)
        const int kb = kt * 16 + 2 * t4;
        const unsigned a0 = lds_u32(s1 + g * K2s + kb), a1 = lds_u32(s1 + (g + 8) * K2s + kb);
        const unsigned a2 = lds_u32(s1 + g * K2s + kb + 8);
        const unsigned a3 = lds_u32(s1 + (g + 8) * K2s + kb + 8);
#pragma unroll
        for (int nc = 0; nc < C::NC8; ++nc) {
          if constexpr (kHoistB) {
            mma16816(acc[nc], a0, a1, a2, a3, bh[kt][0], bh[kt][1]);
          } else {
            const int n = colq + nc * 8 + g;
            mma16816(acc[nc], a0, a1, a2, a3,
                     pack_h2(sB[kb * N2s + n], sB[(kb + 1) * N2s + n]),
                     pack_h2(sB[(kb + 8) * N2s + n], sB[(kb + 9) * N2s + n]));
          }
        }
      }
      const __half* ck = sC + kk * Rc * Rrs;  // stage 3: Y += acc (as FP16 A) @ C[c, r]
      if constexpr (R == 8) {
        const unsigned a0 = pack_f2(acc[0][0], acc[0][1]), a1 = pack_f2(acc[0][2], acc[0][3]);
#pragma unroll
        for (int rt = 0; rt < RT; ++rt) {
          const int r = rt * 8 + g;
          mma1688(yv[rt], a0, a1, pack_h2(ck[(2 * t4) * Rrs + r], ck[(2 * t4 + 1) * Rrs + r]));
        }
      } else {
        const unsigned a0 = pack_f2(acc[0][0], acc[0][1]), a1 = pack_f2(acc[0][2], acc[0][3]);
        const unsigned a2 = pack_f2(acc[1][0], acc[1][1]), a3 = pack_f2(acc[1][2], acc[1][3]);
#pragma unroll
        for (int rt = 0; rt < RT; ++rt) {
          const int r = rt * 8 + g;
          mma16816(yv[rt], a0, a1, a2, a3,
                   pack_h2(ck[(2 * t4) * Rrs + r], ck[(2 * t4 + 1) * Rrs + r]),
                   pack_h2(ck[(2 * t4 + 8) * Rrs + r], ck[(2 * t4 + 9) * Rrs + r]));
        }
      }
    }
    // Y rows g, g+8 = p; columns r = 8 rt + 2 t4, +1
    const int q = q0 + ql;
#pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
      const int r = rt * 8 + 2 * t4;
      if (r < Rr) {
        if (g < P) red_add_v2(ws + (g * Q + q) * Rr + r, yv[rt][0], yv[rt][1]);
        if (g + 8 < P) red_add_v2(ws + ((g + 8) * Q + q) * Rr + r, yv[rt][2], yv[rt][3]);
      }
    }
  }

  // ---- design B: the last block converts FP32 -> FP16 and clears the workspace.
  // kTail 0: one counter, the last block of the grid converts all of y, one float at a time
  //       1: one counter, all of y, float4 loads issued together before any store
  //       2: one counter per q chunk (tile_done[q chunk]); the last of that chunk's R * nkc
  //          blocks converts only its q's, float4 loads issued together
  if constexpr (kLastBlockFinishes) {
    __shared__ bool is_last;
    unsigned int* ctr = kTail == 2 ? tile_done + rem / C::nkc : tile_done;
    const unsigned arrivals = kTail == 2 ? unsigned(R * C::nkc) : gridDim.x;
    __threadfence();
    __syncthreads();
    if (tid == 0) is_last = atomicAdd(ctr, 1u) == arrivals - 1;
    __syncthreads();
    if (is_last) {
      __threadfence();
      if constexpr (kTail == 0) {
        for (int e = tid; e < out_f; e += kThreads) {
          const float v = __ldcg(ws + e);
          y[e] = __float2half(v);
          ws[e] = 0.f;
        }
      } else {
        // rows p, q in [qs, qs + nq), all r: per p a contiguous run of nq * Rr floats
        constexpr int R4 = Rr / 4, nqmax = kTail == 2 ? QC : Q;
        constexpr int kMax = (P * nqmax * R4 + kThreads - 1) / kThreads;
        const int nq = kTail == 2 ? qc_valid : Q, qs = kTail == 2 ? q0 : 0;
        const int per_p = nq * R4, n4 = P * per_p;
        float4* ws4 = reinterpret_cast<float4*>(ws);
        float4 v[kMax];
        int idx[kMax];
#pragma unroll
        for (int s = 0; s < kMax; ++s) {  // all loads in flight at once
          const int e = tid + s * kThreads;
          idx[s] = e < n4 ? ((e / per_p) * Q + qs) * R4 + e % per_p : -1;
          if (idx[s] >= 0) v[s] = __ldcg(ws4 + idx[s]);
        }
#pragma unroll
        for (int s = 0; s < kMax; ++s) {
          if (idx[s] >= 0) {
            uint2 h;
            h.x = pack_f2(v[s].x, v[s].y);
            h.y = pack_f2(v[s].z, v[s].w);
            reinterpret_cast<uint2*>(y)[idx[s]] = h;
            ws4[idx[s]] = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
      }
      if (tid == 0) *ctr = 0;
    }
  }
}

template <bool kB, int kTail, int R, int KC, int QC>
static void launch_v3(cudaStream_t stream, const __half* x, const __half* a1, const __half* b2,
                      const __half* c3, float* ws, __half* y, unsigned int* tile_done) {
  using C = Cfg<R, KC, QC>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tr_ring_fused_v3_kernel<kB, kTail, R, KC, QC>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)C::smem));
    attr = true;
  }
  tr_ring_fused_v3_kernel<kB, kTail, R, KC, QC><<<C::units, kThreads, C::smem, stream>>>(
      x, a1, b2, c3, ws, y, tile_done);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the compiled (R, kc, qc) tilings; everything else runs the WMMA kernel
#define TR_V3_TILINGS(X) X(8, 2, 4) X(8, 2, 5) X(16, 4, 4) X(16, 4, 5)

// counters design B needs: one per q chunk (kTail 2) or one (kTail 0, 1)
constexpr int kV3MaxCounters = 16;

template <bool kB>
static bool dispatch_v3(int R, int kc, int qc, int tail, cudaStream_t s, const __half* x,
                        const __half* a1, const __half* b2, const __half* c3, float* ws,
                        __half* y, unsigned int* td) {
#define TR_V3_CASE(RR, KK, QQ)                                                       \
  if (R == RR && kc == KK && qc == QQ) {                                             \
    if (!kB || tail == 0) launch_v3<kB, 0, RR, KK, QQ>(s, x, a1, b2, c3, ws, y, td); \
    else if (tail == 1) launch_v3<kB, 1, RR, KK, QQ>(s, x, a1, b2, c3, ws, y, td);   \
    else launch_v3<kB, 2, RR, KK, QQ>(s, x, a1, b2, c3, ws, y, td);                  \
    return true;                                                                     \
  }
  TR_V3_TILINGS(TR_V3_CASE)
#undef TR_V3_CASE
  return false;
}
}  // namespace v3

bool v3_supported(int64_t R, int64_t kc, int64_t qc) {
#define TR_V3_SUP(RR, KK, QQ) if (R == RR && kc == KK && qc == QQ) return true;
  TR_V3_TILINGS(TR_V3_SUP)
#undef TR_V3_SUP
  return false;
}

// t = 1 on the real modes (8,12,20) -> (12,10,24). Design A: zero-init workspace, kernel,
// convert (3 launches). Design B: persistent workspace, the last block converts (1 launch).
torch::Tensor tr_ring_forward_v3(torch::Tensor x, torch::Tensor A1, torch::Tensor B2,
                                 torch::Tensor C3, int64_t R, int64_t kc, int64_t qc,
                                 bool last_block_finishes, torch::Tensor ws_b,
                                 torch::Tensor tile_done_b, int64_t tail) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf && x.dim() == 2 && x.is_contiguous(),
              "x must be a contiguous 2D CUDA float16 tensor");
  TORCH_CHECK(x.size(0) == 1 && x.size(1) == v3::ni * v3::nj * v3::nk, "V3 is for t = 1 on 1920");
  TORCH_CHECK(v3_supported(R, kc, qc), "V3 tiling not compiled");
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int out_f = v3::P * v3::Q * v3::Rr;
  auto y = torch::empty({1, out_f}, x.options());
  auto xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  auto a1 = reinterpret_cast<const __half*>(A1.data_ptr<at::Half>());
  auto b2 = reinterpret_cast<const __half*>(B2.data_ptr<at::Half>());
  auto c3 = reinterpret_cast<const __half*>(C3.data_ptr<at::Half>());
  auto yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  if (!last_block_finishes) {
    auto ws = torch::zeros({1, out_f}, x.options().dtype(torch::kFloat));        // 1
    v3::dispatch_v3<false>((int)R, (int)kc, (int)qc, 0, stream, xp, a1, b2, c3,  // 2
                           ws.data_ptr<float>(), nullptr, nullptr);
    tr_ring_convert_kernel<<<(out_f + 255) / 256, 256, 0, stream>>>(ws.data_ptr<float>(),  // 3
                                                                     yp, out_f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    TORCH_CHECK(tail >= 0 && tail <= 2, "tail must be 0, 1 or 2");
    TORCH_CHECK(ws_b.numel() >= out_f && tile_done_b.numel() >= v3::kV3MaxCounters,
                "design B workspace too small");
    v3::dispatch_v3<true>((int)R, (int)kc, (int)qc, (int)tail, stream, xp, a1, b2, c3,
                          ws_b.data_ptr<float>(), yp,
                          reinterpret_cast<unsigned int*>(tile_done_b.data_ptr<int32_t>()));
  }
  return y;
}

// ---- floors: what one call costs when it does nothing (tools/measure_kernels.py) ----------
__global__ void tr_ring_empty_kernel() {}

void empty_launch() {  // one empty kernel on the current stream
  tr_ring_empty_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void empty_call() {}   // Python -> C++ -> Python, no GPU work

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tr_ring_forward, "fused tensor-ring forward (design A or B)");
  m.def("forward_v3", &tr_ring_forward_v3, "t = 1 real modes: stages 2 -> 3 in registers");
  m.def("v3_supported", &v3_supported, "(R, kc, qc) compiled for V3");
  m.def("smem_bytes", &smem_bytes, "dynamic shared memory per block for a tiling");
  m.def("y_tiles", &y_tiles, "Y accumulator tiles per block for a token tile");
  m.def("max_y_tiles", &max_y_tiles, "Y tiles a block can hold in registers");
  m.def("empty_launch", &empty_launch, "launch floor: one empty kernel");
  m.def("empty_call", &empty_call, "host floor: an extension call that does nothing");
}
