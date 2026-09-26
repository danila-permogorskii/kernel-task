// Fused tensor-ring forward (v1):  y[t,(p,q,r)] = sum_{a,k} piece(a,k)[t,p,q,r]
//
// One thread block owns one ring link `a`, a chunk of `kc` input digits `k` and a tile of
// `tt` tokens. It loads B, A[a] and C[:,:,k,a] into shared memory once, then loops over its
// k values; every piece stays on-chip:
//
//   for each k in the chunk:
//     stage 1 (CUDA cores)   S1[(t,p),(j,b)]   = sum_i  x[t,i,j,k] * A[a,p,i,b]
//     stage 2 (Tensor Cores) S2[(t,p),(q,c)]   = sum_jb S1 * B2[(j,b),(q,c)]
//     stage 3 (Tensor Cores) Y[(t,p,q), r]    += sum_c  S2[(t,p,q),c] * C[c,r,k,a]
//
// Y (FP32, shared memory) is added into an FP32 workspace with atomicAdd once per block.
//   design A: a separate kernel converts the workspace to FP16          (3 launches per call)
//   design B: the last block of each token tile converts and clears it  (1 launch per call)
//
// Walkthrough with pictures: kernel-guides/G2-kernel-walkthrough.md

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

struct Dims {
  int T, ni, nj, nk, P, Q, Rr, R;  // problem: tokens, input modes, output modes, ring rank
  int kc, tt;                      // tiling: k values and tokens per block
  int Rc, Rrp;                     // R and Rr padded to 16 (Tensor Core tile edge)
  int K2p, N2c, Mp;                // stage 2: K = nj*R -> K2p, N = Q*Rc, M = tt*P -> Mp
};

__host__ __device__ inline int round16(int v) { return (v + 15) / 16 * 16; }
__host__ __device__ inline size_t align128(size_t v) { return (v + 127) & ~size_t(127); }

// Shared-memory map of one block. Host and device use the same function.
struct SmemLayout { size_t b, a, c, x, s1, s2, y, stage, total; };

__host__ __device__ inline SmemLayout smem_layout(const Dims& d) {
  SmemLayout s;
  size_t o = 0;
  s.b  = o; o = align128(o + sizeof(__half) * d.K2p * d.N2c);             // B2: all of B
  s.a  = o; o = align128(o + sizeof(float)  * d.ni * d.P * d.R);          // A[a]
  s.c  = o; o = align128(o + sizeof(__half) * d.kc * d.Rc * d.Rrp);       // C[:,:,k,a], kc of them
  s.x  = o; o = align128(o + sizeof(float)  * d.kc * d.tt * d.ni * d.nj); // x slice
  s.s1 = o; o = align128(o + sizeof(__half) * d.Mp * d.K2p);              // S1, one k at a time
  s.s2 = o; o = align128(o + sizeof(__half) * d.Mp * d.N2c);              // S2, one k at a time
  s.y  = o; o = align128(o + sizeof(float)  * d.Mp * d.Q * d.Rrp);        // Y, summed over k
  s.stage = o; o = align128(o + sizeof(float) * kWarps * 256);            // per-warp 16x16 FP32
  s.total = o;
  return s;
}

using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

template <bool kLastBlockFinishes>
__global__ void __launch_bounds__(kThreads)
tr_ring_fused_kernel(const __half* __restrict__ x,    // [T, ni*nj*nk]
                     const __half* __restrict__ A1,   // [R(a)][ni][P*R (p,b)]
                     const __half* __restrict__ B2,   // [K2p (j,b)][N2c (q,c)], zero padded
                     const __half* __restrict__ C3,   // [nk][R(a)][Rc (c)][Rrp (r)], zero padded
                     float* __restrict__ ws,          // [T, P*Q*Rr] FP32 accumulator
                     __half* __restrict__ y,          // [T, P*Q*Rr]   (design B only)
                     unsigned int* __restrict__ tile_done,  // [token tiles] (design B only)
                     const Dims d) {
  extern __shared__ __align__(128) unsigned char smem[];
  const SmemLayout L = smem_layout(d);
  __half* sB  = reinterpret_cast<__half*>(smem + L.b);
  float*  sA  = reinterpret_cast<float*>(smem + L.a);
  __half* sC  = reinterpret_cast<__half*>(smem + L.c);
  float*  sX  = reinterpret_cast<float*>(smem + L.x);
  __half* sS1 = reinterpret_cast<__half*>(smem + L.s1);
  __half* sS2 = reinterpret_cast<__half*>(smem + L.s2);
  float*  sY  = reinterpret_cast<float*>(smem + L.y);

  const int a = blockIdx.y;
  const int k0 = blockIdx.x * d.kc, t0 = blockIdx.z * d.tt;
  const int kc_valid = min(d.kc, d.nk - k0);
  const int tt_valid = min(d.tt, d.T - t0);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int in_features = d.ni * d.nj * d.nk;
  const int out_features = d.P * d.Q * d.Rr;
  const int PR = d.P * d.R;

  // ---- load: global -> shared, once per block --------------------------------------------
  {  // B2 and C: 16-byte vector copies (8 halves), all threads at once
    const int4* src = reinterpret_cast<const int4*>(B2);
    int4* dst = reinterpret_cast<int4*>(sB);
    for (int e = tid; e < d.K2p * d.N2c / 8; e += kThreads) dst[e] = src[e];
    const int cvec = d.Rc * d.Rrp / 8;  // one C[:,:,k,a] slab, in int4 units
    for (int e = tid; e < d.kc * cvec; e += kThreads) {
      const int kk = e / cvec;
      const int4 zero = make_int4(0, 0, 0, 0);
      reinterpret_cast<int4*>(sC)[e] = kk < kc_valid
          ? reinterpret_cast<const int4*>(C3)[((size_t)(k0 + kk) * d.R + a) * cvec + e % cvec]
          : zero;
    }
  }
  const __half* A1a = A1 + (size_t)a * d.ni * PR;  // sA[i][p*R+b] = A[a,p,i,b]
  for (int e = tid; e < d.ni * PR; e += kThreads) sA[e] = __half2float(A1a[e]);
  // sX[kk][t][i][j] = x[t0+t, i, j, k0+kk], zero outside the valid k / token range
  for (int e = tid; e < d.kc * d.tt * d.ni * d.nj; e += kThreads) {
    const int j = e % d.nj;
    int rest = e / d.nj;
    const int i = rest % d.ni;  rest /= d.ni;
    const int t = rest % d.tt;
    const int kk = rest / d.tt;
    float v = 0.f;
    if (kk < kc_valid && t < tt_valid)
      v = __half2float(x[(size_t)(t0 + t) * in_features + (i * d.nj + j) * d.nk + (k0 + kk)]);
    sX[e] = v;
  }
  for (int e = tid; e < d.Mp * d.Q * d.Rrp; e += kThreads) sY[e] = 0.f;
  // S1's padding (rows >= tt*P, columns >= nj*R) must be zero; stage 1 never writes it
  for (int e = tid; e < d.Mp * d.K2p; e += kThreads) sS1[e] = __float2half(0.f);
  __syncthreads();

  const int M = d.tt * d.P;
  const bool vec4 = (d.nj % 4 == 0) && (d.R % 4 == 0);  // true for the real workloads

  for (int kk = 0; kk < kc_valid; ++kk) {
    // ---- stage 1: S1[(t,p)][j*R+b] = sum_i x[t,i,j] * A[i][p*R+b] ------------------------
    // Written directly in the layout stage 2 reads (rows (t,p), columns (j,b)): no permute.
    const float* xk = sX + (size_t)kk * d.tt * d.ni * d.nj;
    if (vec4) {
      // register tile: one thread = 4 j x 4 b for one (t,p); per i: 2 float4 loads, 16 FMAs
      const int J4 = d.nj / 4, B4 = d.R / 4;
      for (int e = tid; e < M * J4 * B4; e += kThreads) {
        const int b4 = e % B4, j4 = (e / B4) % J4, row = e / (B4 * J4);
        const int t = row / d.P, p = row % d.P;
        float acc[4][4] = {};
        for (int i = 0; i < d.ni; ++i) {
          const float4 xv = *reinterpret_cast<const float4*>(xk + (t * d.ni + i) * d.nj + j4 * 4);
          const float4 av = *reinterpret_cast<const float4*>(sA + i * PR + p * d.R + b4 * 4);
          const float xs[4] = {xv.x, xv.y, xv.z, xv.w}, as[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
          for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[u][v] += xs[u] * as[v];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          __half2* dst = reinterpret_cast<__half2*>(sS1 + row * d.K2p + (j4 * 4 + u) * d.R + b4 * 4);
          dst[0] = __floats2half2_rn(acc[u][0], acc[u][1]);
          dst[1] = __floats2half2_rn(acc[u][2], acc[u][3]);
        }
      }
    } else {  // any shape: one thread = one S1 value
      for (int e = tid; e < M * d.nj * d.R; e += kThreads) {
        const int row = e / (d.nj * d.R), col = e % (d.nj * d.R);
        const int t = row / d.P, p = row % d.P, j = col / d.R, b = col % d.R;
        float acc = 0.f;
        for (int i = 0; i < d.ni; ++i)
          acc += xk[(t * d.ni + i) * d.nj + j] * sA[i * PR + p * d.R + b];
        sS1[row * d.K2p + col] = __float2half(acc);
      }
    }
    __syncthreads();

    // ---- stage 2: S2 = S1 @ B2 on Tensor Cores (FP16 in, FP32 accumulate, FP16 out) --------
    {
      float* stage = reinterpret_cast<float*>(smem + L.stage) + warp * 256;
      const int mt_n = d.Mp / 16, nt_n = d.N2c / 16, kt_n = d.K2p / 16;
      for (int tile = warp; tile < mt_n * nt_n; tile += kWarps) {
        const int mt = tile % mt_n, nt = tile / mt_n;
        FragC acc;
        wmma::fill_fragment(acc, 0.f);
        for (int kt = 0; kt < kt_n; ++kt) {
          FragA fa;
          FragB fb;
          wmma::load_matrix_sync(fa, sS1 + mt * 16 * d.K2p + kt * 16, d.K2p);
          wmma::load_matrix_sync(fb, sB + kt * 16 * d.N2c + nt * 16, d.N2c);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        // FP32 tile -> per-warp staging -> FP16 into S2 (stage 3 needs FP16 operands)
        wmma::store_matrix_sync(stage, acc, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32)
          sS2[(mt * 16 + e / 16) * d.N2c + nt * 16 + e % 16] = __float2half(stage[e]);
        __syncwarp();
      }
    }
    __syncthreads();

    // ---- stage 3: Y[(t,p,q), r] += S2[(t,p,q), c] @ C[c, r]  on Tensor Cores ---------------
    // S2 rows (t,p) hold Q blocks of Rc values, so the same bytes read as a matrix
    // [(t,p,q) x c] with row stride Rc: the second "free re-layout".
    {
      const __half* ck = sC + kk * d.Rc * d.Rrp;
      const int mt_n = d.Mp * d.Q / 16, nt_n = d.Rrp / 16, kt_n = d.Rc / 16;
      for (int tile = warp; tile < mt_n * nt_n; tile += kWarps) {
        const int mt = tile / nt_n, nt = tile % nt_n;
        float* yt = sY + mt * 16 * d.Rrp + nt * 16;
        FragC acc;
        wmma::load_matrix_sync(acc, yt, d.Rrp, wmma::mem_row_major);
        for (int kt = 0; kt < kt_n; ++kt) {
          FragA fa;
          FragB fb;
          wmma::load_matrix_sync(fa, sS2 + mt * 16 * d.Rc + kt * 16, d.Rc);
          wmma::load_matrix_sync(fb, ck + kt * 16 * d.Rrp + nt * 16, d.Rrp);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(yt, acc, d.Rrp, wmma::mem_row_major);
      }
    }
    __syncthreads();
  }

  // ---- add this block's Y into the FP32 workspace (one atomic per output per block) --------
  // consecutive threads = consecutive r = coalesced atomics
  for (int e = tid; e < tt_valid * out_features; e += kThreads) {
    const int t = e / out_features, o = e % out_features;
    const int r = o % d.Rr, pq = o / d.Rr;  // pq = p*Q + q
    atomicAdd(ws + (size_t)(t0 + t) * out_features + o, sY[(t * d.P * d.Q + pq) * d.Rrp + r]);
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
      for (int e = tid; e < tt_valid * out_features; e += kThreads) {
        const float v = __ldcg(ws + base + e);  // read from L2, not a stale L1 line
        y[base + e] = __float2half(v);
        ws[base + e] = 0.f;                     // read-and-clear: ready for the next call
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

static Dims make_dims(int64_t T, const std::vector<int64_t>& m, int64_t kc, int64_t tt) {
  TORCH_CHECK(m.size() == 7, "modes = [ni, nj, nk, P, Q, Rr, R]");
  Dims d;
  d.T = (int)T;
  d.ni = (int)m[0]; d.nj = (int)m[1]; d.nk = (int)m[2];
  d.P = (int)m[3];  d.Q = (int)m[4];  d.Rr = (int)m[5]; d.R = (int)m[6];
  d.kc = (int)kc; d.tt = (int)tt;
  d.Rc = round16(d.R); d.Rrp = round16(d.Rr);
  d.K2p = round16(d.nj * d.R); d.N2c = d.Q * d.Rc; d.Mp = round16(d.tt * d.P);
  return d;
}

int64_t smem_bytes(std::vector<int64_t> modes, int64_t kc, int64_t tt) {
  return (int64_t)smem_layout(make_dims(1, modes, kc, tt)).total;
}

template <bool kB>
static void set_smem_limit(size_t bytes) {
  static size_t current = 48 * 1024;  // default without opt-in
  if (bytes > current) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tr_ring_fused_kernel<kB>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
    current = bytes;
  }
}

// ws_b / tile_done_b: persistent buffers for design B (all zero between calls); ignored for A.
torch::Tensor tr_ring_forward(torch::Tensor x, torch::Tensor A1, torch::Tensor B2,
                              torch::Tensor C3, std::vector<int64_t> modes, int64_t kc,
                              int64_t tt, bool last_block_finishes, torch::Tensor ws_b,
                              torch::Tensor tile_done_b) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf && x.dim() == 2 && x.is_contiguous(),
              "x must be a contiguous 2D CUDA float16 tensor");
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const Dims d = make_dims(x.size(0), modes, kc, tt);
  const int64_t out_features = (int64_t)d.P * d.Q * d.Rr;
  TORCH_CHECK(x.size(1) == (int64_t)d.ni * d.nj * d.nk, "x has the wrong number of features");
  TORCH_CHECK(B2.size(0) == d.K2p && B2.size(1) == d.N2c, "B2 packing does not match");
  auto y = torch::empty({x.size(0), out_features}, x.options());
  if (d.T == 0) return y;

  const size_t smem = smem_layout(d).total;
  const dim3 grid((d.nk + d.kc - 1) / d.kc, d.R, (d.T + d.tt - 1) / d.tt);
  auto xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  auto a1 = reinterpret_cast<const __half*>(A1.data_ptr<at::Half>());
  auto b2 = reinterpret_cast<const __half*>(B2.data_ptr<at::Half>());
  auto c3 = reinterpret_cast<const __half*>(C3.data_ptr<at::Half>());
  auto yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

  if (!last_block_finishes) {  // ---------------------------- design A: 3 launches
    auto ws = torch::zeros({x.size(0), out_features}, x.options().dtype(torch::kFloat));  // 1
    set_smem_limit<false>(smem);
    tr_ring_fused_kernel<false><<<grid, kThreads, smem, stream>>>(                       // 2
        xp, a1, b2, c3, ws.data_ptr<float>(), nullptr, nullptr, d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const size_t n = (size_t)d.T * out_features;
    tr_ring_convert_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(              // 3
        ws.data_ptr<float>(), yp, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {  // ------------------------------------------------ design B: 1 launch
    TORCH_CHECK(ws_b.numel() >= (int64_t)d.T * out_features && tile_done_b.numel() >= grid.z,
                "design B workspace too small");
    set_smem_limit<true>(smem);
    tr_ring_fused_kernel<true><<<grid, kThreads, smem, stream>>>(
        xp, a1, b2, c3, ws_b.data_ptr<float>(), yp,
        reinterpret_cast<unsigned int*>(tile_done_b.data_ptr<int32_t>()), d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  m.def("smem_bytes", &smem_bytes, "dynamic shared memory per block for a tiling");
  m.def("empty_launch", &empty_launch, "launch floor: one empty kernel");
  m.def("empty_call", &empty_call, "host floor: an extension call that does nothing");
}
