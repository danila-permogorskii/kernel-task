// Fused tensor-ring forward:  y[t,(p,q,r)] = sum_{a,k} piece(a,k)[t,p,q,r]
//
// One thread block computes the pieces for one ring link `a`, a chunk of `kc` input digits
// `k` and a tile of `tt` tokens, entirely in shared memory:
//
//   stage 1 (CUDA cores)   S1[(k,t,p),(j,b)] = sum_i   x[t,i,j,k] * A[a,p,i,b]
//   stage 2 (Tensor Cores) S2[(k,t,p),(q,c)] = sum_jb  S1 * B2[(j,b),(q,c)]      (WMMA 16x16x16)
//   stage 3 (CUDA cores)   y[t,(p,q,r)]     += sum_k,c S2[(k,t,p,q),c] * C[c,r,k,a]
//
// The partial y is added into an FP32 workspace with atomicAdd. Then either
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

constexpr int kThreads = 256;  // 8 warps per block

struct Dims {
  int T, ni, nj, nk, P, Q, Rr, R;  // problem: tokens, input modes, output modes, ring rank
  int kc, tt;                      // tiling: k values and tokens per block
  int K2p, N2p, Mp;                // stage-2 GEMM sizes, padded to multiples of 16
};

__host__ __device__ inline size_t align128(size_t v) { return (v + 127) & ~size_t(127); }

// Shared-memory map of one block. Host and device use the same function.
struct SmemLayout { size_t x, a, c, b, s1, s2, total; };

__host__ __device__ inline SmemLayout smem_layout(const Dims& d) {
  SmemLayout s;
  size_t o = 0;
  s.x  = o; o = align128(o + sizeof(float)  * d.kc * d.tt * d.ni * d.nj);  // x slice
  s.a  = o; o = align128(o + sizeof(float)  * d.ni * d.P * d.R);           // A[a]
  s.c  = o; o = align128(o + sizeof(float)  * d.kc * d.R * d.Rr);          // C[:,:,k,a] for kc k's
  s.b  = o; o = align128(o + sizeof(__half) * d.K2p * d.N2p);              // B2 (all of B)
  s.s1 = o; o = align128(o + sizeof(__half) * d.Mp * d.K2p);               // S1 (FP16, stage-2 operand)
  s.s2 = o; o = align128(o + sizeof(float)  * d.Mp * d.N2p);               // S2 (FP32 accumulators)
  s.total = o;
  return s;
}

template <bool kLastBlockFinishes>
__global__ void __launch_bounds__(kThreads)
tr_ring_fused_kernel(const __half* __restrict__ x,    // [T, ni*nj*nk]
                     const __half* __restrict__ A1,   // [R(a)][ni][P*R(b)]
                     const __half* __restrict__ B2,   // [K2p = nj*R (j,b)][N2p = Q*R (q,c)], zero padded
                     const __half* __restrict__ C3,   // [nk][R(a)][R(c)][Rr]
                     float* __restrict__ ws,          // [T, P*Q*Rr] FP32 accumulator
                     __half* __restrict__ y,          // [T, P*Q*Rr]   (design B only)
                     unsigned int* __restrict__ tile_done,  // [token tiles] (design B only)
                     const Dims d) {
  extern __shared__ __align__(128) unsigned char smem[];
  const SmemLayout L = smem_layout(d);
  float*  sX  = reinterpret_cast<float*>(smem + L.x);
  float*  sA  = reinterpret_cast<float*>(smem + L.a);
  float*  sC  = reinterpret_cast<float*>(smem + L.c);
  __half* sB  = reinterpret_cast<__half*>(smem + L.b);
  __half* sS1 = reinterpret_cast<__half*>(smem + L.s1);
  float*  sS2 = reinterpret_cast<float*>(smem + L.s2);

  const int a = blockIdx.y;
  const int k0 = blockIdx.x * d.kc, t0 = blockIdx.z * d.tt;
  const int kc_valid = min(d.kc, d.nk - k0);
  const int tt_valid = min(d.tt, d.T - t0);
  const int tid = threadIdx.x;
  const int in_features = d.ni * d.nj * d.nk;
  const int out_features = d.P * d.Q * d.Rr;
  const int PR = d.P * d.R;

  // ---- load: global -> shared, FP16 -> FP32 --------------------------------------------
  // sX[kk][t][i][j] = x[t0+t, i, j, k0+kk]   (zero outside the valid k / token range)
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
  // sB = B2, 16-byte vector loads (8 halves each): every thread issues its loads at once,
  // so the ~500-cycle global latency is paid once, not once per Tensor Core step
  {
    const int4* src = reinterpret_cast<const int4*>(B2);
    int4* dst = reinterpret_cast<int4*>(sB);
    for (int e = tid; e < d.K2p * d.N2p / 8; e += kThreads) dst[e] = src[e];
  }
  // sA[i][p*R+b] = A[a,p,i,b]
  const __half* A1a = A1 + (size_t)a * d.ni * PR;
  for (int e = tid; e < d.ni * PR; e += kThreads) sA[e] = __half2float(A1a[e]);
  // sC[kk][c][r] = C[c,r,k0+kk,a]
  const int RRr = d.R * d.Rr;
  for (int e = tid; e < d.kc * RRr; e += kThreads) {
    const int kk = e / RRr;
    sC[e] = kk < kc_valid ? __half2float(C3[((size_t)(k0 + kk) * d.R + a) * RRr + e % RRr]) : 0.f;
  }
  __syncthreads();

  // ---- stage 1: S1[row][col], row = (kk*tt + t)*P + p, col = j*R + b ----------------------
  // Written directly in the layout stage 2 reads: no separate permute.
  const int M = d.kc * d.tt * d.P;
  const int jR = d.nj * d.R;
  for (int e = tid; e < d.Mp * d.K2p; e += kThreads) {
    const int row = e / d.K2p, col = e % d.K2p;
    float acc = 0.f;
    if (row < M && col < jR) {
      const int p = row % d.P, kt = row / d.P;  // kt = kk*tt + t
      const int j = col / d.R, b = col % d.R;
      const float* xr = sX + (size_t)kt * d.ni * d.nj + j;  // x[kk,t,i,j], stride nj over i
      const float* ar = sA + p * d.R + b;                    // A[a,p,i,b],  stride PR over i
      for (int i = 0; i < d.ni; ++i) acc += xr[i * d.nj] * ar[i * PR];
    }
    sS1[e] = __float2half(acc);
  }
  __syncthreads();

  // ---- stage 2: S2 = S1 @ B2 on Tensor Cores (WMMA, FP16 in, FP32 accumulate) ------------
  // Both operands come from shared memory. Each warp takes a contiguous run of 16x16
  // output tiles.
  {
    const int warp = tid / 32, nwarps = kThreads / 32;
    const int mt_n = d.Mp / 16, nt_n = d.N2p / 16, kt_n = d.K2p / 16;
    const int tiles = mt_n * nt_n;
    const int per_warp = (tiles + nwarps - 1) / nwarps;
    const int first = warp * per_warp, last = min(first + per_warp, tiles);
    for (int tile = first; tile < last; ++tile) {
      const int nt = tile / mt_n, mt = tile % mt_n;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
      for (int kt = 0; kt < kt_n; ++kt) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
        wmma::load_matrix_sync(fa, sS1 + (size_t)mt * 16 * d.K2p + kt * 16, d.K2p);
        wmma::load_matrix_sync(fb, sB + (size_t)kt * 16 * d.N2p + nt * 16, d.N2p);
        wmma::mma_sync(acc, fa, fb, acc);
      }
      wmma::store_matrix_sync(sS2 + (size_t)mt * 16 * d.N2p + nt * 16, acc, d.N2p,
                              wmma::mem_row_major);
    }
  }
  __syncthreads();

  // ---- stage 3: y[t,(p,q,r)] += sum_kk sum_c S2[(kk,t,p),(q,c)] * C[c,r,k0+kk,a] ----------
  // One thread per output value; consecutive threads = consecutive r = coalesced atomics.
  for (int e = tid; e < tt_valid * out_features; e += kThreads) {
    const int t = e / out_features, o = e % out_features;
    const int r = o % d.Rr, pq = o / d.Rr;
    const int q = pq % d.Q, p = pq / d.Q;
    float acc = 0.f;
    for (int kk = 0; kk < kc_valid; ++kk) {
      const float* s2 = sS2 + (size_t)((kk * d.tt + t) * d.P + p) * d.N2p + q * d.R;
      const float* cc = sC + kk * RRr + r;
      for (int c = 0; c < d.R; ++c) acc += s2[c] * cc[c * d.Rr];
    }
    atomicAdd(ws + (size_t)(t0 + t) * out_features + o, acc);
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

static Dims make_dims(int64_t T, const std::vector<int64_t>& m, int64_t kc, int64_t tt,
                      int64_t K2p, int64_t N2p) {
  TORCH_CHECK(m.size() == 7, "modes = [ni, nj, nk, P, Q, Rr, R]");
  Dims d;
  d.T = (int)T;
  d.ni = (int)m[0]; d.nj = (int)m[1]; d.nk = (int)m[2];
  d.P = (int)m[3];  d.Q = (int)m[4];  d.Rr = (int)m[5]; d.R = (int)m[6];
  d.kc = (int)kc; d.tt = (int)tt; d.K2p = (int)K2p; d.N2p = (int)N2p;
  d.Mp = (int)((kc * tt * d.P + 15) / 16 * 16);
  return d;
}

int64_t smem_bytes(std::vector<int64_t> modes, int64_t kc, int64_t tt, int64_t K2p, int64_t N2p) {
  return (int64_t)smem_layout(make_dims(1, modes, kc, tt, K2p, N2p)).total;
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

// ws / tile_done: persistent buffers for design B (all zero between calls); ignored for A.
torch::Tensor tr_ring_forward(torch::Tensor x, torch::Tensor A1, torch::Tensor B2,
                              torch::Tensor C3, std::vector<int64_t> modes, int64_t kc,
                              int64_t tt, int64_t K2p, int64_t N2p, bool last_block_finishes,
                              torch::Tensor ws_b, torch::Tensor tile_done_b) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf && x.dim() == 2 && x.is_contiguous(),
              "x must be a contiguous 2D CUDA float16 tensor");
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const Dims d = make_dims(x.size(0), modes, kc, tt, K2p, N2p);
  const int64_t out_features = (int64_t)d.P * d.Q * d.Rr;
  TORCH_CHECK(x.size(1) == (int64_t)d.ni * d.nj * d.nk, "x has the wrong number of features");
  auto y = torch::empty({x.size(0), out_features}, x.options());
  if (d.T == 0) return y;

  const size_t smem = smem_layout(d).total;
  const dim3 grid((d.nk + d.kc - 1) / d.kc, d.R, (d.T + d.tt - 1) / d.tt);

  if (!last_block_finishes) {  // ---------------------------- design A: 3 launches
    auto ws = torch::zeros({x.size(0), out_features}, x.options().dtype(torch::kFloat));  // 1
    set_smem_limit<false>(smem);
    tr_ring_fused_kernel<false><<<grid, kThreads, smem, stream>>>(                       // 2
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(A1.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B2.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(C3.data_ptr<at::Half>()),
        ws.data_ptr<float>(), nullptr, nullptr, d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const size_t n = (size_t)d.T * out_features;
    tr_ring_convert_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(              // 3
        ws.data_ptr<float>(), reinterpret_cast<__half*>(y.data_ptr<at::Half>()), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {  // ------------------------------------------------ design B: 1 launch
    TORCH_CHECK(ws_b.numel() >= (int64_t)d.T * out_features && tile_done_b.numel() >= grid.z,
                "design B workspace too small");
    set_smem_limit<true>(smem);
    tr_ring_fused_kernel<true><<<grid, kThreads, smem, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(A1.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B2.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(C3.data_ptr<at::Half>()),
        ws_b.data_ptr<float>(), reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
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
