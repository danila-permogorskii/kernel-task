// V0 of the stack kernel, exactly as measured in the first stack session (2026-09-26,
// results/h100/stack/r8.json): kept only as the regression baseline for tr_stack.cu.
// See tr_stack.cu for the documented version.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <mma.h>

using namespace nvcuda;
namespace cg = cooperative_groups;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxYTiles = 8;
constexpr int kStageLd = 20;

__host__ __device__ constexpr int round16(int v) { return (v + 15) / 16 * 16; }
__host__ __device__ inline size_t align128(size_t v) { return (v + 127) & ~size_t(127); }

template <int NI, int NJ, int NK, int P_, int Q_, int RR, int R_>
struct Shape {
  static constexpr int ni = NI, nj = NJ, nk = NK, P = P_, Q = Q_, Rr = RR, R = R_;
  static constexpr int Rc = round16(R_), Rrp = round16(RR), K2p = round16(NJ * R_);
  static constexpr int N2c = Q_ * Rc, K2s = K2p + 8, Rrs = Rrp + 8, PR = P_ * R_;
  static constexpr int in_f = NI * NJ * NK, out_f = P_ * Q_ * RR;
  static constexpr size_t a1_size = (size_t)R_ * NI * P_ * R_;
  static constexpr size_t b2_size = (size_t)K2p * N2c;
  static constexpr size_t c3_size = (size_t)NK * R_ * Rc * Rrp;
};
template <int R> using Up = Shape<8, 12, 20, 12, 10, 24, R>;
template <int R> using Down = Shape<12, 10, 24, 8, 12, 20, R>;
constexpr int kMaxFeatures = 2880;

struct Tiling { int kc, qc, nkc, nqc, units, Mp, Nb, N2s; };

template <class S>
__host__ __device__ inline Tiling make_tiling(int kc, int qc, int T) {
  Tiling t;
  t.kc = kc; t.qc = qc;
  t.nkc = (S::nk + kc - 1) / kc; t.nqc = (S::Q + qc - 1) / qc;
  t.units = S::R * t.nkc * t.nqc;
  t.Mp = round16(T * S::P); t.Nb = qc * S::Rc; t.N2s = t.Nb + 8;
  return t;
}

struct Layout { size_t b, a, c, x, s1, s2, stage, total; };

template <class S>
__host__ __device__ inline Layout layout(const Tiling& t, int T) {
  Layout s;
  size_t o = 0;
  s.b  = o; o = align128(o + sizeof(__half) * S::K2p * t.N2s);
  s.a  = o; o = align128(o + sizeof(float)  * S::ni * S::PR);
  s.c  = o; o = align128(o + sizeof(__half) * t.kc * S::Rc * S::Rrs);
  s.x  = o; o = align128(o + sizeof(float)  * t.kc * T * S::ni * S::nj);
  s.s1 = o; o = align128(o + sizeof(__half) * t.Mp * S::K2s);
  s.s2 = o; o = align128(o + sizeof(__half) * t.Mp * t.Nb);
  s.stage = o; o = align128(o + sizeof(float) * kWarps * 16 * kStageLd);
  s.total = o;
  return s;
}

using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

template <class S>
__device__ __noinline__ void run_unit(const Tiling& tl, const int T, const int u,
                                      const float* __restrict__ xin,
                                      float* __restrict__ yout,
                                      const __half* __restrict__ A1,
                                      const __half* __restrict__ B2,
                                      const __half* __restrict__ C3,
                                      unsigned char* smem) {
  constexpr int ni = S::ni, nj = S::nj, nk = S::nk, P = S::P, Q = S::Q, Rr = S::Rr, R = S::R;
  constexpr int Rc = S::Rc, Rrp = S::Rrp, K2p = S::K2p, N2c = S::N2c, K2s = S::K2s;
  constexpr int Rrs = S::Rrs, PR = S::PR, in_f = S::in_f, out_f = S::out_f;
  const int kc = tl.kc, qc = tl.qc, nkc = tl.nkc, Mp = tl.Mp, Nb = tl.Nb, N2s = tl.N2s;
  const int tt = T;

  const Layout L = layout<S>(tl, T);
  __half* sB  = reinterpret_cast<__half*>(smem + L.b);
  float*  sA  = reinterpret_cast<float*>(smem + L.a);
  __half* sC  = reinterpret_cast<__half*>(smem + L.c);
  float*  sX  = reinterpret_cast<float*>(smem + L.x);
  __half* sS1 = reinterpret_cast<__half*>(smem + L.s1);
  __half* sS2 = reinterpret_cast<__half*>(smem + L.s2);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  float* stage = reinterpret_cast<float*>(smem + L.stage) + warp * 16 * kStageLd;

  const int per_a = nkc * tl.nqc;
  const int a = u / per_a, rem = u % per_a;
  const int k0 = (rem % nkc) * kc, q0 = (rem / nkc) * qc;
  const int qc_valid = min(qc, Q - q0);
  const int kc_valid = min(kc, nk - k0);

  __syncthreads();

  {
    const int4* src = reinterpret_cast<const int4*>(B2);
    int4* dst = reinterpret_cast<int4*>(sB);
    const int grow = N2c / 8, brow = Nb / 8, bstride = N2s / 8;
    const int valid8 = qc_valid * Rc / 8, col0 = q0 * Rc / 8;
    const int4 zero = make_int4(0, 0, 0, 0);
    for (int e = tid; e < K2p * brow; e += kThreads) {
      const int row = e / brow, col = e % brow;
      dst[row * bstride + col] = col < valid8 ? src[row * grow + col0 + col] : zero;
    }
    constexpr int crow = Rrp / 8, cstride = Rrs / 8, cvec = Rc * crow;
    for (int e = tid; e < kc * cvec; e += kThreads) {
      const int kk = e / cvec, c = (e % cvec) / crow, col = e % crow;
      reinterpret_cast<int4*>(sC)[(kk * Rc + c) * cstride + col] = kk < kc_valid
          ? reinterpret_cast<const int4*>(C3)[((size_t)(k0 + kk) * R + a) * cvec + e % cvec]
          : zero;
    }
  }
  const __half* A1a = A1 + (size_t)a * ni * PR;
  for (int e = tid; e < ni * PR; e += kThreads) sA[e] = __half2float(A1a[e]);
  for (int e = tid; e < kc * tt * ni * nj; e += kThreads) {
    const int j = e % nj;
    int rest = e / nj;
    const int i = rest % ni;  rest /= ni;
    const int t = rest % tt;
    const int kk = rest / tt;
    float v = 0.f;
    if (kk < kc_valid) v = __ldcg(xin + (size_t)t * in_f + (i * nj + j) * nk + (k0 + kk));
    sX[e] = v;
  }
  for (int e = tid; e < Mp * K2s; e += kThreads) sS1[e] = __float2half(0.f);

  constexpr int y_nt_n = Rrp / 16;
  const int y_tiles = (Mp * qc / 16) * y_nt_n;
  FragC yacc[kMaxYTiles];
#pragma unroll
  for (int i = 0; i < kMaxYTiles; ++i) wmma::fill_fragment(yacc[i], 0.f);
  __syncthreads();

  const int M = tt * P;
  constexpr bool vec4 = (nj % 4 == 0) && (R % 4 == 0);

  for (int kk = 0; kk < kc_valid; ++kk) {
    const float* xk = sX + (size_t)kk * tt * ni * nj;
    if constexpr (vec4) {
      constexpr int J4 = nj / 4, B4 = R / 4;
      for (int e = tid; e < M * J4 * B4; e += kThreads) {
        const int b4 = e % B4, j4 = (e / B4) % J4, row = e / (B4 * J4);
        const int t = row / P, p = row % P;
        float acc[4][4] = {};
#pragma unroll
        for (int i = 0; i < ni; ++i) {
          const float4 xv = *reinterpret_cast<const float4*>(xk + (t * ni + i) * nj + j4 * 4);
          const float4 av = *reinterpret_cast<const float4*>(sA + i * PR + p * R + b4 * 4);
          const float xs[4] = {xv.x, xv.y, xv.z, xv.w}, as[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
          for (int uu = 0; uu < 4; ++uu)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[uu][v] += xs[uu] * as[v];
        }
#pragma unroll
        for (int uu = 0; uu < 4; ++uu) {
          __half2* dst = reinterpret_cast<__half2*>(sS1 + row * K2s + (j4 * 4 + uu) * R + b4 * 4);
          dst[0] = __floats2half2_rn(acc[uu][0], acc[uu][1]);
          dst[1] = __floats2half2_rn(acc[uu][2], acc[uu][3]);
        }
      }
    } else {
      for (int e = tid; e < M * nj * R; e += kThreads) {
        const int row = e / (nj * R), col = e % (nj * R);
        const int t = row / P, p = row % P, j = col / R, b = col % R;
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < ni; ++i) acc += xk[(t * ni + i) * nj + j] * sA[i * PR + p * R + b];
        sS1[row * K2s + col] = __float2half(acc);
      }
    }
    __syncthreads();
    {
      const int mt_n = Mp / 16, nt_n = Nb / 16;
      constexpr int kt_n = K2p / 16;
      for (int tile = warp; tile < mt_n * nt_n; tile += kWarps) {
        const int mt = tile % mt_n, nt = tile / mt_n;
        FragC acc;
        wmma::fill_fragment(acc, 0.f);
#pragma unroll
        for (int kt = 0; kt < kt_n; ++kt) {
          FragA fa;
          FragB fb;
          wmma::load_matrix_sync(fa, sS1 + mt * 16 * K2s + kt * 16, K2s);
          wmma::load_matrix_sync(fb, sB + kt * 16 * N2s + nt * 16, N2s);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(stage, acc, kStageLd, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32)
          sS2[(mt * 16 + e / 16) * Nb + nt * 16 + e % 16] =
              __float2half(stage[(e / 16) * kStageLd + e % 16]);
        __syncwarp();
      }
    }
    __syncthreads();
    {
      const __half* ck = sC + kk * Rc * Rrs;
#pragma unroll
      for (int i = 0; i < kMaxYTiles; ++i) {
        const int tile = warp + i * kWarps;
        if (tile < y_tiles) {
          const int mt = tile / y_nt_n, nt = tile % y_nt_n;
#pragma unroll
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

#pragma unroll
  for (int i = 0; i < kMaxYTiles; ++i) {
    const int tile = warp + i * kWarps;
    if (tile < y_tiles) {
      const int mt = tile / y_nt_n, nt = tile % y_nt_n;
      wmma::store_matrix_sync(stage, yacc[i], kStageLd, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        const int m = mt * 16 + e / 16, r = nt * 16 + e % 16;
        const int ql = m % qc, tp = m / qc, t = tp / P, p = tp % P;
        if (r < Rr && t < tt && ql < qc_valid)
          atomicAdd(yout + (size_t)t * out_f + (p * Q + q0 + ql) * Rr + r,
                    stage[(e / 16) * kStageLd + e % 16]);
      }
      __syncwarp();
    }
  }
}

__device__ __forceinline__ void grid_barrier(unsigned int* sync, unsigned int nblocks) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> count(sync[0]), gen(sync[1]);
    const unsigned int g = gen.load(cuda::memory_order_relaxed);
    if (count.fetch_add(1u, cuda::memory_order_acq_rel) == nblocks - 1) {
      count.store(0u, cuda::memory_order_relaxed);
      gen.fetch_add(1u, cuda::memory_order_release);
    } else {
      while (gen.load(cuda::memory_order_acquire) == g) {
      }
    }
  }
  __syncthreads();
}

struct StackArgs {
  const __half* x;
  __half* y;
  float* buf;
  unsigned int* sync;
  const __half* A1[2];
  const __half* B2[2];
  const __half* C3[2];
  Tiling tl[2];
  int L, T, bufstride;
};

template <int R>
__global__ void __launch_bounds__(kThreads, 2) tr_stack_kernel(const StackArgs g) {
  extern __shared__ __align__(128) unsigned char smem[];
  using U = Up<R>;
  using D = Down<R>;
  const int gtid = blockIdx.x * kThreads + threadIdx.x, gstride = gridDim.x * kThreads;
  const int T = g.T, BS = g.bufstride;

  for (int e = gtid; e < T * U::in_f; e += gstride) g.buf[e] = __half2float(g.x[e]);
  for (int e = gtid; e < BS; e += gstride) g.buf[BS + e] = 0.f;
  grid_barrier(g.sync, gridDim.x);

  for (int l = 0; l < g.L; ++l) {
    const float* in = g.buf + (size_t)(l % 3) * BS;
    float* out = g.buf + (size_t)((l + 1) % 3) * BS;
    float* clr = g.buf + (size_t)((l + 2) % 3) * BS;
    for (int e = gtid; e < BS; e += gstride) clr[e] = 0.f;
    const int idx = l >> 1;
    if ((l & 1) == 0) {
      const Tiling tl = g.tl[0];
      for (int u = blockIdx.x; u < tl.units; u += gridDim.x)
        run_unit<U>(tl, T, u, in, out, g.A1[0] + idx * U::a1_size,
                    g.B2[0] + idx * U::b2_size, g.C3[0] + idx * U::c3_size, smem);
    } else {
      const Tiling tl = g.tl[1];
      for (int u = blockIdx.x; u < tl.units; u += gridDim.x)
        run_unit<D>(tl, T, u, in, out, g.A1[1] + idx * D::a1_size,
                    g.B2[1] + idx * D::b2_size, g.C3[1] + idx * D::c3_size, smem);
    }
    grid_barrier(g.sync, gridDim.x);
  }

  const int n = T * (((g.L - 1) & 1) == 0 ? U::out_f : D::out_f);
  const float* fin = g.buf + (size_t)(g.L % 3) * BS;
  for (int e = gtid; e < n; e += gstride) g.y[e] = __float2half(__ldcg(fin + e));
}

template <int R>
static void launch(StackArgs& args, size_t smem, cudaStream_t stream) {
  auto kern = tr_stack_kernel<R>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)smem));
  int per_sm = 0, dev = 0, sms = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, kThreads, smem));
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  const int grid = per_sm * sms;
  TORCH_CHECK(grid > 0, "stack kernel does not fit on an SM");
  void* params[] = {&args};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kern, dim3(grid), dim3(kThreads), params,
                                             smem, stream));
}

torch::Tensor stack_forward(torch::Tensor x, std::vector<torch::Tensor> A1,
                            std::vector<torch::Tensor> B2, std::vector<torch::Tensor> C3,
                            int64_t L, int64_t R, std::vector<int64_t> tiling, torch::Tensor buf,
                            torch::Tensor sync) {
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int T = (int)x.size(0);
  StackArgs args;
  args.x = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  const int64_t out_f = ((L - 1) & 1) == 0 ? 2880 : 1920;
  auto y = torch::empty({x.size(0), out_f}, x.options());
  args.y = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  args.buf = buf.data_ptr<float>();
  args.sync = reinterpret_cast<unsigned int*>(sync.data_ptr<int32_t>());
  for (int i = 0; i < 2; ++i) {
    args.A1[i] = reinterpret_cast<const __half*>(A1[i].data_ptr<at::Half>());
    args.B2[i] = reinterpret_cast<const __half*>(B2[i].data_ptr<at::Half>());
    args.C3[i] = reinterpret_cast<const __half*>(C3[i].data_ptr<at::Half>());
  }
  args.L = (int)L; args.T = T; args.bufstride = T * kMaxFeatures;
  if (R == 8) {
    args.tl[0] = make_tiling<Up<8>>((int)tiling[0], (int)tiling[1], T);
    args.tl[1] = make_tiling<Down<8>>((int)tiling[2], (int)tiling[3], T);
    launch<8>(args, std::max(layout<Up<8>>(args.tl[0], T).total,
                             layout<Down<8>>(args.tl[1], T).total), stream);
  } else {
    args.tl[0] = make_tiling<Up<16>>((int)tiling[0], (int)tiling[1], T);
    args.tl[1] = make_tiling<Down<16>>((int)tiling[2], (int)tiling[3], T);
    launch<16>(args, std::max(layout<Up<16>>(args.tl[0], T).total,
                              layout<Down<16>>(args.tl[1], T).total), stream);
  }
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &stack_forward, "V0 stack kernel (regression baseline)");
}
