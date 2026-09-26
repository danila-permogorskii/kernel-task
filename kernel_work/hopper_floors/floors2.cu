// Hopper floors, part 2 (H100, sm_90), after danila-permogorskii/batch1-cdna 02-gemv / 03-dlops.
//
//   mma_chain    one warp, clock64: cycles per mma.sync.m16n8k16 (FP16 in, FP32 accumulate)
//     0 one dependent chain (accumulator forwarded to the next mma)
//     1 two independent chains interleaved      2 four independent chains
//     3 one chain, an FMUL touches the accumulator after every mma (breaks the chain)
//   lane_move    one warp, clock64: cycles per dependent step
//     0 __shfl_down_sync   1 through shared memory (store, __syncwarp, load)
//   block_sync   one block of 256 threads: cycles per __syncthreads
//   dense_stack  Kog-style dense baseline in the SAME persistent framework as the ring stack:
//                L different FP16 weights (alternating 2880x1920 / 1920x2880), one warp per
//                output row, 16-byte loads, 4 independent accumulators, x in shared memory,
//                monotonic grid barrier between layers. Load variants:
//     0 plain ld.global.v4      1 ld.global.nc (__ldg)      2 ld.global.cs (__ldcs, streaming)
//     3 ld.global.nc.L1::no_allocate
//   PD = prefetch distance: at the start of layer l every warp issues cp.async.bulk.prefetch.L2
//   for its own rows of layer l + PD (weights do not depend on x: Kog-style weight streaming).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>

// ---- mma chains -----------------------------------------------------------------------------
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4],
                                         const unsigned (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int V>
__global__ void mma_chain_kernel(float* out, long long* cycles, int iters) {
  const unsigned lane = threadIdx.x;
  unsigned a[4] = {0x3c003c00u ^ lane, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};  // ~1.0 halves
  unsigned b[2] = {0x00003c00u, 0x3c000000u};
  float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
  __syncwarp();
  const long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
    if constexpr (V == 0) {
      mma16816(d0, a, b); mma16816(d0, a, b); mma16816(d0, a, b); mma16816(d0, a, b);
    } else if constexpr (V == 1) {
      mma16816(d0, a, b); mma16816(d1, a, b); mma16816(d0, a, b); mma16816(d1, a, b);
    } else if constexpr (V == 2) {
      mma16816(d0, a, b); mma16816(d1, a, b); mma16816(d2, a, b); mma16816(d3, a, b);
    } else {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        mma16816(d0, a, b);
        d0[0] *= 0.999f;  // another opcode on the accumulator between two mma
      }
    }
  }
  const long long t1 = clock64();
  if (lane == 0) cycles[0] = t1 - t0;
  out[lane] = d0[0] + d0[1] + d1[2] + d2[3] + d3[0];
}

// ---- moving a value between lanes -------------------------------------------------------------
template <int V>
__global__ void lane_move_kernel(float* out, long long* cycles, int iters) {
  __shared__ float s[32];
  const int lane = threadIdx.x;
  float v = (float)lane;
  __syncwarp();
  const long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
    if constexpr (V == 0) {
      v = __shfl_down_sync(0xffffffffu, v, 1) + 1.f;
    } else {
      s[lane] = v;
      __syncwarp();
      v = s[(lane + 1) & 31] + 1.f;
      __syncwarp();
    }
  }
  const long long t1 = clock64();
  if (lane == 0) cycles[0] = t1 - t0;
  out[lane] = v;
}

__global__ void block_sync_kernel(long long* cycles, int iters) {
  const long long t0 = clock64();
  for (int it = 0; it < iters; ++it) __syncthreads();
  const long long t1 = clock64();
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
}

std::vector<int64_t> micro(int64_t which, int64_t variant, int64_t iters, torch::Tensor out) {
  auto st = at::cuda::getCurrentCUDAStream();
  auto cyc = torch::zeros({1}, out.options().dtype(torch::kLong));
  long long* c = reinterpret_cast<long long*>(cyc.data_ptr<int64_t>());
  float* o = out.data_ptr<float>();
  if (which == 0) {
    switch (variant) {
      case 0: mma_chain_kernel<0><<<1, 32, 0, st>>>(o, c, (int)iters); break;
      case 1: mma_chain_kernel<1><<<1, 32, 0, st>>>(o, c, (int)iters); break;
      case 2: mma_chain_kernel<2><<<1, 32, 0, st>>>(o, c, (int)iters); break;
      default: mma_chain_kernel<3><<<1, 32, 0, st>>>(o, c, (int)iters); break;
    }
  } else if (which == 1) {
    if (variant == 0) lane_move_kernel<0><<<1, 32, 0, st>>>(o, c, (int)iters);
    else lane_move_kernel<1><<<1, 32, 0, st>>>(o, c, (int)iters);
  } else {
    block_sync_kernel<<<1, 256, 0, st>>>(c, (int)iters);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaStreamSynchronize(st));
  return {cyc.item<int64_t>()};
}

// ---- dense stack ------------------------------------------------------------------------------
constexpr int kDThreads = 256;

template <int V>
__device__ __forceinline__ int4 wload(const int4* p) {
  if constexpr (V == 0) return *p;
  if constexpr (V == 1) return __ldg(p);
  if constexpr (V == 2) return __ldcs(p);
  int4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ float dot8(const int4 w, const float* x) {
  const __half2* h = reinterpret_cast<const __half2*>(&w);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __half22float2(h[i]);
    s = fmaf(f.x, x[2 * i], s);
    s = fmaf(f.y, x[2 * i + 1], s);
  }
  return s;
}

struct DenseArgs {
  const __half* Wup;    // [layers/2][2880][1920]
  const __half* Wdown;  // [layers/2][1920][2880]
  const __half* x;      // [1920]
  __half* y;
  float* buf;           // 2 x 2880
  unsigned* sync;       // [0] monotonic counter
  unsigned base;
  int L;
};

__device__ __forceinline__ void prefetch_l2(const void* p, unsigned bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

template <int V, int PD>
__global__ void __launch_bounds__(kDThreads) dense_stack_kernel(const DenseArgs g) {
  __shared__ __align__(16) float sx[2880];
  const int tid = threadIdx.x, lane = tid & 31;
  const int gwarp = blockIdx.x * (kDThreads / 32) + tid / 32;
  const int nwarps = gridDim.x * (kDThreads / 32);
  const int gtid = blockIdx.x * kDThreads + tid, gstride = gridDim.x * kDThreads;
  for (int e = gtid; e < 1920; e += gstride) g.buf[e] = __half2float(g.x[e]);
  unsigned target = g.base;
  auto barrier = [&]() {
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(g.sync), "r"(1u) : "memory");
      unsigned v;
      target += gridDim.x;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(g.sync) : "memory");
      } while ((int)(v - target) < 0);
    } else {
      target += gridDim.x;
    }
    __syncthreads();
  };
  auto prefetch_layer = [&](int l) {  // this warp's rows of layer l into L2
    if (l >= g.L || lane != 0) return;
    const bool up = (l & 1) == 0;
    const int K = up ? 1920 : 2880, N = up ? 2880 : 1920;
    const __half* W = up ? g.Wup + (size_t)(l >> 1) * 2880 * 1920
                         : g.Wdown + (size_t)(l >> 1) * 1920 * 2880;
    for (int row = gwarp; row < N; row += nwarps) prefetch_l2(W + (size_t)row * K, K * 2);
  };
  if constexpr (PD > 0) {
    for (int d = 0; d < PD; ++d) prefetch_layer(d);
  }
  barrier();
  for (int l = 0; l < g.L; ++l) {
    if constexpr (PD > 0) prefetch_layer(l + PD);
    const bool up = (l & 1) == 0;
    const int K = up ? 1920 : 2880, N = up ? 2880 : 1920;
    const float* in = g.buf + (l & 1) * 2880;
    float* out = g.buf + ((l + 1) & 1) * 2880;
    const __half* W = up ? g.Wup + (size_t)(l >> 1) * 2880 * 1920
                         : g.Wdown + (size_t)(l >> 1) * 1920 * 2880;
    for (int e = tid; e < K; e += kDThreads) sx[e] = __ldcg(in + e);
    __syncthreads();
    const int K8 = K / 8;  // int4 per row
    for (int row = gwarp; row < N; row += nwarps) {
      const int4* w = reinterpret_cast<const int4*>(W + (size_t)row * K);
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      int c = lane;
      for (; c + 96 < K8; c += 128) {  // 4 independent 16-byte loads in flight per lane
        const int4 w0 = wload<V>(w + c), w1 = wload<V>(w + c + 32);
        const int4 w2 = wload<V>(w + c + 64), w3 = wload<V>(w + c + 96);
        acc[0] += dot8(w0, sx + 8 * c);
        acc[1] += dot8(w1, sx + 8 * (c + 32));
        acc[2] += dot8(w2, sx + 8 * (c + 64));
        acc[3] += dot8(w3, sx + 8 * (c + 96));
      }
      for (; c < K8; c += 32) acc[0] += dot8(wload<V>(w + c), sx + 8 * c);
      float s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      if (lane == 0) out[row] = s;
    }
    barrier();
  }
  const float* fin = g.buf + (g.L & 1) * 2880;
  const int n = ((g.L - 1) & 1) == 0 ? 2880 : 1920;
  for (int e = gtid; e < n; e += gstride) g.y[e] = __float2half(__ldcg(fin + e));
}

// grid: 0 = all resident blocks. base: monotonic counter value at launch (caller tracks it).
torch::Tensor dense_stack(torch::Tensor x, torch::Tensor Wup, torch::Tensor Wdown, int64_t L,
                          torch::Tensor buf, torch::Tensor sync, int64_t base, int64_t variant,
                          int64_t grid) {
  auto st = at::cuda::getCurrentCUDAStream();
  DenseArgs g;
  g.Wup = reinterpret_cast<const __half*>(Wup.data_ptr<at::Half>());
  g.Wdown = reinterpret_cast<const __half*>(Wdown.data_ptr<at::Half>());
  g.x = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  auto y = torch::empty({1, ((L - 1) & 1) == 0 ? 2880 : 1920}, x.options());
  g.y = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  g.buf = buf.data_ptr<float>();
  g.sync = reinterpret_cast<unsigned*>(sync.data_ptr<int32_t>());
  g.base = (unsigned)base;
  g.L = (int)L;
  void* kern = nullptr;
  switch (variant) {  // 0-3: load variants, no prefetch; 10 + d: plain loads, prefetch d ahead
    case 0: kern = (void*)dense_stack_kernel<0, 0>; break;
    case 1: kern = (void*)dense_stack_kernel<1, 0>; break;
    case 2: kern = (void*)dense_stack_kernel<2, 0>; break;
    case 3: kern = (void*)dense_stack_kernel<3, 0>; break;
    case 11: kern = (void*)dense_stack_kernel<0, 1>; break;
    case 12: kern = (void*)dense_stack_kernel<0, 2>; break;
    default: kern = (void*)dense_stack_kernel<0, 4>; break;
  }
  int per_sm = 0, dev = 0, sms = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, kDThreads, 0));
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  int cap = per_sm * sms;
  if (grid <= 0 || grid > cap) grid = cap;
  void* params[] = {&g};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kern, dim3((unsigned)grid), dim3(kDThreads),
                                             params, 0, st));
  return y;
}

int64_t dense_grid(int64_t variant) {
  int per_sm = 0, dev = 0, sms = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, dense_stack_kernel<0, 0>,
                                                               kDThreads, 0));
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  return per_sm * sms;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("micro", &micro, "which: 0 mma chain, 1 lane move, 2 __syncthreads -> [cycles]");
  m.def("dense_stack", &dense_stack);
  m.def("dense_grid", &dense_grid);
}
