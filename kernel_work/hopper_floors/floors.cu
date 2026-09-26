// Hopper floors: what the non-compute parts of one stack layer cost on the H100, in isolation.
// (In the spirit of danila-permogorskii/batch1-cdna: one hardware mechanism per experiment.)
//
//   barrier_bench  grid-wide barrier variants, µs per barrier
//     0 ours: counter + generation, __threadfence in every thread (the stack kernel today)
//     1 cooperative groups grid.sync()
//     2 ours + __nanosleep backoff while spinning
//     3 ours, fence only in thread 0 (what cooperative groups does)
//     4 monotonic counter: red.release (no return) to arrive, ld.acquire spin to wait
//     5 hierarchical: cluster barrier (hardware), one red.release per cluster, leader spins,
//       cluster barrier again
//   copy_bench     one L2-resident chunk -> shared memory, µs per copy (latency)
//     0 all 256 threads, 16-byte loads      1 cp.async.bulk by thread 0 + mbarrier
//     2 cp.async.bulk split into 4 pieces issued by 4 threads
//   atomic_bench   264 blocks x 1152 floats added into a 2880-float vector, µs per round
//     0 scalar atomicAdd                   1 red.global.add.v4.f32 (sm_90)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cstdint>

namespace cg = cooperative_groups;
constexpr int kThreads = 256;

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_add(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// ---- barriers -----------------------------------------------------------------------------
template <bool kAllFence, int kSleepNs>
__device__ __forceinline__ void barrier_gen(unsigned* sync, unsigned nb) {
  if (kAllFence) __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    if (!kAllFence) __threadfence();
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> count(sync[0]), gen(sync[1]);
    const unsigned g = gen.load(cuda::memory_order_relaxed);
    if (count.fetch_add(1u, cuda::memory_order_acq_rel) == nb - 1) {
      count.store(0u, cuda::memory_order_relaxed);
      gen.fetch_add(1u, cuda::memory_order_release);
    } else {
      while (gen.load(cuda::memory_order_acquire) == g) {
        if (kSleepNs) __nanosleep(kSleepNs);
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void barrier_mono(unsigned* ctr, unsigned target) {
  __syncthreads();
  if (threadIdx.x == 0) {
    red_release_add(ctr, 1u);
    while (ld_acquire(ctr) < target) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void barrier_cluster_mono(unsigned* ctr, unsigned target) {
  cg::cluster_group cl = cg::this_cluster();
  cl.sync();  // every block of the cluster has arrived (hardware barrier)
  if (cl.block_rank() == 0 && threadIdx.x == 0) {
    red_release_add(ctr, 1u);
    while (ld_acquire(ctr) < target) {
    }
  }
  cl.sync();  // the leader saw everyone: release the cluster
}

template <int V>
__global__ void __launch_bounds__(kThreads) barrier_kernel(unsigned* sync, int iters) {
  const unsigned nb = gridDim.x;
  unsigned nclusters = nb;
  if constexpr (V == 5) nclusters = nb / cg::this_cluster().num_blocks();
  for (int it = 0; it < iters; ++it) {
    if constexpr (V == 0) barrier_gen<true, 0>(sync, nb);
    if constexpr (V == 1) cg::this_grid().sync();
    if constexpr (V == 2) barrier_gen<true, 64>(sync, nb);
    if constexpr (V == 3) barrier_gen<false, 0>(sync, nb);
    if constexpr (V == 4) barrier_mono(sync + 2, nb * (it + 1));
    if constexpr (V == 5) barrier_cluster_mono(sync + 2, nclusters * (it + 1));
  }
}

template <int V>
static void launch_barrier(int grid, int cluster, int iters, unsigned* sync, cudaStream_t st) {
  auto kern = barrier_kernel<V>;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kThreads);
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  int nattr = 0;
  if (V == 5) {
    attr[nattr].id = cudaLaunchAttributeClusterDimension;
    attr[nattr].val.clusterDim.x = cluster;
    attr[nattr].val.clusterDim.y = 1;
    attr[nattr].val.clusterDim.z = 1;
    ++nattr;
  } else {
    attr[nattr].id = cudaLaunchAttributeCooperative;  // guarantees co-residency
    attr[nattr].val.cooperative = 1;
    ++nattr;
  }
  cfg.attrs = attr;
  cfg.numAttrs = nattr;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, sync, iters));
}

void barrier_bench(int64_t variant, int64_t grid, int64_t cluster, int64_t iters,
                   torch::Tensor sync) {
  auto st = at::cuda::getCurrentCUDAStream();
  auto* s = reinterpret_cast<unsigned*>(sync.data_ptr<int32_t>());
  C10_CUDA_CHECK(cudaMemsetAsync(s, 0, 16, st));
  switch (variant) {
    case 0: launch_barrier<0>((int)grid, 1, (int)iters, s, st); break;
    case 1: launch_barrier<1>((int)grid, 1, (int)iters, s, st); break;
    case 2: launch_barrier<2>((int)grid, 1, (int)iters, s, st); break;
    case 3: launch_barrier<3>((int)grid, 1, (int)iters, s, st); break;
    case 4: launch_barrier<4>((int)grid, 1, (int)iters, s, st); break;
    default: launch_barrier<5>((int)grid, (int)cluster, (int)iters, s, st); break;
  }
}

// ---- global (L2) -> shared copies ---------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes,
                                         uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned phase) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(phase)
      : "memory");
}

template <int V>
__global__ void __launch_bounds__(kThreads) copy_kernel(const int4* src, int bytes, int iters,
                                                        float* sink) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) uint64_t bar;
  const int tid = threadIdx.x;
  const int n16 = bytes / 16;
  const int4* s = src + (size_t)(blockIdx.x % 8) * n16;  // 8 distinct chunks, L2-resident
  if (V != 0 && tid == 0) {
    mbar_init(&bar, V == 2 ? 4 : 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  float acc = 0.f;
  for (int it = 0; it < iters; ++it) {
    if constexpr (V == 0) {
      int4* d = reinterpret_cast<int4*>(sm);
      for (int e = tid; e < n16; e += kThreads) d[e] = s[e];
      __syncthreads();
    } else if constexpr (V == 1) {
      if (tid == 0) {
        mbar_expect_tx(&bar, (unsigned)bytes);
        bulk_g2s(sm, s, (unsigned)bytes, &bar);
      }
      mbar_wait(&bar, it & 1);
    } else {
      const int piece = bytes / 4;
      if (tid < 4) {
        mbar_expect_tx(&bar, (unsigned)piece);
        bulk_g2s(sm + tid * piece, reinterpret_cast<const unsigned char*>(s) + tid * piece,
                 (unsigned)piece, &bar);
      }
      mbar_wait(&bar, it & 1);
    }
    acc += reinterpret_cast<const float*>(sm)[(tid * 4 + it) % (bytes / 4)];
    __syncthreads();  // everyone read before the next copy overwrites
  }
  if (acc == 12345.f) sink[0] = acc;  // keep the loads alive
}

void copy_bench(int64_t variant, int64_t grid, int64_t bytes, int64_t iters, torch::Tensor src,
                torch::Tensor sink) {
  auto st = at::cuda::getCurrentCUDAStream();
  auto* s = reinterpret_cast<const int4*>(src.data_ptr<at::Half>());
  auto* k = sink.data_ptr<float>();
  const size_t smem = (size_t)bytes;
#define LAUNCH_COPY(V)                                                                       \
  C10_CUDA_CHECK(cudaFuncSetAttribute(copy_kernel<V>,                                        \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); \
  copy_kernel<V><<<(unsigned)grid, kThreads, smem, st>>>(s, (int)bytes, (int)iters, k);
  if (variant == 0) { LAUNCH_COPY(0) } else if (variant == 1) { LAUNCH_COPY(1) } else { LAUNCH_COPY(2) }
#undef LAUNCH_COPY
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- atomics --------------------------------------------------------------------------------
template <int V>
__global__ void __launch_bounds__(kThreads) atomic_kernel(float* out, int iters) {
  constexpr int kPerBlock = 1152, kN = 2880;  // one unit's outputs; the layer's vector
  const int tid = threadIdx.x, base = (blockIdx.x * 96) % kN;
  for (int it = 0; it < iters; ++it) {
    if constexpr (V == 0) {
      for (int e = tid; e < kPerBlock; e += kThreads) atomicAdd(out + (base + e) % kN, 1.f);
    } else {
      for (int e = tid * 4; e < kPerBlock; e += kThreads * 4) {
        float* p = out + (base + e) % kN;
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(1.f),
                     "f"(1.f), "f"(1.f), "f"(1.f)
                     : "memory");
      }
    }
    __threadfence();  // as before a barrier: the adds must be visible
    __syncthreads();
  }
}

void atomic_bench(int64_t variant, int64_t grid, int64_t iters, torch::Tensor out) {
  auto st = at::cuda::getCurrentCUDAStream();
  if (variant == 0)
    atomic_kernel<0><<<(unsigned)grid, kThreads, 0, st>>>(out.data_ptr<float>(), (int)iters);
  else
    atomic_kernel<1><<<(unsigned)grid, kThreads, 0, st>>>(out.data_ptr<float>(), (int)iters);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t max_clusters(int64_t cluster) {  // resident clusters of barrier_kernel<5>
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)cluster * 132);
  cfg.blockDim = dim3(kThreads);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = (unsigned)cluster;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int n = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, barrier_kernel<5>, &cfg));
  return n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("barrier_bench", &barrier_bench);
  m.def("copy_bench", &copy_bench);
  m.def("atomic_bench", &atomic_bench);
  m.def("max_clusters", &max_clusters);
}
