// Weight-stationary stack experiment (kernel-design/WEIGHT_STATIONARY.md).
//
// L tensor-ring layers in ONE persistent kernel. Layers alternate
//   up   (8,12,20) -> (12,10,24)   1920 -> 2880
//   down (12,10,24) -> (8,12,20)   2880 -> 1920
// so the output of one layer is the input of the next (an MLP-like chain).
//
//   prologue   x (FP16) -> buf[0] (FP32), zero buf[1]                          barrier
//   layer l    zero buf[(l+2)%3]; every block takes work units (a, k-chunk, q-chunk)
//              of layer l: read buf[l%3], add into buf[(l+1)%3]                    barrier
//   epilogue   buf[L%3] -> y (FP16)
//
// What was measured on the H100 and is built in (results/h100/stack/, results/h100/floors*.json):
//  - TWO-PHASE input loads: every thread issues all its loads (cores slice, x slice) into
//    registers, then stores them to shared memory (one L2 round trip instead of 5-6).
//  - partial y leaves with red.global.add.v4.f32 / v2.f32 (sm_90 vector reductions).
//  - barrier variants (template): generation, cooperative-groups grid.sync, or MONOTONIC counter
//    (red.release to arrive, ld.acquire spin to wait; splits into arrive / wait for prefetch).
//  - Y accumulator tiles per warp and, for the tuned tilings, the tiling itself (k chunk, q
//    chunk, tokens) are COMPILE-TIME constants: no register spills, index math without division
//    by runtime values. -DSTACK_RUNTIME_TILING builds the runtime-tiling version for comparison.
//  - -DSTACK_V3 (t = 1): stages 2 and 3 on PTX mma.sync in registers. One warp per q:
//        stage 2  m16n8k16 over (j,b), accumulator 16 x 8c (R = 8) or two tiles (R = 16)
//        -> packed to FP16 IN REGISTERS: the m16n8 accumulator layout is exactly the A-operand
//           layout of m16n8k8 (R = 8) / of m16n8k16 built from two tiles (R = 16)
//        stage 3  m16n8k8 / m16n8k16 over c into Y (three n8 tiles of r), Y stays in registers
//        -> red.global.add.v2.f32 straight from the Y fragment.
//    No S2 buffer, no FP32 staging, no barrier between stages 2 and 3; stage 1 runs once for
//    all k of the unit, so a unit has 2 __syncthreads instead of 1 + 3 * kc. S1 padding rows
//    are not zeroed (an mma output row depends only on its own A row; those rows are dropped),
//    and for R = 8 the B fragments of the unit stay in registers across its k values.
//  - -DSTACK_ASYNC: the cores slice is copied with cp.async (16-byte async global -> shared,
//    zero-fill for padding), so issuing a prefetch costs a few instructions per thread and the
//    block goes straight to the barrier; A stays FP16 in shared memory.
//
// Modes                        work   prefetch  barrier
//   0  generation              yes    no        generation
//   1  barriers only           -      -         generation
//   2  barriers only           -      -         grid.sync
//   3  grid.sync               yes    no        grid.sync
//   4  prefetch, generation    yes    yes       generation
//   5  prefetch, monotonic     yes    yes       monotonic
//   6  monotonic               yes    no        monotonic
//   7  barriers only           -      -         monotonic
//   8  prefetch, grid.sync     yes    yes       grid.sync (prefetch issued before the barrier)

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
  // elements of one layer's packed cores (tr_kernel.pack_cores)
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

// Tiling values of one unit: compile-time constants when KC / QC / TT are non-zero.
template <class S, int KC, int QC, int TT>
struct TP {
  int kc, qc, nkc, nqc, tt, Mp, Nb, N2s;
  __host__ __device__ __forceinline__ TP(const Tiling& tl, int T)
      : kc(KC ? KC : tl.kc), qc(QC ? QC : tl.qc),
        nkc(KC ? (S::nk + KC - 1) / KC : tl.nkc), nqc(QC ? (S::Q + QC - 1) / QC : tl.nqc),
        tt(TT ? TT : T), Mp(TT ? round16(TT * S::P) : tl.Mp),
        Nb(QC ? QC * S::Rc : tl.Nb), N2s((QC ? QC * S::Rc : tl.Nb) + 8) {}
};

#ifdef STACK_V3
constexpr int kS1Bufs = 8;  // V3 keeps one S1 per k of the unit (kc <= 8)
#else
constexpr int kS1Bufs = 1;
#endif

#ifdef STACK_ASYNC
using AElem = __half;  // cp.async copies bytes: A stays FP16, stage 1 converts on read
#else
using AElem = float;
#endif

// Shared memory = [core slot 0][core slot 1 (prefetch modes only)][work area]
struct CoreLayout { size_t b, a, c, total; };
struct WorkLayout { size_t x, s1, s2, stage, total; };

template <class S>
__host__ __device__ inline CoreLayout core_layout_k(int kc, int N2s) {
  CoreLayout s;
  size_t o = 0;
  s.b = o; o = align128(o + sizeof(__half) * S::K2p * N2s);
  s.a = o; o = align128(o + sizeof(AElem)  * S::ni * S::PR);
  s.c = o; o = align128(o + sizeof(__half) * kc * S::Rc * S::Rrs);
  s.total = o;
  return s;
}

template <class S>
__host__ __device__ inline WorkLayout work_layout_k(int kc, int T, int Mp, int Nb) {
  WorkLayout s;
  size_t o = 0;
  s.x  = o; o = align128(o + sizeof(float)  * kc * T * S::ni * S::nj);
  s.s1 = o; o = align128(o + sizeof(__half) * Mp * S::K2s * (kS1Bufs > 1 ? kc : 1));
  s.s2 = o; o = align128(o + sizeof(__half) * Mp * Nb);
  s.stage = o; o = align128(o + sizeof(float) * kWarps * 16 * kStageLd);
  s.total = o;
  return s;
}

template <class S>
__host__ __device__ inline CoreLayout core_layout(const Tiling& t) {
  return core_layout_k<S>(t.kc, t.N2s);
}
template <class S>
__host__ __device__ inline WorkLayout work_layout(const Tiling& t, int T) {
  return work_layout_k<S>(t.kc, T, t.Mp, t.Nb);
}

// Ablation flags (timing only: results are wrong when any is set).
constexpr int kNoX = 1, kNoS1 = 2, kNoS2 = 4, kNoS3 = 8, kNoAtomics = 16, kNoCores = 32;
constexpr int kNoUnit = 64;  // skip the units entirely: barrier + loop skeleton only

// STACK_TIMING: thread 0 of every block writes timestamps for every layer into
// tbuf[layer][block][kTSlots]:
//   0 globaltimer at layer start   1 globaltimer at arrive   2 globaltimer at release
//   3 clock64 layer start   4 unit entry   5 after inputs landed (sync)   6 after Y init
//   7 sum of stage 1        8 sum of stage 2   9 sum of stage 3 (cycles; V3: 8 = stages 2+3)
//   10 after the k loop     11 after the reductions (+ sync)   12 before arrive
//   13 after prefetch       14 after release
constexpr int kTSlots = 16;
#ifdef STACK_TIMING
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define TS(tb, i) do { if ((tb) != nullptr && threadIdx.x == 0) (tb)[i] = (unsigned long long)clock64(); } while (0)
#define TG(tb, i) do { if ((tb) != nullptr && threadIdx.x == 0) (tb)[i] = gtimer(); } while (0)
#define TSTART(v) long long v = clock64()
#define TACC(tb, i, v) do { if ((tb) != nullptr && threadIdx.x == 0) { const long long n_ = clock64(); (tb)[i] += (unsigned long long)(n_ - (v)); (v) = n_; } } while (0)
#else
#define TS(tb, i) do { } while (0)
#define TG(tb, i) do { } while (0)
#define TSTART(v) do { } while (0)
#define TACC(tb, i, v) do { } while (0)
#endif

using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ void red_add_v4(float* p, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(v.x), "f"(v.y),
               "f"(v.z), "f"(v.w)
               : "memory");
}
__device__ __forceinline__ void red_add_v2(float* p, float a, float b) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(p), "f"(a), "f"(b) : "memory");
}

// ---- PTX mma helpers (V3) -------------------------------------------------------------------
// m16n8k16, row.col, FP16 inputs, FP32 accumulator in place.
//   A (16x16): a0 (g, 2t..2t+1)  a1 (g+8, 2t..)  a2 (g, 2t+8..)  a3 (g+8, 2t+8..)
//   B (16x8):  b0 (k = 2t..2t+1, n = g)          b1 (k = 2t+8.., n = g)
//   C (16x8):  c0,c1 (g, 2t..2t+1)               c2,c3 (g+8, 2t..2t+1)
//   with g = lane / 4, t = lane % 4.
__device__ __forceinline__ void mma16816(float (&d)[4], unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// m16n8k8: A (16x8): a0 (g, 2t..2t+1)  a1 (g+8, 2t..);  B (8x8): b0 (k = 2t..2t+1, n = g)
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
// 16-byte asynchronous copy global -> shared; src_bytes = 0 fills the 16 bytes with zeros
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(
                   (unsigned)__cvta_generic_to_shared(dst)),
               "l"(src), "r"(src_bytes)
               : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}
// four consecutive A values as floats (A is FP32, or FP16 with STACK_ASYNC)
__device__ __forceinline__ float4 load_a4(const AElem* p) {
#ifdef STACK_ASYNC
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
  const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
  return make_float4(lo.x, lo.y, hi.x, hi.y);
#else
  return *reinterpret_cast<const float4*>(p);
#endif
}

// V3 is used for the compiled t = 1 tilings (same condition in load_inputs and compute_any)
template <int KC, int QC, int TT>
__host__ __device__ constexpr bool uses_v3() {
#ifdef STACK_V3
  return TT == 1 && KC != 0 && QC != 0 && QC <= 8 && KC <= 8;
#else
  return false;
#endif
}

// ---- inputs of one unit: cores slice and/or x slice, two-phase -----------------------------
constexpr int kLB = 8, kLX = 4;

template <class S, bool kCores, bool kX, int KC, int QC, int TT>
__device__ __forceinline__ void load_inputs(const Tiling& tl, const int T, const int u,
                                            const __half* __restrict__ A1,
                                            const __half* __restrict__ B2,
                                            const __half* __restrict__ C3,
                                            const float* __restrict__ xin,
                                            unsigned char* cs, unsigned char* ws,
                                            const int flags) {
  constexpr int ni = S::ni, nj = S::nj, nk = S::nk, Q = S::Q, R = S::R;
  constexpr int Rc = S::Rc, Rrp = S::Rrp, K2p = S::K2p, N2c = S::N2c, K2s = S::K2s;
  constexpr int Rrs = S::Rrs, PR = S::PR, in_f = S::in_f;
  constexpr int crow = Rrp / 8, cstride = Rrs / 8, cvec = Rc * crow;  // C, int4 units
  constexpr int nA = ni * PR / 8;                                      // A, int4 of 8 halves
  const TP<S, KC, QC, TT> p(tl, T);
  const int tid = threadIdx.x;
  const int per_a = p.nkc * p.nqc;
  const int a = u / per_a, rem = u % per_a;
  const int k0 = (rem % p.nkc) * p.kc, q0 = (rem / p.nkc) * p.qc;
  const int qc_valid = min(p.qc, Q - q0);
  const int kc_valid = min(p.kc, nk - k0);
  const bool cores = kCores && !(flags & kNoCores);
  const bool xon = kX && !(flags & kNoX);

  const CoreLayout CL = core_layout_k<S>(p.kc, p.N2s);
  const WorkLayout WL = work_layout_k<S>(p.kc, p.tt, p.Mp, p.Nb);
  int4* sB = reinterpret_cast<int4*>(cs + CL.b);
  AElem* sA = reinterpret_cast<AElem*>(cs + CL.a);
  int4* sC = reinterpret_cast<int4*>(cs + CL.c);
  float* sX = reinterpret_cast<float*>(ws + WL.x);
  const int4* gB = reinterpret_cast<const int4*>(B2);
  const int4* gC = reinterpret_cast<const int4*>(C3);
  const int4* gA = reinterpret_cast<const int4*>(A1 + (size_t)a * ni * PR);
  const int brow = p.Nb / 8, bstride = p.N2s / 8;
  constexpr int grow = N2c / 8;
  const int valid8 = qc_valid * Rc / 8, col0 = q0 * Rc / 8;
#ifdef STACK_ASYNC
  if (cores) {  // cores slice: cp.async, no registers, zero-fill for padding
    for (int e = tid; e < K2p * brow; e += kThreads) {
      const int row = e / brow, col = e % brow;
      const bool ok = col < valid8;
      cp_async16(sB + row * bstride + col, gB + row * grow + col0 + (ok ? col : 0), ok ? 16 : 0);
    }
    for (int e = tid; e < p.kc * cvec; e += kThreads) {
      const int kk = e / cvec, c = (e % cvec) / crow, col = e % crow;
      const bool ok = kk < kc_valid;
      cp_async16(sC + (kk * Rc + c) * cstride + col,
                 gC + ((size_t)(k0 + (ok ? kk : 0)) * R + a) * cvec + e % cvec, ok ? 16 : 0);
    }
    for (int e = tid; e < nA; e += kThreads)
      cp_async16(reinterpret_cast<int4*>(sA) + e, gA + e, 16);
    cp_async_commit();
  }
  const bool cores_sync = false;  // the register path below only handles x
#else
  const bool cores_sync = cores;
#endif
  const int nB = cores_sync ? K2p * brow : 0, nC = cores_sync ? p.kc * cvec : 0;
  const int nX = xon ? p.kc * p.tt * ni * nj : 0;
  const int4 zero = make_int4(0, 0, 0, 0);

  for (int r = 0;; ++r) {
    int4 rb[kLB], rc[2], ra = zero;
    float rx[kLX];
    // ---- phase 1: issue every load of this round
#pragma unroll
    for (int j = 0; j < kLB; ++j) {
      const int e = tid + (r * kLB + j) * kThreads;
      rb[j] = zero;
      if (e < nB) {
        const int row = e / brow, col = e % brow;
        if (col < valid8) rb[j] = gB[row * grow + col0 + col];
      }
    }
#pragma unroll
    for (int j = 0; j < kLX; ++j) {
      const int e = tid + (r * kLX + j) * kThreads;
      rx[j] = 0.f;
      if (e < nX) {
        const int jj = e % nj;
        int rest = e / nj;
        const int i = rest % ni;  rest /= ni;
        const int t = rest % p.tt;
        const int kk = rest / p.tt;
        if (kk < kc_valid) rx[j] = __ldcg(xin + (size_t)t * in_f + (i * nj + jj) * nk + (k0 + kk));
      }
    }
    if (r == 0 && cores_sync) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int e = tid + j * kThreads;
        rc[j] = zero;
        if (e < nC) {
          const int kk = e / cvec;
          if (kk < kc_valid) rc[j] = gC[((size_t)(k0 + kk) * R + a) * cvec + e % cvec];
        }
      }
      if (tid < nA) ra = gA[tid];
    }
    // ---- phase 2: store (and convert A to FP32)
#pragma unroll
    for (int j = 0; j < kLB; ++j) {
      const int e = tid + (r * kLB + j) * kThreads;
      if (e < nB) sB[(e / brow) * bstride + e % brow] = rb[j];
    }
#pragma unroll
    for (int j = 0; j < kLX; ++j) {
      const int e = tid + (r * kLX + j) * kThreads;
      if (e < nX) sX[e] = rx[j];
    }
    if (r == 0 && cores_sync) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int e = tid + j * kThreads;
        if (e < nC) {
          const int kk = e / cvec, c = (e % cvec) / crow, col = e % crow;
          sC[(kk * Rc + c) * cstride + col] = rc[j];
        }
      }
      if (tid < nA) {
#ifdef STACK_ASYNC
        reinterpret_cast<int4*>(sA)[tid] = ra;  // not reached: the async path copies A itself
#else
        const __half2* h = reinterpret_cast<const __half2*>(&ra);
        float4* d = reinterpret_cast<float4*>(sA + tid * 8);
        const float2 f0 = __half22float2(h[0]), f1 = __half22float2(h[1]);
        const float2 f2 = __half22float2(h[2]), f3 = __half22float2(h[3]);
        d[0] = make_float4(f0.x, f0.y, f1.x, f1.y);
        d[1] = make_float4(f2.x, f2.y, f3.x, f3.y);
#endif
      }
    }
    if ((r + 1) * kLB * kThreads >= nB && (r + 1) * kLX * kThreads >= nX) break;
  }
  if (kX && !uses_v3<KC, QC, TT>()) {  // S1 padding (rows >= tt*P, columns >= nj*R) must be zero: stage 1 skips it
    __half* sS1 = reinterpret_cast<__half*>(ws + WL.s1);
    const int n = p.Mp * K2s * (kS1Bufs > 1 ? p.kc : 1) / 2;
    for (int e = tid; e < n; e += kThreads)
      reinterpret_cast<__half2*>(sS1)[e] = __floats2half2_rn(0.f, 0.f);
  }
}

// ---- stage 1 for one k: S1[(t,p)][j*R+b] = sum_i x[t,i,j,k] * A[a,p,i,b] -------------------
template <class S>
__device__ __forceinline__ void stage1(const float* xk, const AElem* sA, __half* sS1, int M,
                                       int tid) {
  constexpr int ni = S::ni, nj = S::nj, P = S::P, R = S::R, K2s = S::K2s, PR = S::PR;
  if constexpr (nj % 4 == 0 && R % 4 == 0) {  // up layers: 4 j x 4 b per thread
    constexpr int J4 = nj / 4, B4 = R / 4;
    for (int e = tid; e < M * J4 * B4; e += kThreads) {
      const int b4 = e % B4, j4 = (e / B4) % J4, row = e / (B4 * J4);
      const int t = row / P, p = row % P;
      float acc[4][4] = {};
#pragma unroll
      for (int i = 0; i < ni; ++i) {
        const float4 xv = *reinterpret_cast<const float4*>(xk + (t * ni + i) * nj + j4 * 4);
        const float4 av = load_a4(sA + i * PR + p * R + b4 * 4);
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
  } else {  // down layers (nj = 10): one j x 4 b per thread
    constexpr int B4 = R / 4;
    for (int e = tid; e < M * nj * B4; e += kThreads) {
      const int b4 = e % B4, j = (e / B4) % nj, row = e / (B4 * nj);
      const int t = row / P, p = row % P;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < ni; ++i) {
        const float xv = xk[(t * ni + i) * nj + j];
        const float4 av = load_a4(sA + i * PR + p * R + b4 * 4);
        acc[0] += xv * av.x; acc[1] += xv * av.y; acc[2] += xv * av.z; acc[3] += xv * av.w;
      }
      __half2* dst = reinterpret_cast<__half2*>(sS1 + row * K2s + j * R + b4 * 4);
      dst[0] = __floats2half2_rn(acc[0], acc[1]);
      dst[1] = __floats2half2_rn(acc[2], acc[3]);
    }
  }
}

// ---- V2 compute path (WMMA, S2 through shared memory) ---------------------------------------
template <class S, int YT, int KC, int QC, int TT>
__device__ __forceinline__ void compute_unit(const Tiling& tl, const int T, const int u,
                                             float* __restrict__ yout,
                                             const unsigned char* cs, unsigned char* ws,
                                             const int flags, unsigned long long* tb) {
  constexpr int ni = S::ni, nj = S::nj, nk = S::nk, P = S::P, Q = S::Q, Rr = S::Rr;
  constexpr int Rc = S::Rc, Rrp = S::Rrp, K2p = S::K2p, K2s = S::K2s;
  constexpr int Rrs = S::Rrs, out_f = S::out_f;
  const TP<S, KC, QC, TT> p(tl, T);
  const CoreLayout CL = core_layout_k<S>(p.kc, p.N2s);
  const WorkLayout WL = work_layout_k<S>(p.kc, p.tt, p.Mp, p.Nb);
  const __half* sB = reinterpret_cast<const __half*>(cs + CL.b);
  const AElem*  sA = reinterpret_cast<const AElem*>(cs + CL.a);
  const __half* sC = reinterpret_cast<const __half*>(cs + CL.c);
  const float*  sX = reinterpret_cast<const float*>(ws + WL.x);
  __half* sS1 = reinterpret_cast<__half*>(ws + WL.s1);
  __half* sS2 = reinterpret_cast<__half*>(ws + WL.s2);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  float* stage = reinterpret_cast<float*>(ws + WL.stage) + warp * 16 * kStageLd;
  const int per_a = p.nkc * p.nqc;
  const int rem = u % per_a;
  const int k0 = (rem % p.nkc) * p.kc, q0 = (rem / p.nkc) * p.qc;
  const int qc_valid = min(p.qc, Q - q0);
  const int kc_valid = min(p.kc, nk - k0);
  const int Mp = p.Mp, Nb = p.Nb, N2s = p.N2s, qc = p.qc, tt = p.tt;

#ifdef STACK_ASYNC
  cp_async_wait_all();  // this thread's cp.async copies (cores) have landed
#endif
  __syncthreads();  // the unit's inputs are in shared memory
  TS(tb, 5);
  constexpr int y_nt_n = Rrp / 16;
  const int y_tiles = (Mp * qc / 16) * y_nt_n;
  FragC yacc[YT];
#pragma unroll
  for (int i = 0; i < YT; ++i) wmma::fill_fragment(yacc[i], 0.f);
  TS(tb, 6);
  TSTART(tk);

  for (int kk = 0; kk < kc_valid; ++kk) {
    if (!(flags & kNoS1)) stage1<S>(sX + (size_t)kk * tt * ni * nj, sA, sS1, tt * P, tid);
    __syncthreads();
    TACC(tb, 7, tk);

    // stage 2: S2 = S1 @ B2 (block's q chunk), Tensor Cores. (Split-K over warps was
    // measured slower: the extra barrier and partial sums cost more than the 3 saved mma.)
    if (!(flags & kNoS2)) {
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
    TACC(tb, 8, tk);

    // stage 3: Y[(t,p,q), r] += S2[(t,p,q), c] @ C[c, r], Tensor Cores, Y in registers
    if (!(flags & kNoS3)) {
      const __half* ck = sC + kk * Rc * Rrs;
#pragma unroll
      for (int i = 0; i < YT; ++i) {
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
    TACC(tb, 9, tk);
  }
  TS(tb, 10);

  // 4 consecutive r per red.global.add.v4.f32 (Rr = 24 or 20: a group of 4 is all valid or
  // all padding)
#pragma unroll
  for (int i = 0; i < YT; ++i) {
    const int tile = warp + i * kWarps;
    if (tile < y_tiles) {
      const int mt = tile / y_nt_n, nt = tile % y_nt_n;
      wmma::store_matrix_sync(stage, yacc[i], kStageLd, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 64; e += 32) {
        const int row = e / 4, c4 = e % 4;
        const int m = mt * 16 + row, r = nt * 16 + c4 * 4;
        const int ql = m % qc, tp = m / qc, t = tp / P, pp = tp % P;
        if (r < Rr && t < tt && ql < qc_valid && !(flags & kNoAtomics))
          red_add_v4(yout + (size_t)t * out_f + (pp * Q + q0 + ql) * Rr + r,
                     *reinterpret_cast<const float4*>(stage + row * kStageLd + c4 * 4));
      }
      __syncwarp();
    }
  }
#ifdef STACK_TIMING
  __syncthreads();
#endif
  TS(tb, 11);
}

// ---- V3 compute path (t = 1): stages 2 -> 3 in registers on PTX mma.sync ------------------
template <class S, int KC, int QC>
__device__ __forceinline__ void compute_unit_v3(const Tiling& tl, const int u,
                                                float* __restrict__ yout,
                                                const unsigned char* cs, unsigned char* ws,
                                                const int flags, unsigned long long* tb) {
  constexpr int ni = S::ni, nj = S::nj, nk = S::nk, P = S::P, Q = S::Q, Rr = S::Rr, R = S::R;
  constexpr int Rc = S::Rc, K2p = S::K2p, K2s = S::K2s, Rrs = S::Rrs, out_f = S::out_f;
  constexpr int kt_n = K2p / 16;      // k16 steps of stage 2
  constexpr int NC8 = R / 8;          // n8 tiles of c per q (1 for R = 8, 2 for R = 16)
  constexpr int RT = (Rr + 7) / 8;    // n8 tiles of r (3)
  static_assert(R == 8 || R == 16, "V3 handles R = 8 and 16");
  const TP<S, KC, QC, 1> p(tl, 1);
  const CoreLayout CL = core_layout_k<S>(p.kc, p.N2s);
  const WorkLayout WL = work_layout_k<S>(p.kc, 1, p.Mp, p.Nb);
  const __half* sB = reinterpret_cast<const __half*>(cs + CL.b);
  const AElem*  sA = reinterpret_cast<const AElem*>(cs + CL.a);
  const __half* sC = reinterpret_cast<const __half*>(cs + CL.c);
  const float*  sX = reinterpret_cast<const float*>(ws + WL.x);
  __half* sS1 = reinterpret_cast<__half*>(ws + WL.s1);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int g = lane >> 2, t4 = lane & 3;
  const int per_a = p.nkc * p.nqc;
  const int rem = u % per_a;
  const int k0 = (rem % p.nkc) * p.kc, q0 = (rem / p.nkc) * p.qc;
  const int qc_valid = min(p.qc, Q - q0);
  const int kc_valid = min(p.kc, nk - k0);
  constexpr int S1sz = 16 * K2s;      // one S1 buffer (Mp = 16 at t = 1)

#ifdef STACK_ASYNC
  cp_async_wait_all();  // this thread's cp.async copies (cores) have landed
#endif
  __syncthreads();  // the unit's inputs are in shared memory
  TS(tb, 5);
  TSTART(tk);
  // stage 1 for every k of the unit, one S1 buffer each, then a single barrier
  if (!(flags & kNoS1))
    for (int kk = 0; kk < kc_valid; ++kk)
      stage1<S>(sX + (size_t)kk * ni * nj, sA, sS1 + kk * S1sz, P, tid);
  __syncthreads();
  TACC(tb, 7, tk);
  TS(tb, 6);

  const int ql = warp;  // one warp per q of the chunk
  if (ql < qc_valid) {
    float y[RT][4];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt) y[rt][0] = y[rt][1] = y[rt][2] = y[rt][3] = 0.f;
    const int colq = ql * Rc;  // this q's first column in the block's B2 slice
    // R = 8: the unit's B fragments (kt_n x 2 registers) do not depend on k: load them once
    constexpr bool kHoistB = (R == 8);
    unsigned bh[kHoistB ? kt_n : 1][2];
    if constexpr (kHoistB) {
#pragma unroll
      for (int kt = 0; kt < kt_n; ++kt) {
        const int kb = kt * 16 + 2 * t4, n = colq + g;
        bh[kt][0] = pack_h2(sB[kb * p.N2s + n], sB[(kb + 1) * p.N2s + n]);
        bh[kt][1] = pack_h2(sB[(kb + 8) * p.N2s + n], sB[(kb + 9) * p.N2s + n]);
      }
    }

    for (int kk = 0; kk < kc_valid; ++kk) {
      const __half* s1 = sS1 + kk * S1sz;
      // ---- stage 2: acc[nc] (16 rows (t,p) x 8 c) = S1 (16 x K2p) @ B2[:, q, c-tile]
      float acc[NC8][4];
#pragma unroll
      for (int nc = 0; nc < NC8; ++nc) acc[nc][0] = acc[nc][1] = acc[nc][2] = acc[nc][3] = 0.f;
      if (!(flags & kNoS2)) {
#pragma unroll
        for (int kt = 0; kt < kt_n; ++kt) {
          const int kb = kt * 16 + 2 * t4;
          const unsigned a0 = lds_u32(s1 + g * K2s + kb);
          const unsigned a1 = lds_u32(s1 + (g + 8) * K2s + kb);
          const unsigned a2 = lds_u32(s1 + g * K2s + kb + 8);
          const unsigned a3 = lds_u32(s1 + (g + 8) * K2s + kb + 8);
#pragma unroll
          for (int nc = 0; nc < NC8; ++nc) {
            if constexpr (kHoistB) {
              mma16816(acc[nc], a0, a1, a2, a3, bh[kt][0], bh[kt][1]);
            } else {
              const int n = colq + nc * 8 + g;
              const unsigned b0 = pack_h2(sB[kb * p.N2s + n], sB[(kb + 1) * p.N2s + n]);
              const unsigned b1 = pack_h2(sB[(kb + 8) * p.N2s + n], sB[(kb + 9) * p.N2s + n]);
              mma16816(acc[nc], a0, a1, a2, a3, b0, b1);
            }
          }
        }
      }
      // ---- accumulator -> FP16 A operand of stage 3, in registers
      const __half* ck = sC + kk * Rc * Rrs;
      if (!(flags & kNoS3)) {
        if constexpr (R == 8) {
          const unsigned a0 = pack_f2(acc[0][0], acc[0][1]);  // (g,   c = 2t, 2t+1)
          const unsigned a1 = pack_f2(acc[0][2], acc[0][3]);  // (g+8, c = 2t, 2t+1)
#pragma unroll
          for (int rt = 0; rt < RT; ++rt) {
            const int r = rt * 8 + g;
            const unsigned b0 = pack_h2(ck[(2 * t4) * Rrs + r], ck[(2 * t4 + 1) * Rrs + r]);
            mma1688(y[rt], a0, a1, b0);
          }
        } else {
          const unsigned a0 = pack_f2(acc[0][0], acc[0][1]);  // (g,   c = 2t..)
          const unsigned a1 = pack_f2(acc[0][2], acc[0][3]);  // (g+8, c = 2t..)
          const unsigned a2 = pack_f2(acc[1][0], acc[1][1]);  // (g,   c = 8+2t..)
          const unsigned a3 = pack_f2(acc[1][2], acc[1][3]);  // (g+8, c = 8+2t..)
#pragma unroll
          for (int rt = 0; rt < RT; ++rt) {
            const int r = rt * 8 + g;
            const unsigned b0 = pack_h2(ck[(2 * t4) * Rrs + r], ck[(2 * t4 + 1) * Rrs + r]);
            const unsigned b1 = pack_h2(ck[(2 * t4 + 8) * Rrs + r], ck[(2 * t4 + 9) * Rrs + r]);
            mma16816(y[rt], a0, a1, a2, a3, b0, b1);
          }
        }
      }
    }
    TACC(tb, 8, tk);
    TS(tb, 10);
    // ---- Y rows (g, g+8) = (t=0, p), columns r = 8 rt + 2t, +1 -> red.global.add.v2.f32
    if (!(flags & kNoAtomics)) {
      const int q = q0 + ql;
#pragma unroll
      for (int rt = 0; rt < RT; ++rt) {
        const int r = rt * 8 + 2 * t4;
        if (r < Rr) {
          if (g < P) red_add_v2(yout + (g * Q + q) * Rr + r, y[rt][0], y[rt][1]);
          if (g + 8 < P) red_add_v2(yout + ((g + 8) * Q + q) * Rr + r, y[rt][2], y[rt][3]);
        }
      }
    }
  }
#ifdef STACK_TIMING
  __syncthreads();
#endif
  TS(tb, 11);
  (void)out_f;
}

// ---- non-inlined entry points: one call per unit keeps register pressure bounded ----------
template <class S, int YT, int KC, int QC, int TT>
__device__ __forceinline__ void compute_any(const Tiling& tl, const int T, const int u,
                                            float* yout, unsigned char* cs, unsigned char* ws,
                                            const int flags, unsigned long long* tb) {
#ifdef STACK_V3
  if constexpr (uses_v3<KC, QC, TT>()) {
    compute_unit_v3<S, KC, QC>(tl, u, yout, cs, ws, flags, tb);
    return;
  }
#endif
  compute_unit<S, YT, KC, QC, TT>(tl, T, u, yout, cs, ws, flags, tb);
}

template <class S, int YT, int KC, int QC, int TT>
__device__ __noinline__ void run_unit(const Tiling& tl, const int T, const int u,
                                      const float* xin, float* yout, const __half* A1,
                                      const __half* B2, const __half* C3, unsigned char* cs,
                                      unsigned char* ws, const int flags,
                                      unsigned long long* tb) {
  TS(tb, 4);
  __syncthreads();  // the previous unit of this block is done with shared memory
  load_inputs<S, true, true, KC, QC, TT>(tl, T, u, A1, B2, C3, xin, cs, ws, flags);
  compute_any<S, YT, KC, QC, TT>(tl, T, u, yout, cs, ws, flags, tb);
}

template <class S, int YT, int KC, int QC, int TT>
__device__ __noinline__ void run_prefetched(const Tiling& tl, const int T, const int u,
                                            const float* xin, float* yout, unsigned char* cs,
                                            unsigned char* ws, const int flags,
                                            unsigned long long* tb) {
  TS(tb, 4);  // the work area is free: the previous layer ended with a barrier
  load_inputs<S, false, true, KC, QC, TT>(tl, T, u, nullptr, nullptr, nullptr, xin, cs, ws,
                                          flags);
  compute_any<S, YT, KC, QC, TT>(tl, T, u, yout, cs, ws, flags, tb);
}

template <class S, int KC, int QC, int TT>
__device__ __noinline__ void prefetch_cores(const Tiling& tl, const int T, const int u,
                                            const __half* A1, const __half* B2,
                                            const __half* C3, unsigned char* cs,
                                            const int flags) {
  load_inputs<S, true, false, KC, QC, TT>(tl, T, u, A1, B2, C3, nullptr, cs, nullptr, flags);
}

// ---- barriers -------------------------------------------------------------------------------
__device__ __forceinline__ unsigned int gen_arrive(unsigned int* sync, unsigned int nblocks) {
  __threadfence();
  __syncthreads();
  unsigned int g = 0;
  if (threadIdx.x == 0) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> count(sync[0]), gen(sync[1]);
    g = gen.load(cuda::memory_order_relaxed);
    if (count.fetch_add(1u, cuda::memory_order_acq_rel) == nblocks - 1) {
      count.store(0u, cuda::memory_order_relaxed);
      gen.fetch_add(1u, cuda::memory_order_release);
    }
  }
  return g;
}

__device__ __forceinline__ void gen_wait(unsigned int* sync, unsigned int g) {
  if (threadIdx.x == 0) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> gen(sync[1]);
    while (gen.load(cuda::memory_order_acquire) == g) {
    }
  }
  __syncthreads();
}

// Monotonic barrier: sync[2] only ever grows. Barrier b of this launch is passed when it
// reaches base + nblocks * (b + 1); arriving is a single fire-and-forget red.release.
__device__ __forceinline__ void mono_arrive(unsigned int* sync) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(sync + 2), "r"(1u) : "memory");
  }
}

__device__ __forceinline__ void mono_wait(const unsigned int* sync, unsigned int target) {
  if (threadIdx.x == 0) {
    unsigned int v;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(sync + 2) : "memory");
    } while ((int)(v - target) < 0);
  }
  __syncthreads();
}

struct StackArgs {
  const __half* x;
  __half* y;
  float* buf;             // 3 activation buffers of bufstride floats each
  unsigned int* sync;     // [count, generation, monotonic counter]
  const __half* A1[2];    // [up, down], each [layers of that type][...]
  const __half* B2[2];
  const __half* C3[2];
  Tiling tl[2];
  int L, T, bufstride;
  int core_slot, work_off;  // bytes: one core slot; start of the work area
  int flags;                // ablation flags (kNo*)
  unsigned int bar_base;    // monotonic barrier: counter value at launch (caller tracks it)
  unsigned long long* tbuf;  // STACK_TIMING timeline, or nullptr
};

template <int kMode> struct ModeTraits {
  static constexpr bool work = kMode == 0 || kMode == 3 || kMode == 4 || kMode == 5 ||
                               kMode == 6 || kMode == 8;
  static constexpr bool prefetch = kMode == 4 || kMode == 5 || kMode == 8;
  static constexpr int bar = (kMode == 2 || kMode == 3 || kMode == 8) ? 1
                             : (kMode >= 5 ? 2 : 0);  // gen / cg / mono
};

template <int kMode, class F>
__device__ __forceinline__ void barrier(const StackArgs& g, int b, F between) {
  constexpr int kBar = ModeTraits<kMode>::bar;
  if constexpr (kBar == 0) {
    const unsigned int gen = gen_arrive(g.sync, gridDim.x);
    between();
    gen_wait(g.sync, gen);
  } else if constexpr (kBar == 1) {
    between();
    cg::this_grid().sync();
  } else {
    mono_arrive(g.sync);
    between();
    mono_wait(g.sync, g.bar_base + gridDim.x * (unsigned)(b + 1));
  }
}

// ---- per-unit dispatch: compiled tilings for t = 1, runtime tiling otherwise ---------------
template <class S, int KC, int QC, int TT>
__device__ __forceinline__ void unit_fixed(const StackArgs& g, const Tiling& tl, int u,
                                           const float* in, float* out, const __half* A1,
                                           const __half* B2, const __half* C3, unsigned char* cs,
                                           unsigned char* ws, bool prefetched,
                                           unsigned long long* tb) {
  constexpr int y_tiles = (round16(TT * S::P) * QC / 16) * (S::Rrp / 16);
  constexpr int YT = (y_tiles + kWarps - 1) / kWarps;
  if (prefetched) run_prefetched<S, YT, KC, QC, TT>(tl, g.T, u, in, out, cs, ws, g.flags, tb);
  else            run_unit<S, YT, KC, QC, TT>(tl, g.T, u, in, out, A1, B2, C3, cs, ws, g.flags, tb);
}

template <class S>
__device__ __forceinline__ void unit_runtime(const StackArgs& g, const Tiling& tl, int u,
                                             const float* in, float* out, const __half* A1,
                                             const __half* B2, const __half* C3,
                                             unsigned char* cs, unsigned char* ws,
                                             bool prefetched, unsigned long long* tb) {
  const int y_tiles = (tl.Mp * tl.qc / 16) * (S::Rrp / 16);
  const int ytw = (y_tiles + kWarps - 1) / kWarps;
  if (ytw <= 1) {
    if (prefetched) run_prefetched<S, 1, 0, 0, 0>(tl, g.T, u, in, out, cs, ws, g.flags, tb);
    else            run_unit<S, 1, 0, 0, 0>(tl, g.T, u, in, out, A1, B2, C3, cs, ws, g.flags, tb);
  } else if (ytw <= 2) {
    if (prefetched) run_prefetched<S, 2, 0, 0, 0>(tl, g.T, u, in, out, cs, ws, g.flags, tb);
    else            run_unit<S, 2, 0, 0, 0>(tl, g.T, u, in, out, A1, B2, C3, cs, ws, g.flags, tb);
  } else {
    if (prefetched) run_prefetched<S, kMaxYTiles, 0, 0, 0>(tl, g.T, u, in, out, cs, ws, g.flags, tb);
    else run_unit<S, kMaxYTiles, 0, 0, 0>(tl, g.T, u, in, out, A1, B2, C3, cs, ws, g.flags, tb);
  }
}

#ifdef STACK_RUNTIME_TILING
#define STACK_TRY(KCv, QCv)
#else
#define STACK_TRY(KCv, QCv)                                                                   \
  if (tl.kc == KCv && tl.qc == QCv) {                                                         \
    unit_fixed<S, KCv, QCv, 1>(g, tl, u, in, out, A1, B2, C3, cs, ws, have, tb);              \
    done = true;                                                                              \
  } else
#endif

// All units of one layer that belong to this block. With `first_prefetched`, the cores of
// the block's first unit are already in `cs`; every other unit loads its own.
template <class S>
__device__ __forceinline__ void layer_units(const StackArgs& g, int type, int idx,
                                            const float* in, float* out, unsigned char* cs,
                                            unsigned char* ws, bool first_prefetched,
                                            unsigned long long* tb) {
  if (g.flags & kNoUnit) return;
  const Tiling tl = g.tl[type];
  const __half* A1 = g.A1[type] + idx * S::a1_size;
  const __half* B2 = g.B2[type] + idx * S::b2_size;
  const __half* C3 = g.C3[type] + idx * S::c3_size;
  bool have = first_prefetched;
  for (int u = blockIdx.x; u < tl.units; u += gridDim.x) {  // the timeline records unit 1
    bool done = false;
    if (g.T == 1) {
      if constexpr (S::nk == 20) {  // up layers
        STACK_TRY(2, 4) STACK_TRY(2, 5) STACK_TRY(4, 4) STACK_TRY(4, 5) {}
      } else {                      // down layers
        STACK_TRY(2, 6) STACK_TRY(3, 6) STACK_TRY(2, 4) {}
      }
    }
    if (!done) unit_runtime<S>(g, tl, u, in, out, A1, B2, C3, cs, ws, have, tb);
    have = false;
    tb = nullptr;
  }
}

#ifdef STACK_RUNTIME_TILING
#define STACK_TRY_PF(KCv, QCv)
#else
#define STACK_TRY_PF(KCv, QCv)                                                                \
  if (tl.kc == KCv && tl.qc == QCv) {                                                         \
    prefetch_cores<S, KCv, QCv, 1>(tl, g.T, blockIdx.x, A1, B2, C3, cs, g.flags);             \
    return;                                                                                   \
  }
#endif

template <class S>
__device__ __forceinline__ void prefetch_type(const StackArgs& g, int type, int idx,
                                              unsigned char* cs) {
  const Tiling tl = g.tl[type];
  if ((int)blockIdx.x >= tl.units || (g.flags & kNoUnit)) return;
  const __half* A1 = g.A1[type] + idx * S::a1_size;
  const __half* B2 = g.B2[type] + idx * S::b2_size;
  const __half* C3 = g.C3[type] + idx * S::c3_size;
  if (g.T == 1) {
    if constexpr (S::nk == 20) {
      STACK_TRY_PF(2, 4) STACK_TRY_PF(2, 5) STACK_TRY_PF(4, 4) STACK_TRY_PF(4, 5)
    } else {
      STACK_TRY_PF(2, 6) STACK_TRY_PF(3, 6) STACK_TRY_PF(2, 4)
    }
  }
  prefetch_cores<S, 0, 0, 0>(tl, g.T, blockIdx.x, A1, B2, C3, cs, g.flags);
}

// Load the cores of this block's first unit of layer l into slot cs (if it has one).
template <int R>
__device__ __forceinline__ void prefetch_layer(const StackArgs& g, int l, unsigned char* cs) {
  if ((l & 1) == 0) prefetch_type<Up<R>>(g, 0, l >> 1, cs);
  else              prefetch_type<Down<R>>(g, 1, l >> 1, cs);
}

template <int R, int kMode>
__global__ void __launch_bounds__(kThreads, 2) tr_stack_kernel(const StackArgs g) {
  extern __shared__ __align__(128) unsigned char smem[];
  constexpr bool kWork = ModeTraits<kMode>::work;
  constexpr bool kPrefetch = ModeTraits<kMode>::prefetch;
  using U = Up<R>;
  using D = Down<R>;
  const int gtid = blockIdx.x * kThreads + threadIdx.x, gstride = gridDim.x * kThreads;
  const int T = g.T, BS = g.bufstride;
  unsigned char* slot[2] = {smem, smem + g.core_slot};
  unsigned char* ws = smem + g.work_off;
  int cur = 0;

  if constexpr (kWork) {
    for (int e = gtid; e < T * U::in_f; e += gstride) g.buf[e] = __half2float(g.x[e]);
    for (int e = gtid; e < BS; e += gstride) g.buf[BS + e] = 0.f;
  }
  barrier<kMode>(g, 0, [&] { if constexpr (kPrefetch) prefetch_layer<R>(g, 0, slot[0]); });

  for (int l = 0; l < g.L; ++l) {
    unsigned long long* tb =
        g.tbuf ? g.tbuf + ((size_t)l * gridDim.x + blockIdx.x) * kTSlots : nullptr;
    TG(tb, 0);
    TS(tb, 3);
    if constexpr (kWork) {
      const float* in = g.buf + (size_t)(l % 3) * BS;
      float* out = g.buf + (size_t)((l + 1) % 3) * BS;
      float* clr = g.buf + (size_t)((l + 2) % 3) * BS;  // the previous layer's input
      for (int e = gtid; e < BS; e += gstride) clr[e] = 0.f;
      if ((l & 1) == 0) layer_units<U>(g, 0, l >> 1, in, out, slot[cur], ws, kPrefetch, tb);
      else              layer_units<D>(g, 1, l >> 1, in, out, slot[cur], ws, kPrefetch, tb);
    }
    TG(tb, 1);
    TS(tb, 12);
    barrier<kMode>(g, l + 1, [&] {
      if constexpr (kPrefetch) {
        if (l + 1 < g.L) prefetch_layer<R>(g, l + 1, slot[cur ^ 1]);  // while others finish
      }
      TS(tb, 13);
    });
    if constexpr (kPrefetch) cur ^= 1;
    TG(tb, 2);
    TS(tb, 14);
  }

  if constexpr (kWork) {
    const int n = T * (((g.L - 1) & 1) == 0 ? U::out_f : D::out_f);
    const float* fin = g.buf + (size_t)(g.L % 3) * BS;
    for (int e = gtid; e < n; e += gstride) g.y[e] = __float2half(__ldcg(fin + e));
  }
}

// ---- host ---------------------------------------------------------------------------------

template <int R>
static size_t core_slot_bytes(const Tiling& tu, const Tiling& td) {
  return std::max(core_layout<Up<R>>(tu).total, core_layout<Down<R>>(td).total);
}

template <int R>
static size_t stack_smem(const Tiling& tu, const Tiling& td, int T, int slots) {
  return slots * core_slot_bytes<R>(tu, td) +
         std::max(work_layout<Up<R>>(tu, T).total, work_layout<Down<R>>(td, T).total);
}

template <int R, int kMode>
static int max_blocks(size_t smem) {  // 0 if one block does not fit in shared memory
  auto kern = tr_stack_kernel<R, kMode>;
  int dev = 0, optin = 0, per_sm = 0, sms = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  if (smem > (size_t)optin) return 0;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)smem));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, kThreads, smem));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  return per_sm * sms;
}

template <int R, int kMode>
static void launch(StackArgs& args, int grid, size_t smem, cudaStream_t stream) {
  auto kern = tr_stack_kernel<R, kMode>;
  const int cap = max_blocks<R, kMode>(smem);
  TORCH_CHECK(cap > 0, "stack kernel does not fit: ", smem, " bytes of shared memory");
  if (grid <= 0 || grid > cap) grid = cap;  // every block must be resident: it spins
  int dev = 0, coop = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
  void* params[] = {&args};
  if (coop) {
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kern, dim3(grid), dim3(kThreads), params,
                                               smem, stream));
  } else {
    TORCH_CHECK(ModeTraits<kMode>::bar != 1, "grid.sync needs cooperative launch");
    kern<<<grid, kThreads, smem, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <int R>
static void dispatch_mode(int64_t mode, StackArgs& args, int grid, size_t smem,
                          cudaStream_t stream) {
  switch (mode) {
    case 0: launch<R, 0>(args, grid, smem, stream); break;
    case 1: launch<R, 1>(args, grid, smem, stream); break;
    case 2: launch<R, 2>(args, grid, smem, stream); break;
    case 3: launch<R, 3>(args, grid, smem, stream); break;
    case 4: launch<R, 4>(args, grid, smem, stream); break;
    case 5: launch<R, 5>(args, grid, smem, stream); break;
    case 6: launch<R, 6>(args, grid, smem, stream); break;
    case 7: launch<R, 7>(args, grid, smem, stream); break;
    default: launch<R, 8>(args, grid, smem, stream); break;
  }
}

template <int R>
static Tiling tiling_for(int type, int64_t kc, int64_t qc, int T) {
  return type == 0 ? make_tiling<Up<R>>((int)kc, (int)qc, T)
                   : make_tiling<Down<R>>((int)kc, (int)qc, T);
}

static void check_half(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kHalf && t.is_contiguous(), name,
              " must be a contiguous CUDA float16 tensor");
}

// tiling = [kc_up, qc_up, kc_down, qc_down]
torch::Tensor stack_forward(torch::Tensor x, std::vector<torch::Tensor> A1,
                            std::vector<torch::Tensor> B2, std::vector<torch::Tensor> C3,
                            int64_t L, int64_t R, std::vector<int64_t> tiling, torch::Tensor buf,
                            torch::Tensor sync, int64_t mode, int64_t grid, int64_t flags,
                            int64_t tbuf_ptr, int64_t bar_base) {
  check_half(x, "x");
  TORCH_CHECK(A1.size() == 2 && B2.size() == 2 && C3.size() == 2 && tiling.size() == 4);
  TORCH_CHECK(R == 8 || R == 16, "R must be 8 or 16");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 1920, "x must be [T, 1920]");
  TORCH_CHECK(L >= 1 && mode >= 0 && mode <= 8);
  for (int i = 0; i < 2; ++i) { check_half(A1[i], "A1"); check_half(B2[i], "B2"); check_half(C3[i], "C3"); }
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int T = (int)x.size(0);
  const int bufstride = T * kMaxFeatures;
  TORCH_CHECK(buf.scalar_type() == torch::kFloat && buf.numel() >= 3 * (int64_t)bufstride,
              "buf must hold 3 * T * 2880 floats");
  TORCH_CHECK(sync.scalar_type() == torch::kInt && sync.numel() >= 3, "sync must be int32[>=3]");
  // two-phase loads: C (kc * 64 int4) and A are loaded in the first round only
  TORCH_CHECK(tiling[0] * 64 <= 2 * kThreads && tiling[2] * 64 <= 2 * kThreads,
              "k chunk too large for the C load round");
  TORCH_CHECK(tiling[0] <= 8 && tiling[2] <= 8, "k chunk at most 8");

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
  args.L = (int)L; args.T = T; args.bufstride = bufstride; args.flags = (int)flags;
  args.bar_base = (unsigned int)bar_base;  // the caller tracks the monotonic counter
  args.tbuf = reinterpret_cast<unsigned long long*>(tbuf_ptr);
  // Y accumulators live in registers: (Mp * qc / 16) * (Rrp / 16) tiles per unit, at most 64
  TORCH_CHECK((round16(T * 12) * tiling[1] / 16) * 2 <= kMaxYTiles * kWarps &&
              (round16(T * 8) * tiling[3] / 16) * 2 <= kMaxYTiles * kWarps,
              "token count x q chunk too large for register Y");

  const int slots = (mode == 4 || mode == 5 || mode == 8) ? 2 : 1;
  size_t smem;
  if (R == 8) {
    args.tl[0] = tiling_for<8>(0, tiling[0], tiling[1], T);
    args.tl[1] = tiling_for<8>(1, tiling[2], tiling[3], T);
    args.core_slot = (int)core_slot_bytes<8>(args.tl[0], args.tl[1]);
    smem = stack_smem<8>(args.tl[0], args.tl[1], T, slots);
  } else {
    args.tl[0] = tiling_for<16>(0, tiling[0], tiling[1], T);
    args.tl[1] = tiling_for<16>(1, tiling[2], tiling[3], T);
    args.core_slot = (int)core_slot_bytes<16>(args.tl[0], args.tl[1]);
    smem = stack_smem<16>(args.tl[0], args.tl[1], T, slots);
  }
  args.work_off = slots * args.core_slot;
  if (R == 8) dispatch_mode<8>(mode, args, (int)grid, smem, stream);
  else        dispatch_mode<16>(mode, args, (int)grid, smem, stream);
  return y;
}

// [smem bytes, resident blocks (one slot), units up, units down, smem bytes, resident blocks
//  (two slots, prefetch)]
std::vector<int64_t> stack_info(int64_t R, int64_t T, std::vector<int64_t> tiling) {
  Tiling tu, td;
  size_t smem, smem4;
  int cap, cap4;
  if (R == 8) {
    tu = tiling_for<8>(0, tiling[0], tiling[1], (int)T); td = tiling_for<8>(1, tiling[2], tiling[3], (int)T);
    smem = stack_smem<8>(tu, td, (int)T, 1); cap = max_blocks<8, 0>(smem);
    smem4 = stack_smem<8>(tu, td, (int)T, 2); cap4 = max_blocks<8, 5>(smem4);
  } else {
    tu = tiling_for<16>(0, tiling[0], tiling[1], (int)T); td = tiling_for<16>(1, tiling[2], tiling[3], (int)T);
    smem = stack_smem<16>(tu, td, (int)T, 1); cap = max_blocks<16, 0>(smem);
    smem4 = stack_smem<16>(tu, td, (int)T, 2); cap4 = max_blocks<16, 5>(smem4);
  }
  return {(int64_t)smem, (int64_t)cap, (int64_t)tu.units, (int64_t)td.units,
          (int64_t)smem4, (int64_t)cap4};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &stack_forward, "L tensor-ring layers in one persistent kernel");
  m.def("info", &stack_info, "[smem, resident blocks, units up, units down, smem/blocks with prefetch]");
}
