// Python entry points of the V3G extension (see v3g.cuh). Design B only: ws and tile_done are
// persistent, zero between calls, owned by the caller (PreparedTRKernel).
#include <cstdlib>

#include "v3g.cuh"
#include "units.h"

#define V3G_DECLARE(NAME)                                   \
  bool launch_##NAME(const v3g::Call& c);                   \
  int64_t smem_##NAME(const v3g::Call& c);                  \
  void list_##NAME(std::vector<std::vector<int64_t>>& out);
V3G_UNITS(V3G_DECLARE)

static v3g::Call make_call(const std::vector<int64_t>& m, int64_t kc, int64_t qc, int64_t tt,
                           int64_t mg, int64_t nt, int64_t ks, int64_t cl, bool bf16) {
  TORCH_CHECK(m.size() == 7, "modes = [ni, nj, nk, P, Q, Rr, R]");
  v3g::Call c{};
  c.i1 = m[0]; c.i2 = m[1]; c.i3 = m[2]; c.o1 = m[3]; c.o2 = m[4]; c.o3 = m[5]; c.rank = m[6];
  c.kc = kc; c.qc = qc; c.tt = tt; c.mg = mg; c.nt = nt; c.ks = ks; c.cl = cl; c.bf16 = bf16;
  return c;
}

// dynamic shared memory of a compiled variant, 0 if it is not compiled
int64_t v3g_smem(std::vector<int64_t> modes, int64_t kc, int64_t qc, int64_t tt, int64_t mg,
                 int64_t nt, int64_t ks, int64_t cl, bool bf16) {
  const v3g::Call c = make_call(modes, kc, qc, tt, mg, nt, ks, cl, bf16);
  int64_t s = 0;
#define V3G_SMEM(NAME) if (!s) s = smem_##NAME(c);
  V3G_UNITS(V3G_SMEM)
#undef V3G_SMEM
  return s;
}

// every compiled variant: [bf16, ni, nj, nk, P, Q, Rr, R, kc, qc, tt, mg, nt, ks, cl]
std::vector<std::vector<int64_t>> v3g_list() {
  std::vector<std::vector<int64_t>> out;
#define V3G_LIST(NAME) list_##NAME(out);
  V3G_UNITS(V3G_LIST)
#undef V3G_LIST
  return out;
}

static torch::Tensor v3g_run(torch::Tensor x, torch::Tensor A1, torch::Tensor B2,
                             torch::Tensor C3, std::vector<int64_t> modes, int64_t kc, int64_t qc,
                             int64_t tt, int64_t mg, int64_t nt, int64_t ks, int64_t cl,
                             torch::Tensor ws, torch::Tensor tile_done, bool fp32_out) {
  const auto st = x.scalar_type();
  TORCH_CHECK(x.is_cuda() && (st == torch::kHalf || st == torch::kBFloat16) && x.dim() == 2 &&
                  x.is_contiguous(),
              "x must be a contiguous 2D CUDA float16 or bfloat16 tensor");
  TORCH_CHECK(A1.scalar_type() == st && B2.scalar_type() == st && C3.scalar_type() == st,
              "x and the packed cores must have the same dtype");
  v3g::Call c = make_call(modes, kc, qc, tt, mg, nt, ks, cl, st == torch::kBFloat16);
  const int64_t in_f = c.i1 * c.i2 * c.i3, out_f = c.o1 * c.o2 * c.o3, T = x.size(0);
  TORCH_CHECK(x.size(1) == in_f, "x has the wrong number of features");
  const int64_t counters = ((T + tt - 1) / tt) * ((c.o2 + qc - 1) / qc);
  TORCH_CHECK(ws.numel() >= T * out_f && tile_done.numel() >= counters,
              "design B workspace too small");
  const at::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty({T, out_f}, x.options());
  if (T == 0) return y;
  c.T = (int)T;
  c.x = x.data_ptr(); c.a1 = A1.data_ptr(); c.b2 = B2.data_ptr(); c.c3 = C3.data_ptr();
  c.ws = ws.data_ptr<float>(); c.y = y.data_ptr();
  c.td = reinterpret_cast<unsigned int*>(tile_done.data_ptr<int32_t>());
  c.stream = at::cuda::getCurrentCUDAStream();
  const char* ab = std::getenv("TR_V3G_ABLATE");  // measurement only
  c.ablate = fp32_out ? 3 : (ab ? std::atoi(ab) : 0);
  bool done = false;
#define V3G_LAUNCH(NAME) if (!done) done = launch_##NAME(c);
  V3G_UNITS(V3G_LAUNCH)
#undef V3G_LAUNCH
  TORCH_CHECK(done, "V3G variant not compiled for these modes / tiling / dtype");
  return fp32_out ? ws.narrow(0, 0, T * out_f).view({T, out_f}) : y;
}

torch::Tensor v3g_forward(torch::Tensor x, torch::Tensor A1, torch::Tensor B2, torch::Tensor C3,
                          std::vector<int64_t> modes, int64_t kc, int64_t qc, int64_t tt,
                          int64_t mg, int64_t nt, int64_t ks, int64_t cl, torch::Tensor ws,
                          torch::Tensor tile_done) {
  return v3g_run(x, A1, B2, C3, modes, kc, qc, tt, mg, nt, ks, cl, ws, tile_done, false);
}

// FP32 output: the returned tensor is a view of ws holding y; the consumer must clear it
// (consume_* below do) before this prepared layer is called again
torch::Tensor v3g_forward_fp32(torch::Tensor x, torch::Tensor A1, torch::Tensor B2,
                               torch::Tensor C3, std::vector<int64_t> modes, int64_t kc,
                               int64_t qc, int64_t tt, int64_t mg, int64_t nt, int64_t ks,
                               int64_t cl, torch::Tensor ws, torch::Tensor tile_done) {
  return v3g_run(x, A1, B2, C3, modes, kc, qc, tt, mg, nt, ks, cl, ws, tile_done, true);
}

// ---- consumers of FP32 outputs: the model's next elementwise op, then clear the workspace ----
__global__ void consume_silu_mul_kernel(float4* __restrict__ g, float4* __restrict__ u,
                                        __nv_bfloat162* __restrict__ out, int64_t n4) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n4;
       i += (int64_t)gridDim.x * blockDim.x) {
    const float4 a = g[i], b = u[i];
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    g[i] = z;
    u[i] = z;
    out[2 * i] = __floats2bfloat162_rn(a.x / (1.f + __expf(-a.x)) * b.x,
                                       a.y / (1.f + __expf(-a.y)) * b.y);
    out[2 * i + 1] = __floats2bfloat162_rn(a.z / (1.f + __expf(-a.z)) * b.z,
                                           a.w / (1.f + __expf(-a.w)) * b.w);
  }
}

__global__ void consume_add_kernel(const __nv_bfloat162* __restrict__ h, float4* __restrict__ y,
                                   __nv_bfloat162* __restrict__ out, int64_t n4) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n4;
       i += (int64_t)gridDim.x * blockDim.x) {
    const float4 v = y[i];
    y[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    const float2 h0 = __bfloat1622float2(h[2 * i]), h1 = __bfloat1622float2(h[2 * i + 1]);
    out[2 * i] = __floats2bfloat162_rn(h0.x + v.x, h0.y + v.y);
    out[2 * i + 1] = __floats2bfloat162_rn(h1.x + v.z, h1.y + v.w);
  }
}

static int grid_for(int64_t n4) { return (int)std::min<int64_t>((n4 + 255) / 256, 1056); }

// silu(g) * u -> BF16, and g, u (FP32 V3G outputs) cleared: the MLP between gate/up and down
torch::Tensor consume_silu_mul(torch::Tensor g, torch::Tensor u) {
  TORCH_CHECK(g.scalar_type() == torch::kFloat && u.sizes() == g.sizes() && g.is_contiguous() &&
                  u.is_contiguous() && g.numel() % 4 == 0, "consume_silu_mul: FP32 [T, N], N % 4 == 0");
  auto out = torch::empty(g.sizes(), g.options().dtype(torch::kBFloat16));
  const int64_t n4 = g.numel() / 4;
  consume_silu_mul_kernel<<<grid_for(n4), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<float4*>(g.data_ptr<float>()), reinterpret_cast<float4*>(u.data_ptr<float>()),
      reinterpret_cast<__nv_bfloat162*>(out.data_ptr()), n4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// h + y -> BF16, and y (FP32 V3G output) cleared: the residual add after a projection
torch::Tensor consume_add(torch::Tensor h, torch::Tensor y) {
  TORCH_CHECK(h.scalar_type() == torch::kBFloat16 && y.scalar_type() == torch::kFloat &&
                  h.sizes() == y.sizes() && h.is_contiguous() && y.is_contiguous() &&
                  y.numel() % 4 == 0, "consume_add: h BF16, y FP32, same [T, N], N % 4 == 0");
  auto out = torch::empty_like(h);
  const int64_t n4 = y.numel() / 4;
  consume_add_kernel<<<grid_for(n4), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat162*>(h.data_ptr()),
      reinterpret_cast<float4*>(y.data_ptr<float>()),
      reinterpret_cast<__nv_bfloat162*>(out.data_ptr()), n4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &v3g_forward, "V3G: V3T for any compiled modes, FP16 / BF16, design B");
  m.def("forward_fp32", &v3g_forward_fp32, "V3G with FP32 output: a view of ws, no tail");
  m.def("consume_silu_mul", &consume_silu_mul, "silu(g) * u -> BF16, clears g and u");
  m.def("consume_add", &consume_add, "h + y -> BF16, clears y");
  m.def("smem", &v3g_smem, "dynamic smem of a compiled variant, 0 if not compiled");
  m.def("variants", &v3g_list, "[bf16, ni, nj, nk, P, Q, Rr, R, kc, qc, tt, mg, nt, ks, cl] per variant");
}
