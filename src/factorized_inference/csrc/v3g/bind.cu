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
                           int64_t mg, int64_t nt, int64_t ks, bool bf16) {
  TORCH_CHECK(m.size() == 7, "modes = [ni, nj, nk, P, Q, Rr, R]");
  v3g::Call c{};
  c.i1 = m[0]; c.i2 = m[1]; c.i3 = m[2]; c.o1 = m[3]; c.o2 = m[4]; c.o3 = m[5]; c.rank = m[6];
  c.kc = kc; c.qc = qc; c.tt = tt; c.mg = mg; c.nt = nt; c.ks = ks; c.bf16 = bf16;
  return c;
}

// dynamic shared memory of a compiled variant, 0 if it is not compiled
int64_t v3g_smem(std::vector<int64_t> modes, int64_t kc, int64_t qc, int64_t tt, int64_t mg,
                 int64_t nt, int64_t ks, bool bf16) {
  const v3g::Call c = make_call(modes, kc, qc, tt, mg, nt, ks, bf16);
  int64_t s = 0;
#define V3G_SMEM(NAME) if (!s) s = smem_##NAME(c);
  V3G_UNITS(V3G_SMEM)
#undef V3G_SMEM
  return s;
}

// every compiled variant: [bf16, ni, nj, nk, P, Q, Rr, R, kc, qc, tt, mg, nt, ks]
std::vector<std::vector<int64_t>> v3g_list() {
  std::vector<std::vector<int64_t>> out;
#define V3G_LIST(NAME) list_##NAME(out);
  V3G_UNITS(V3G_LIST)
#undef V3G_LIST
  return out;
}

torch::Tensor v3g_forward(torch::Tensor x, torch::Tensor A1, torch::Tensor B2, torch::Tensor C3,
                          std::vector<int64_t> modes, int64_t kc, int64_t qc, int64_t tt,
                          int64_t mg, int64_t nt, int64_t ks, torch::Tensor ws,
                          torch::Tensor tile_done) {
  const auto st = x.scalar_type();
  TORCH_CHECK(x.is_cuda() && (st == torch::kHalf || st == torch::kBFloat16) && x.dim() == 2 &&
                  x.is_contiguous(),
              "x must be a contiguous 2D CUDA float16 or bfloat16 tensor");
  TORCH_CHECK(A1.scalar_type() == st && B2.scalar_type() == st && C3.scalar_type() == st,
              "x and the packed cores must have the same dtype");
  v3g::Call c = make_call(modes, kc, qc, tt, mg, nt, ks, st == torch::kBFloat16);
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
  c.ablate = ab ? std::atoi(ab) : 0;
  bool done = false;
#define V3G_LAUNCH(NAME) if (!done) done = launch_##NAME(c);
  V3G_UNITS(V3G_LAUNCH)
#undef V3G_LAUNCH
  TORCH_CHECK(done, "V3G variant not compiled for these modes / tiling / dtype");
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &v3g_forward, "V3G: V3T for any compiled modes, FP16 / BF16, design B");
  m.def("smem", &v3g_smem, "dynamic smem of a compiled variant, 0 if not compiled");
  m.def("variants", &v3g_list, "[bf16, ni, nj, nk, P, Q, Rr, R, kc, qc, tt, mg, nt, ks] per variant");
}
