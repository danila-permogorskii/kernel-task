"""Which low-bit dense paths work in this torch on this GPU (tools/lowbit_bench.py uses them)."""
import torch

N, K = 17408, 5120
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K ** 0.5
x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)
ref = x.float() @ W.float().T


def rel(y):
    return (torch.linalg.vector_norm(y.float() - ref) / torch.linalg.vector_norm(ref)).item()


# FP8 W8A8, rowwise scales: weight per output channel, activation per token (dynamic)
f8 = torch.float8_e4m3fn
fmax = torch.finfo(f8).max
ws = W.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / fmax          # [N, 1]
W8 = (W.float() / ws).to(f8)
try:
    xs = x.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / fmax      # [T, 1]
    x8 = (x.float() / xs).to(f8)
    y = torch._scaled_mm(x8, W8.T, scale_a=xs, scale_b=ws.T, out_dtype=torch.bfloat16)
    print("fp8 rowwise ok, rel err", rel(y))
except Exception as e:  # noqa: BLE001
    print("fp8 rowwise FAILED:", repr(e)[:300])

# INT4 weight-only (tinygemm), group size 128, BF16 activations
try:
    from torch.ao.quantization.fx._decomposed import quantize_per_channel_group  # noqa: F401
except Exception:  # noqa: BLE001
    pass
try:
    g = 128
    Wg = W.float().reshape(N, K // g, g)
    mn, mx = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
    scale = ((mx - mn) / 15).clamp_min(1e-8)
    q = ((Wg - mn) / scale).round().clamp(0, 15).to(torch.int32).reshape(N, K)
    # tinygemm: w = (q - 8) * scale + zero, zero = mn + 8 * scale
    zero = mn + 8 * scale
    sz = torch.cat([scale, zero], -1).squeeze(-2) if False else torch.stack(
        [scale.squeeze(-1), zero.squeeze(-1)], -1)                           # [N, K/g, 2]
    sz = sz.transpose(0, 1).contiguous().to(torch.bfloat16)                  # [K/g, N, 2]
    qu8 = (q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8)                      # [N, K/2]
    packed = torch._convert_weight_to_int4pack(qu8, 8)
    y = torch._weight_int4pack_mm(x, packed, g, sz)
    print("int4 tinygemm ok, rel err", rel(y), "packed", tuple(packed.shape), packed.dtype)
except Exception as e:  # noqa: BLE001
    print("int4 tinygemm FAILED:", repr(e)[:400])
