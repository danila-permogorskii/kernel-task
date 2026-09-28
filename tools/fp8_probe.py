"""Why is torch FP8 slower than BF16 at small T? Time the pieces (H100)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_kernels import stream_us  # noqa: E402

F8 = torch.float8_e4m3fn
for N, K in ((17408, 5120), (1024, 5120)):
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K ** 0.5
    w8t = W.to(F8).T
    one = torch.ones((), device="cuda")
    ws_row = torch.ones(1, N, device="cuda")
    print(f"--- {N}x{K}")
    for T in (1, 8, 32):
        x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
        x8 = x.to(F8)
        xs_row = torch.ones(T, 1, device="cuda")
        r = {
            "bf16 F.linear": stream_us(lambda: torch.nn.functional.linear(x, W)),
            "fp8 tensorwise, x already fp8": stream_us(
                lambda: torch._scaled_mm(x8, w8t, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)),
            "fp8 rowwise, x already fp8": stream_us(
                lambda: torch._scaled_mm(x8, w8t, scale_a=xs_row, scale_b=ws_row, out_dtype=torch.bfloat16)),
            "act quant only (amax, div, cast)": stream_us(
                lambda: (x.float() / (x.float().abs().amax(1, keepdim=True) / 448)).to(F8)),
        }
        print(f"T={T:3d} " + "  ".join(f"{k}: {v:6.1f}" for k, v in r.items()), flush=True)
