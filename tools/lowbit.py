"""Low-bit dense baselines for the Qwen comparisons (tools/lowbit_bench.py, qwen_model_chain.py).

  fp8   W8A8 on H100 Tensor Cores (torch._scaled_mm, cuBLASLt): weight FP8 e4m3 with one
        per-tensor scale; the activation enters as ONE cast kernel. Serving engines fuse the
        activation quantisation into the preceding RMSNorm, so one cast is the fair cost.
        (Measured by tools/fp8_probe.py: rowwise scales 47 µs vs tensorwise 35 µs on 5120->17408,
        and quantising x with separate torch ops costs ~39 µs of launch overhead: both would
        make FP8 look worse than it is.)
  int4  weight-only INT4 (tinygemm, torch._weight_int4pack_mm): asymmetric, group size 128,
        BF16 activations and output.
Both are round-to-nearest quantisations of the same weight; what serving engines ship
(FP8 block scales, GPTQ/AWQ + Marlin) moves the same number of weight bytes.
"""
from __future__ import annotations

import torch

F8 = torch.float8_e4m3fn
F8_MAX = torch.finfo(F8).max


class FP8Linear:
    def __init__(self, W: torch.Tensor):
        self.ws = (W.float().abs().amax().clamp_min(1e-12) / F8_MAX).reshape(())  # per tensor
        self.w8t = (W.float() / self.ws).to(F8).T                                 # [K, N]
        self.one = torch.ones((), device=W.device)
        self.bytes = self.w8t.numel() + 4

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x comes out of RMSNorm at O(1): cast with scale 1 (stands for the fused quantisation)
        return torch._scaled_mm(x.to(F8), self.w8t, scale_a=self.one, scale_b=self.ws,
                                out_dtype=torch.bfloat16)


class INT4Linear:
    def __init__(self, W: torch.Tensor, group: int = 128):
        N, K = W.shape
        Wg = W.float().reshape(N, K // group, group)
        mn, mx = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
        scale = ((mx - mn) / 15).clamp_min(1e-8)
        q = ((Wg - mn) / scale).round().clamp(0, 15).to(torch.int32).reshape(N, K)
        zero = mn + 8 * scale                    # tinygemm dequant: (q - 8) * scale + zero
        sz = torch.stack([scale.squeeze(-1), zero.squeeze(-1)], -1)            # [N, K/g, 2]
        self.sz = sz.transpose(0, 1).contiguous().to(torch.bfloat16)           # [K/g, N, 2]
        self.packed = torch._convert_weight_to_int4pack(
            (q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8), 8)
        self.group = group
        self.bytes = self.packed.numel() * self.packed.element_size() + 2 * self.sz.numel()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch._weight_int4pack_mm(x, self.packed, self.group, self.sz)
