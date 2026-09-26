"""Host side of the fused tensor-ring CUDA kernel (csrc/tr_ring.cu).

prepare  : pack the cores once into the operand layouts the kernel reads (no dense W)
call     : choose the tiling, launch design A (3 launches) or B (1 launch)

Environment switches (read at prepare time, inherited by the harness's worker processes):
  TR_DESIGN = A | B    reduction design, default A
  TR_KC, TR_TT         force the tiling: k values / tokens per block (default: heuristic)
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Sequence

import torch

from .reference import TRSpec, tr_forward_reference

_SRC = Path(__file__).resolve().parent / "csrc" / "tr_ring.cu"
_REPO = Path(__file__).resolve().parents[2]
_ext = None


def load_extension(verbose: bool = False):
    """JIT-compile (first time) or load (cached) the extension. Build dir: <repo>/build/."""
    global _ext
    if _ext is None:
        shim = _REPO / ".cuda_home"
        if "CUDA_HOME" not in os.environ and shim.exists():
            os.environ["CUDA_HOME"] = str(shim)
        from torch.utils import cpp_extension

        if cpp_extension.CUDA_HOME is None and "CUDA_HOME" in os.environ:
            cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]
        build = _REPO / "build" / "tr_ring_ext"
        build.mkdir(parents=True, exist_ok=True)
        _ext = cpp_extension.load(
            name="tr_ring_ext",
            sources=[str(_SRC)],
            build_directory=str(build),
            extra_cuda_cflags=["-O3", "-lineinfo"],  # -lineinfo: source view in Nsight Compute
            verbose=verbose,
        )
    return _ext


def _r16(v: int) -> int:
    return -(-v // 16) * 16


def pack_cores(cores: Sequence[torch.Tensor], spec: TRSpec):
    """Repack the cores into the kernel's operand layouts: the same values, plus zero
    padding to multiples of 16 (the Tensor Core tile edge). No dense W, no expansion."""
    A, B, C = cores
    R = spec.rank
    ni, nj, nk = spec.input_modes
    P, Q, Rr = spec.output_modes
    Rc, Rrp, K2p = _r16(R), _r16(Rr), _r16(nj * R)
    A1 = A.permute(0, 2, 1, 3).reshape(R, ni, P * R).contiguous()          # [a][i][(p,b)]
    B2 = torch.zeros(K2p, Q, Rc, dtype=B.dtype, device=B.device)            # [(j,b)][q][c]
    B2[: nj * R, :, :R] = B.permute(2, 0, 1, 3).reshape(nj * R, Q, R)
    C3 = torch.zeros(nk, R, Rc, Rrp, dtype=C.dtype, device=C.device)        # [k][a][c][r]
    C3[:, :, :R, :Rr] = C.permute(2, 3, 0, 1)
    return A1, B2.reshape(K2p, Q * Rc).contiguous(), C3


def choose_tiling(T: int, spec: TRSpec, smem_limit: int, num_sms: int):
    """Pick (kc, tt).

    tt: tokens per block, up to 4 (M = 12*tt rows = 48, a multiple of 16).
    kc: k values per block, the largest that still gives at least one block per SM:
        bigger kc = B loaded fewer times and fewer atomics, but fewer blocks.
    Then shrink until the block fits in shared memory. Returns None if nothing fits.
    """
    ext = load_extension()
    modes = [*spec.input_modes, *spec.output_modes, spec.rank]
    nk, R = spec.input_modes[2], spec.rank
    tt = max(1, min(int(os.environ.get("TR_TT", min(T, 4))), T))
    if "TR_KC" in os.environ:
        kc = int(os.environ["TR_KC"])
    else:
        tiles = -(-T // tt)
        kc = 1
        for cand in range(1, nk + 1):
            if R * tiles * -(-nk // cand) >= num_sms:
                kc = cand
    kc = max(1, min(kc, nk))
    while ext.smem_bytes(modes, kc, tt) > smem_limit:
        if tt > 1:
            tt = (tt + 1) // 2
        elif kc > 1:
            kc = (kc + 1) // 2
        else:
            return None
    return kc, tt


class PreparedTRKernel:
    """Callable returned by prepare_optimized for CUDA float16 cores.

    Design A keeps no state between calls. Design B keeps a zeroed FP32 workspace and
    per-tile counters between calls: one prepared object must not be called concurrently
    from two CUDA streams (see kernel-design/DISCUSSION_POINTS.md).
    """

    def __init__(self, cores: Sequence[torch.Tensor], spec: TRSpec):
        self.cores, self.spec = tuple(cores), spec
        self.ext = load_extension()
        self.design = os.environ.get("TR_DESIGN", "A").upper()
        if self.design not in ("A", "B"):
            raise ValueError("TR_DESIGN must be A or B")
        self.A1, self.B2, self.C3 = pack_cores(cores, spec)
        self.modes = [*spec.input_modes, *spec.output_modes, spec.rank]
        dev = cores[0].device
        props = torch.cuda.get_device_properties(dev)
        self.smem_limit = props.shared_memory_per_block_optin
        self.num_sms = props.multi_processor_count
        self._tiling: dict[int, tuple[int, int]] = {}
        # design B persistent state, grown on demand
        self.ws = torch.zeros(0, dtype=torch.float32, device=dev)
        self.tile_done = torch.zeros(0, dtype=torch.int32, device=dev)
        self._empty = torch.zeros(0, device=dev)

    def tiling(self, T: int):
        if T not in self._tiling:
            self._tiling[T] = choose_tiling(T, self.spec, self.smem_limit, self.num_sms)
        return self._tiling[T]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.is_cuda and x.dtype == torch.float16):
            return tr_forward_reference(x, self.cores, self.spec)
        x = x.contiguous()
        T = x.shape[0]
        tiling = self.tiling(T)
        if tiling is None:  # does not fit this GPU's shared memory (laptop, R = 16)
            warnings.warn("tr_ring: block does not fit in shared memory, using the reference")
            return tr_forward_reference(x, self.cores, self.spec)
        kc, tt = tiling
        if self.design == "A":
            return self.ext.forward(x, self.A1, self.B2, self.C3, self.modes, kc, tt,
                                    False, self._empty, self._empty)
        need = T * self.spec.out_features
        if self.ws.numel() < need:  # grow once; afterwards the kernel keeps it zeroed
            self.ws = torch.zeros(need, dtype=torch.float32, device=x.device)
            self.tile_done = torch.zeros(T, dtype=torch.int32, device=x.device)
        return self.ext.forward(x, self.A1, self.B2, self.C3, self.modes, kc, tt,
                                True, self.ws, self.tile_done)
