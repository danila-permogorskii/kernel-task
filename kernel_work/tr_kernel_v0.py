"""Host side of the fused tensor-ring CUDA kernel (csrc/tr_ring.cu).

prepare  : pack the cores once into the operand layouts the kernel reads (no dense W)
call     : choose the tiling, launch design A (3 launches) or B (1 launch)

Environment switches (read at prepare time, inherited by the harness's worker processes):
  TR_DESIGN = A | B    reduction design, default A
  TR_KC, TR_TT         force the tiling: k values / tokens per block (default: heuristic)
"""
from __future__ import annotations

import os
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


def pack_cores(cores: Sequence[torch.Tensor], spec: TRSpec):
    """Repack the cores into the kernel's operand layouts. Same values, no expansion
    except zero padding of B2 to multiples of 16 (Tensor Core tile size)."""
    A, B, C = cores
    R = spec.rank
    ni, nj, nk = spec.input_modes
    P, Q, Rr = spec.output_modes
    A1 = A.permute(0, 2, 1, 3).reshape(R, ni, P * R).contiguous()        # [a][i][(p,b)]
    K2, N2 = nj * R, Q * R
    K2p, N2p = -(-K2 // 16) * 16, -(-N2 // 16) * 16
    B2 = torch.zeros(K2p, N2p, dtype=B.dtype, device=B.device)
    B2[:K2, :N2] = B.permute(2, 0, 1, 3).reshape(K2, N2)                 # [(j,b)][(q,c)]
    C3 = C.permute(2, 3, 0, 1).contiguous()                             # [k][a][c][r]
    return A1, B2, C3, K2p, N2p


def choose_tiling(T: int, spec: TRSpec, K2p: int, N2p: int, smem_limit: int):
    """Pick (kc, tt): tokens per block up to 8, one k per block, shrunk until it fits."""
    ext = load_extension()
    modes = [*spec.input_modes, *spec.output_modes, spec.rank]
    kc = int(os.environ.get("TR_KC", 1))
    tt = int(os.environ.get("TR_TT", min(T, 8)))
    tt = max(1, min(tt, T))
    kc = max(1, min(kc, spec.input_modes[2]))
    while ext.smem_bytes(modes, kc, tt, K2p, N2p) > smem_limit:
        if tt > 1:
            tt = (tt + 1) // 2
        elif kc > 1:
            kc = (kc + 1) // 2
        else:
            raise RuntimeError("one piece does not fit in shared memory")
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
        self.A1, self.B2, self.C3, self.K2p, self.N2p = pack_cores(cores, spec)
        self.modes = [*spec.input_modes, *spec.output_modes, spec.rank]
        dev = cores[0].device
        self.smem_limit = torch.cuda.get_device_properties(dev).shared_memory_per_block_optin
        self._tiling: dict[int, tuple[int, int]] = {}
        # design B persistent state, grown on demand
        self.ws = torch.zeros(0, dtype=torch.float32, device=dev)
        self.tile_done = torch.zeros(0, dtype=torch.int32, device=dev)
        self._empty = torch.zeros(0, device=dev)

    def tiling(self, T: int):
        if T not in self._tiling:
            self._tiling[T] = choose_tiling(T, self.spec, self.K2p, self.N2p, self.smem_limit)
        return self._tiling[T]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.is_cuda and x.dtype == torch.float16):
            return tr_forward_reference(x, self.cores, self.spec)
        x = x.contiguous()
        T = x.shape[0]
        kc, tt = self.tiling(T)
        if self.design == "A":
            return self.ext.forward(x, self.A1, self.B2, self.C3, self.modes, kc, tt,
                                    self.K2p, self.N2p, False, self._empty, self._empty)
        need = T * self.spec.out_features
        if self.ws.numel() < need:  # grow once; afterwards the kernel keeps it zeroed
            self.ws = torch.zeros(need, dtype=torch.float32, device=x.device)
            self.tile_done = torch.zeros(T, dtype=torch.int32, device=x.device)
        return self.ext.forward(x, self.A1, self.B2, self.C3, self.modes, kc, tt,
                                self.K2p, self.N2p, True, self.ws, self.tile_done)
