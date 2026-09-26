"""CPU oracle for the kernel: y as a sum of independent (a, k) pieces.

Every later kernel (Triton, CUDA v0, v1) is checked against piece_gemm().
"""
from __future__ import annotations

import torch

from factorized_inference import (
    TRSpec, make_cores, materialize_dense_weight, dense_forward,
)


# ---------- step 6: one piece, written as three einsums ----------------------
def piece_einsum(x, A, B, C, a, k):
    """Partial y[t, p, q, r] from one (a, k) piece. x is [T, ni, nj, nk]."""
    xk = x[:, :, :, k]                                   # [T, i, j]
    S1 = torch.einsum("tij,pib->tjpb", xk, A[a])         # stage 1: sum over i
    S2 = torch.einsum("tjpb,bqjc->tpqc", S1, B)          # stage 2: sum over j, b
    return torch.einsum("tpqc,cr->tpqr", S2, C[:, :, k, a])  # stage 3: sum over c

def forward_pieces(x2d, cores, spec, piece=piece_einsum):
    """Sum all R * nk pieces. x2d is [T, in_features] -> [T, out_features]."""
    A, B, C = cores
    ni, nj, nk = spec.input_modes
    P, Q, Rr = spec.output_modes
    T = x2d.shape[0]
    x = x2d.reshape(T, ni, nj, nk)
    y = torch.zeros(T, P, Q, Rr, dtype=x2d.dtype)
    for a in range(spec.rank):
        for k in range(nk):
            y += piece(x, A, B, C, a, k)
    return y.reshape(T, P * Q * Rr)


# ---------- check against the dense oracle -----------------------------------
def check(spec, tokens=3, seed=0, piece=piece_einsum):
    cores = tuple(c.double() for c in make_cores(spec, seed=seed))
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(tokens, spec.in_features, generator=g, dtype=torch.float64)
    expected = dense_forward(x, materialize_dense_weight(cores, spec))  # oracle only
    got = forward_pieces(x, cores, spec, piece)
    err = (got - expected).abs().max().item()
    print(f"{piece.__name__:13s} modes={spec.input_modes}->{spec.output_modes} "
          f"R={spec.rank:2d} pieces={spec.rank * spec.input_modes[2]:3d}  max|err|={err:.1e}")
    return err


if __name__ == "__main__":
    specs = [
        TRSpec((2, 3, 4), (5, 6, 7), 3),      # small, every mode different
    ]
    for piece in (piece_einsum,):
        for spec in specs:
            assert check(spec, piece=piece) < 1e-10
    print("all pieces agree with the dense oracle")
