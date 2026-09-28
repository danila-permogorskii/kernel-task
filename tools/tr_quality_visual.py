#!/usr/bin/env python3
"""Small data for the pictures of tools/tr_quality_charts.py (run where the weights are).

  patches   the top-left 48 x 48 block of one real matrix, of its ring approximations
            (R = 8, 64) and of the equal-parameter truncated SVD (R = 64 budget)
  spectra   normalised singular values of every matrix kind in one layer, and of a Gaussian
            matrix of the same shape (512 log-spaced points each)

    python tools/tr_quality_visual.py --layer 27 --out results/h100/qwen/quality_visual.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_quality as q  # noqa: E402
from qwen_shapes import SHAPES  # noqa: E402


def untensorize(X, ins, outs):
    (ni, nj, nk), (P, Q, Rr) = ins, outs
    return (X.reshape(P, ni, Q, nj, Rr, nk).permute(0, 2, 4, 1, 3, 5)
            .reshape(P * Q * Rr, ni * nj * nk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path.home() / "qwen_w")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--matrix", default="mlp_up")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    mats = torch.load(a.dir / f"layer{a.layer:02d}.pt")
    W = mats[a.matrix].cuda().float()
    ins, outs, _ = SHAPES[q.KIND[a.matrix]]
    X = q.tensorize(W, ins, outs)
    assert torch.equal(untensorize(X, ins, outs), W)
    n = 48
    patches = {"original": W[:n, :n]}
    for R in (8, 64):
        G, _, e, _ = q.fit_tr(X, R, 25)
        Y = torch.einsum("iajc,cka->ijk", torch.einsum("aib,bjc->iajc", G[0], G[1]), G[2])
        patches[f"ring_R{R} (err {e:.3f})"] = untensorize(Y, ins, outs)[:n, :n]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    params = 64 * 64 * sum(o * i for o, i in zip(outs, ins))
    r = params // sum(W.shape)
    Ws = (U[:, :r] * S[:r]) @ Vh[:r]
    e = (torch.linalg.vector_norm(W - Ws) / torch.linalg.vector_norm(W)).item()
    patches[f"svd rank {r} = R64 budget (err {e:.3f})"] = Ws[:n, :n]
    spectra = {}
    idx = None
    for name, M in list(mats.items()) + [("gaussian 17408x5120", torch.randn(17408, 5120))]:
        sv = torch.linalg.svdvals(M.cuda().float())
        sv = (sv / sv[0]).cpu()
        idx = torch.unique(torch.logspace(0, torch.log10(torch.tensor(float(len(sv)))), 512).long() - 1)
        spectra[name] = {"index": idx.tolist(), "sigma": sv[idx].tolist(), "n": len(sv)}
    out = {"layer": a.layer, "matrix": a.matrix, "shape": list(W.shape),
           "patches": {k: v.cpu().tolist() for k, v in patches.items()}, "spectra": spectra}
    a.out.write_text(json.dumps(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
