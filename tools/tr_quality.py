#!/usr/bin/env python3
"""How well does a three-core tensor ring approximate the REAL weights of Qwen3.8-27B?

The two questions of the talk:
  Q1  does rank R = 8 / 16 (/ 32) keep a real 5120 x 17408-class matrix?
  Q2  does the tensorization (how each side is split into three modes) matter?

Per weight matrix W [out, in] (tools/qwen_fetch_layers.py) and tensorization (ins, outs):
  X[(p,i), (q,j), (r,k)] = W[(p,q,r), (i,j,k)]          the 3-way tensor the ring factorizes
  X ~ sum_abc G1[a,(p,i),b] G2[b,(q,j),c] G3[c,(r,k),a]  = our cores A[a,p,i,b], B, C
Fit: TR-SVD initialisation, then TR-ALS sweeps (Zhao et al. 2016, arXiv:1606.05535), FP32 on the GPU.
Reported: relative Frobenius error ||W - W_TR|| / ||W|| (= relative output error for isotropic
inputs; real activations are not isotropic, see the caveats), after TR-SVD and after ALS.

Baselines and controls (all at the same parameter count as the ring):
  svd_eq      truncated SVD of W with rank = TR params / (in + out): the optimal low-rank matrix
              (Eckart-Young), exact from the singular values
  gaussian    a random Gaussian matrix of the same shape: what "no structure at all" gives
  synthetic   a matrix that IS a rank-8 ring (make_cores): the fit must reach ~0 (checks the fitter)
Q2 variants per matrix: balanced (tools/qwen_shapes.py), balanced with the output modes in
reverse order, "skewed" (the FLOP-optimal digits of tools/qwen_estimate.py), and balanced
after a random permutation of rows and columns (does the natural index order carry structure?).
Kernel check: the fitted R = 8 cores of one matrix run through our kernel (PreparedTRKernel,
BF16) and are compared with W_TR x and with the original W x.

    python tools/tr_quality.py --dir ~/qwen_w --out results/h100/qwen/quality.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_shapes import SHAPES  # noqa: E402

from factorized_inference import TRSpec, make_cores, materialize_dense_weight  # noqa: E402

# matrix name in the fetched files -> shape name in qwen_shapes (same in/out features)
KIND = {"mlp_gate": "mlp_gate_up", "mlp_up": "mlp_gate_up", "mlp_down": "mlp_down",
        "attn_q_gate": "attn_q_gate", "attn_k": "attn_kv", "attn_v": "attn_kv",
        "o_proj": "o_proj", "gdn_out": "o_proj", "gdn_qkvz": "gdn_qkvz"}


# ---- tensorization ------------------------------------------------------------------------
def tensorize(W, ins, outs):
    (ni, nj, nk), (P, Q, Rr) = ins, outs
    return (W.reshape(P, Q, Rr, ni, nj, nk).permute(0, 3, 1, 4, 2, 5)
            .reshape(P * ni, Q * nj, Rr * nk).contiguous())


def triples(n, lo=4, hi=128):
    return [(a, b, n // a // b) for a in range(lo, hi + 1) if n % a == 0
            for b in range(lo, hi + 1) if (n // a) % b == 0 and lo <= n // a // b <= hi]


def skewed(nin, nout, R):
    """FLOP-optimal digits (tools/qwen_estimate.py 'best'): min 2(ijkpR^2 + jkpqR^3 + pqrkR^2)."""
    def flop(a, b):
        (i, j, k), (p, q, r) = a, b
        return i * j * k * p * R * R + j * k * p * q * R ** 3 + p * q * r * k * R * R
    return min(((a, b) for a in triples(nin) for b in triples(nout)), key=lambda ab: flop(*ab))


# ---- TR-SVD + TR-ALS on a 3-way tensor ------------------------------------------------------
def tr_svd(X, R):
    """Zhao et al. 2016, Alg. 1 for three cores with ranks (R, R, R): r0 * r1 = R^2 at the
    first unfolding, split evenly."""
    n1, n2, n3 = X.shape
    U, S, Vh = torch.linalg.svd(X.reshape(n1, n2 * n3), full_matrices=False)
    k = min(R * R, n1)
    G1 = U[:, :k]
    rest = S[:k, None] * Vh[:k]
    if k < R * R:  # n1 < R^2: pad the rank with zeros
        G1 = torch.cat([G1, G1.new_zeros(n1, R * R - k)], 1)
        rest = torch.cat([rest, rest.new_zeros(R * R - k, n2 * n3)], 0)
    G1 = G1.reshape(n1, R, R).permute(1, 0, 2).contiguous()          # [a, n1, b]
    rest = rest.reshape(R, R, n2, n3).permute(1, 2, 3, 0).reshape(R * n2, n3 * R)  # [(b,n2),(n3,a)]
    U2, S2, V2 = torch.svd_lowrank(rest, q=R + 8, niter=4)
    G2 = U2[:, :R].reshape(R, n2, R).contiguous()                    # [b, n2, c]
    G3 = (S2[:R, None] * V2[:, :R].T).reshape(R, n3, R).contiguous()  # [c, n3, a]
    return [G1, G2, G3]


def als_update(Xr, G1, G2, G3, lam=1e-6):
    """New G1 for Xr[i,j,k] ~ sum G1[a,i,b] G2[b,j,c] G3[c,k,a]; also <X, fit> and ||fit||^2.
    The normal equations are solved in FP64 with a small ridge: when R exceeds what the data
    supports the Gram matrix is singular and an FP32 solve blows up (seen at R = 32, 64)."""
    R = G1.shape[0]
    Y = torch.einsum("ijk,cka->ijca", Xr, G3)
    rhs = torch.einsum("ijca,bjc->iab", Y, G2).reshape(Xr.shape[0], R * R)
    del Y
    GB = torch.einsum("bjc,BjC->bcBC", G2, G2)
    GC = torch.einsum("cka,CkA->caCA", G3, G3)
    gram = torch.einsum("bcBC,caCA->abAB", GB, GC).reshape(R * R, R * R).double()
    gram = 0.5 * (gram + gram.T)
    gram_r = gram + lam * gram.diagonal().mean() * torch.eye(R * R, device=gram.device,
                                                               dtype=gram.dtype)
    rhs = rhs.double()
    M = torch.linalg.solve(gram_r, rhs.T).T                          # [i, (a,b)]
    inner = (M * rhs).sum()
    fit2 = torch.einsum("ix,xy,iy->", M, gram, M)
    return M.float().reshape(-1, R, R).permute(1, 0, 2).contiguous(), inner, fit2


def rel_err_exact(X, G):
    with torch.no_grad():
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        Y = torch.einsum("aib,bjc->iajc", G[0], G[1])
        Y = torch.einsum("iajc,cka->ijk", Y, G[2])
        e = (torch.linalg.vector_norm(X - Y) / torch.linalg.vector_norm(X)).item()
        torch.backends.cuda.matmul.allow_tf32 = prev
    return e


def fit_tr(X, R, sweeps):
    X = X.float()
    xs = [X, X.permute(1, 2, 0).contiguous(), X.permute(2, 0, 1).contiguous()]
    nx2 = X.double().pow(2).sum()
    G = tr_svd(X, R)
    e_svd = rel_err_exact(X, G)
    best, best_e, hist = [g.clone() for g in G], e_svd, []
    for _ in range(sweeps):
        for m in range(3):  # rotate: update core m with the tensor's mode m first
            g = [G[m], G[(m + 1) % 3], G[(m + 2) % 3]]
            G[m], inner, fit2 = als_update(xs[m], *g)
        e = math.sqrt(max(0.0, (nx2 - 2 * inner + fit2).item()) / nx2.item())
        hist.append(e)
        if e < best_e:
            best, best_e = [g.clone() for g in G], e
        elif e > best_e + 1e-3:  # exact ALS never goes up: numerical trouble, keep the best
            break
    return best, e_svd, rel_err_exact(X, best), hist


def svd_eq(sv, params, nin, nout):
    r = max(1, params // (nin + nout))
    e2 = sv.pow(2)
    return r, math.sqrt(e2[r:].sum().item() / e2.sum().item())


def energy_rank(sv, frac):
    c = torch.cumsum(sv.pow(2), 0) / sv.pow(2).sum()
    return int((c < frac).sum().item()) + 1


# ---- main -----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path.home() / "qwen_w")
    ap.add_argument("--ranks", default="8,16,32")
    ap.add_argument("--sweeps", type=int, default=25)
    ap.add_argument("--q2-layers", default="20,27", help="layers for the tensorization study")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False  # full FP32 (TF32 made ALS unstable)
    ranks = [int(v) for v in a.ranks.split(",")]
    q2 = {int(v) for v in a.q2_layers.split(",")}
    rows = []
    res = {"device": torch.cuda.get_device_name(0), "sweeps": a.sweeps, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.out.write_text(json.dumps(res, indent=1) + "\n")

    def record(**kw):
        rows.append(kw)
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in kw.items() if k != "hist"), flush=True)
        save()

    # controls first: they validate the fitter
    ins, outs, _ = SHAPES["mlp_gate_up"]
    spec = TRSpec(ins, outs, 8)
    Wsyn = materialize_dense_weight(make_cores(spec, device="cuda", seed=0), spec)
    Wgau = torch.randn(spec.out_features, spec.in_features, device="cuda")
    for name, W in (("synthetic_ring_R8", Wsyn), ("gaussian", Wgau)):
        sv = torch.linalg.svdvals(W)
        for R in ranks:
            t0 = time.time()
            G, e_svd, e_als, hist = fit_tr(tensorize(W, ins, outs), R, a.sweeps)
            params = R * R * sum(o * i for o, i in zip(outs, ins))
            r_eq, e_eq = svd_eq(sv, params, spec.in_features, spec.out_features)
            record(layer=-1, matrix=name, shape="mlp_gate_up", tensorization="balanced", R=R,
                   ins=ins, outs=outs, params=params, err_trsvd=e_svd, err_tr=e_als,
                   svd_eq_rank=r_eq, err_svd_eq=e_eq, seconds=round(time.time() - t0, 1),
                   hist=hist)
    del Wsyn, Wgau

    kernel_checked = False
    for path in sorted(a.dir.glob("layer*.pt")):
        layer = int(path.stem[5:])
        mats = torch.load(path)
        for mname, Wc in mats.items():
            shape = KIND[mname]
            W = Wc.cuda().float()
            nout, nin = W.shape
            ins, outs, _ = SHAPES[shape]
            assert math.prod(ins) == nin and math.prod(outs) == nout, (mname, W.shape)
            sv = torch.linalg.svdvals(W)
            spectrum = {f"rank_for_{int(f * 100)}pct_energy": energy_rank(sv, f)
                        for f in (0.5, 0.9, 0.99)}
            variants = [("balanced", ins, outs, W)]
            if layer in q2:
                variants.append(("balanced_out_reversed", ins, tuple(reversed(outs)), W))
                si, so = skewed(nin, nout, 8)
                variants.append(("skewed", si, so, W))
                g = torch.Generator(device="cuda").manual_seed(7)
                pr = torch.randperm(nout, device="cuda", generator=g)
                pc = torch.randperm(nin, device="cuda", generator=g)
                variants.append(("balanced_random_perm", ins, outs, W[pr][:, pc]))
            for vname, vi, vo, Wv in variants:
                for R in ranks:
                    if vname != "balanced" and R > 16:
                        continue
                    t0 = time.time()
                    G, e_svd, e_als, hist = fit_tr(tensorize(Wv, vi, vo), R, a.sweeps)
                    params = R * R * sum(o * i for o, i in zip(vo, vi))
                    r_eq, e_eq = svd_eq(sv, params, nin, nout)
                    record(layer=layer, matrix=mname, shape=shape, tensorization=vname, R=R,
                           ins=vi, outs=vo, params=params, compression=nin * nout / params,
                           err_trsvd=e_svd, err_tr=e_als, svd_eq_rank=r_eq, err_svd_eq=e_eq,
                           seconds=round(time.time() - t0, 1), hist=hist,
                           **(spectrum if vname == "balanced" and R == ranks[0] else {}))
                    if not kernel_checked and vname == "balanced" and R == 8:
                        kernel_check(res, W, G, vi, vo, R, layer, mname)
                        kernel_checked = True
                        save()
            del W
            torch.cuda.empty_cache()
    print("wrote", a.out)


def kernel_check(res, W, G, ins, outs, R, layer, mname):
    """Run the fitted cores through our kernel: y_kernel vs W_TR x (our numerics) and vs W x."""
    from factorized_inference.tr_kernel import PreparedTRKernel

    (ni, nj, nk), (P, Q, Rr) = ins, outs
    A = G[0].reshape(R, P, ni, R)          # G1[a,(p,i),b] -> A[a,p,i,b]
    B = G[1].reshape(R, Q, nj, R)
    C = G[2].reshape(R, Rr, nk, R)
    spec = TRSpec(ins, outs, R)
    cores = tuple(c.to(torch.bfloat16).contiguous() for c in (A, B, C))
    run = PreparedTRKernel(cores, spec)
    x = torch.randn(8, spec.in_features, device="cuda", dtype=torch.bfloat16)
    y = run(x).float()
    Wtr = materialize_dense_weight(tuple(c.float() for c in cores), spec)
    y_tr = x.float() @ Wtr.T
    y_w = x.float() @ W.T
    rel = lambda u, v: (torch.linalg.vector_norm(u - v) / torch.linalg.vector_norm(v)).item()  # noqa: E731
    res["kernel_check"] = {"layer": layer, "matrix": mname, "R": R, "tokens": 8,
                           "kernel_vs_ring_rel_l2": rel(y, y_tr),
                           "kernel_vs_original_layer_rel_l2": rel(y, y_w),
                           "ring_vs_original_layer_rel_l2": rel(y_tr, y_w)}
    print("KERNEL CHECK", res["kernel_check"], flush=True)


if __name__ == "__main__":
    main()
