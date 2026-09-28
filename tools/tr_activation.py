#!/usr/bin/env python3
"""Does the ring look better on REAL inputs than on the matrix alone?

tools/tr_quality.py measures ||W - W_ring|| / ||W||, which equals the output error only for
isotropic inputs. Real inputs are not isotropic. Layer 0 of Qwen3.8-27B needs no model run to
get real inputs: they are the token embeddings after the layer's input RMSNorm. So for layer 0's
Gated DeltaNet input projection (in_proj_qkv + in_proj_z, 5120 -> 16384):

  X_cal   inputs from calibration text (kernel-design/*.md), X_test from other text
          (kernel-guides/*.md), tokenised with the model's tokenizer
  error   ||(W - W_hat) x|| / ||W x|| over the TEST tokens (relative output error)
  methods at the parameter budget of a rank-R ring (R = 8, 16, 32, 64):
    ring            TR-SVD + TR-ALS on W (as tr_quality.py)
    svd             truncated SVD of W, same parameter count
    svd_act         activation-aware SVD: best rank-r W_hat for ||(W - W_hat) L||, C = X_cal^T X_cal = L L^T
    ring_act        the ring above, then Adam on its cores to minimise ||(W - W_hat) L|| / ||W L||

    python tools/tr_activation.py --out results/h100/qwen/activation.json
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_quality as q  # noqa: E402
from qwen_fetch_layers import REPO, get  # noqa: E402
from qwen_shapes import SHAPES  # noqa: E402
from tr_quality_visual import untensorize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def fetch_tensor(name: str, index: dict) -> torch.Tensor:
    shard = index[name]
    n = struct.unpack("<Q", get(REPO + shard, 0, 7))[0]
    hdr = json.loads(get(REPO + shard, 8, 8 + n - 1))
    s, e = hdr[name]["data_offsets"]
    raw = get(REPO + shard, 8 + n + s, 8 + n + e - 1)
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(hdr[name]["shape"]).clone()


def texts(pattern_dirs):
    out = []
    for d in pattern_dirs:
        for p in sorted((ROOT / d).glob("*.md")):
            out.append(p.read_text(errors="ignore"))
    return "\n\n".join(out)


def ring_to_matrix(G, ins, outs):
    Y = torch.einsum("iajc,cka->ijk", torch.einsum("aib,bjc->iajc", G[0], G[1]), G[2])
    return untensorize(Y, ins, outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path.home() / "qwen_w")
    ap.add_argument("--ranks", default="8,16,32,64")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    index = json.loads(get(REPO + "model.safetensors.index.json"))["weight_map"]

    cache = a.dir / "embed_and_norm0.pt"
    if not cache.exists():
        t0 = time.time()
        emb = fetch_tensor("model.language_model.embed_tokens.weight", index)
        nw = fetch_tensor("model.language_model.layers.0.input_layernorm.weight", index)
        torch.save({"embed": emb, "norm0": nw}, cache)
        print(f"fetched embeddings {tuple(emb.shape)} in {time.time() - t0:.0f} s", flush=True)
    d = torch.load(cache)
    emb, nw = d["embed"], d["norm0"].float()
    tok_path = a.dir / "tokenizer.json"
    if not tok_path.exists():
        tok_path.write_bytes(get(REPO + "tokenizer.json"))
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_path))
    cal_ids = tok.encode(texts(["kernel-design"])).ids
    test_ids = tok.encode(texts(["kernel-guides"])).ids
    # Qwen3-Next / Qwen3.5 RMSNorm stores the weight zero-centred (applies 1 + w); detect it
    zero_centred = nw.abs().mean().item() < 0.5
    gain = (1 + nw) if zero_centred else nw

    def inputs(ids):
        e = emb[torch.tensor(ids)].cuda().float()
        return e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-6) * gain.cuda()

    Xc, Xt = inputs(cal_ids), inputs(test_ids)
    W = torch.load(a.dir / "layer00.pt")["gdn_qkvz"].cuda().float()   # [16384, 5120]
    ins, outs, _ = SHAPES["gdn_qkvz"]
    nout, nin = W.shape
    C = Xc.T @ Xc / Xc.shape[0]
    evals = torch.linalg.eigvalsh(C).flip(0).clamp_min(0)
    cum = torch.cumsum(evals, 0) / evals.sum()
    act_dims = {f"dims_for_{int(f * 100)}pct": int((cum < f).sum()) + 1 for f in (0.5, 0.9, 0.99)}
    L = torch.linalg.cholesky(C + 1e-6 * evals[0] * torch.eye(nin, device="cuda"))
    Linv = torch.linalg.inv(L)
    ref_t = Xt @ W.T

    def out_err(What):
        return (torch.linalg.vector_norm(Xt @ What.T - ref_t) / torch.linalg.vector_norm(ref_t)).item()

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    Ua, Sa, Vha = torch.linalg.svd(W @ L, full_matrices=False)
    res = {"tokens_cal": len(cal_ids), "tokens_test": len(test_ids), "norm_zero_centred": zero_centred,
           "activation_spectrum": act_dims, "rows": []}
    print(f"tokens cal {len(cal_ids)} test {len(test_ids)}; norm zero-centred {zero_centred}; "
          f"input dims for 50/90/99% energy: {act_dims}", flush=True)
    WL = W @ L
    nWL = torch.linalg.vector_norm(WL)
    for R in (int(v) for v in a.ranks.split(",")):
        params = R * R * sum(o * i for o, i in zip(outs, ins))
        r = max(1, params // (nin + nout))
        G, _, e_mat, _ = q.fit_tr(q.tensorize(W, ins, outs), R, 25)
        W_ring = ring_to_matrix(G, ins, outs)
        W_svd = (U[:, :r] * S[:r]) @ Vh[:r]
        W_svda = ((Ua[:, :r] * Sa[:r]) @ Vha[:r]) @ Linv
        # activation-aware ring: Adam on the cores, loss ||(W - W_hat) L||^2 / ||W L||^2
        P = [torch.nn.Parameter(g.clone()) for g in G]
        opt = torch.optim.Adam(P, lr=1e-3 * max(g.abs().mean().item() for g in G))
        torch.backends.cuda.matmul.allow_tf32 = True
        t0 = time.time()
        for step in range(a.steps):
            opt.zero_grad(set_to_none=True)
            loss = (torch.linalg.vector_norm((W - ring_to_matrix(P, ins, outs)) @ L) / nWL) ** 2
            loss.backward()
            opt.step()
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad():
            W_ringa = ring_to_matrix([p.detach() for p in P], ins, outs)
        row = {"R": R, "params": params, "svd_rank": r, "compression": nin * nout / params,
               "matrix_err_ring": e_mat,
               "out_err_ring": out_err(W_ring), "out_err_svd": out_err(W_svd),
               "out_err_svd_act": out_err(W_svda), "out_err_ring_act": out_err(W_ringa),
               "ring_act_seconds": round(time.time() - t0, 1)}
        res["rows"].append(row)
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items()), flush=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
