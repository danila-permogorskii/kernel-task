#!/usr/bin/env python3
"""Tensor-ring cost of every linear layer of Qwen3.8-27B, and the A100 node estimate on top.

Shapes come from kernel-design/physics/qwen3.8-27b_config.json (the model's config.json).
Each matrix N_in x N_out is written as a 3-core ring: N_in = i*j*k, N_out = p*q*r. Per token

    FLOP   = 2 (i j k p R^2  +  j k p q R^3  +  p q r k R^2)     (checked: x3.600 on the test op)
    params = R^2 (i p + j q + k r)

Two digit choices per shape: "best" = the factorisation (digits 4..128) with the fewest FLOP,
"balanced" = digits closest to the cube root, best order. Node numbers reuse physics_charts.py.

    python tools/qwen_estimate.py
"""
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import physics_charts as pc  # noqa: E402

CFG = json.loads((pc.ROOT / "kernel-design" / "physics" / "qwen3.8-27b_config.json")
                 .read_text())["text_config"]
H, I = CFG["hidden_size"], CFG["intermediate_size"]
HD, NQ, NKV = CFG["head_dim"], CFG["num_attention_heads"], CFG["num_key_value_heads"]
KD = CFG["linear_num_key_heads"] * CFG["linear_key_head_dim"]
VD = CFG["linear_num_value_heads"] * CFG["linear_value_head_dim"]
N_FULL = CFG["layer_types"].count("full_attention")
N_LIN = CFG["layer_types"].count("linear_attention")
GATE = 2 if CFG["attn_output_gate"] else 1

# (name, N_in, N_out, count). Linear-attention projections as in Qwen3-Next / Qwen3.5:
# in_proj_qkvz -> q, k (key dim), v, z (value dim); in_proj_ba -> b, a per value head.
SHAPES = [
    ("mlp gate", H, I, N_FULL + N_LIN), ("mlp up", H, I, N_FULL + N_LIN),
    ("mlp down", I, H, N_FULL + N_LIN),
    ("attn q+gate", H, NQ * HD * GATE, N_FULL), ("attn k", H, NKV * HD, N_FULL),
    ("attn v", H, NKV * HD, N_FULL), ("attn o", NQ * HD, H, N_FULL),
    ("gdn qkvz", H, 2 * KD + 2 * VD, N_LIN), ("gdn out", VD, H, N_LIN),
]
DENSE_ONLY = [("gdn ba", H, 2 * CFG["linear_num_value_heads"], N_LIN)]   # too small to factor
CALLS = sum(c for *_, c in SHAPES)
LM_HEAD = CFG["vocab_size"] * H * 2


def triples(n, lo=4, hi=128):
    return [(a, b, n // a // b) for a in range(lo, hi + 1) if n % a == 0
            for b in range(lo, hi + 1) if (n // a) % b == 0 and lo <= n // a // b <= hi]


def cost(ins, outs, R):
    i, j, k = ins
    p, q, r = outs
    return (2 * (i * j * k * p * R * R + j * k * p * q * R ** 3 + p * q * r * k * R * R),
            R * R * (i * p + j * q + k * r))


def perms(t):
    a, b, c = t
    return {(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)}


def best(nin, nout, R):
    return min(((cost(a, b, R), a, b) for a in triples(nin) for b in triples(nout)),
               key=lambda v: v[0][0])


def balanced(nin, nout, R):
    def spread(t):
        return max(t) / min(t)
    a0 = min(triples(nin), key=spread)
    b0 = min(triples(nout), key=spread)
    return min(((cost(a, b, R), a, b) for a in perms(a0) for b in perms(b0)),
               key=lambda v: v[0][0])


assert abs(cost((8, 12, 20), (12, 10, 24), 8)[0] / pc.DENSE_FLOP - 3.6) < 1e-9


def model(R, pick):
    dense_flop = ring_flop = dense_bytes = ring_bytes = 0
    rows = []
    for name, nin, nout, n in SHAPES:
        (f, prm), a, b = pick(nin, nout, R)
        df = 2 * nin * nout
        dense_flop += n * df
        ring_flop += n * f
        dense_bytes += n * 2 * nin * nout
        ring_bytes += n * 2 * prm
        rows.append((name, nin, nout, n, a, b, f / df, nin * nout / prm))
    return dense_flop, ring_flop, dense_bytes, ring_bytes, rows


NOW = {"eta": 0.10, "lat": 7.0e-6}
TARGET = {"eta": 0.30, "lat": 2.5e-6}
DRAFT = pc.W_STEP_SPEC - pc.W_READ_DENSE   # DFlash2 drafter bytes per step


def node(R, pick, k):
    dense_flop, ring_flop, dense_bytes, ring_bytes, _ = model(R, pick)
    small = sum(n * 2 * a * b for _, a, b, n in DENSE_ONLY)
    weights = pc.W_TARGET - dense_bytes + ring_bytes
    sess = (pc.BUDGET - weights - pc.OTHER_MEM) / pc.RESIDENT
    fixed = CALLS * k["lat"] + (LM_HEAD + ring_bytes + small) / pc.BW_EFF
    per_tok = ring_flop / (k["eta"] * pc.A100["peak"])
    kv = pc.TRAFFIC / pc.BW_EFF

    def plain(n):
        return n / (fixed + n * (per_tok + kv))

    def spec(n):
        return n * pc.ACCEPT / (fixed + DRAFT / pc.BW_EFF + n * (8 * per_tok + kv))

    nmax = max(1, int(sess))
    one = max(plain(1), spec(1))
    total = max(plain(nmax), spec(nmax))
    return {"weights": weights / 2 ** 30, "sess": sess, "one": one, "total": total,
            "over": ring_flop / dense_flop, "comp": dense_bytes / ring_bytes}


if __name__ == "__main__":
    dense_bytes = sum(n * 2 * a * b for _, a, b, n in SHAPES + DENSE_ONLY)
    print(f"linear weights {dense_bytes / 2**30:.2f} GiB, lm_head + embed "
          f"{2 * LM_HEAD / 2**30:.2f} GiB, sum {(dense_bytes + 2 * LM_HEAD) / 2**30:.2f} GiB "
          f"(boot log: target 51.05 GiB incl. vision); calls per step {CALLS}")
    for R in (8, 16, 32):
        print(f"\nR = {R}: per shape, best digits  |  balanced digits")
        _, _, _, _, rb = model(R, best)
        _, _, _, _, rbal = model(R, balanced)
        for (name, nin, nout, n, a, b, fo, co), (*_, a2, b2, fo2, co2) in zip(rb, rbal):
            print(f"  {name:12s} {nin:5d}->{nout:5d} x{n:2d}  {str(a):13s}{str(b):13s} "
                  f"FLOP x{fo:6.2f} mem /{co:6.0f}  |  {str(a2):13s}{str(b2):13s} "
                  f"FLOP x{fo2:6.2f} mem /{co2:6.0f}")
    print("\nnode (A100, 55K context): weights GiB, sessions, tok/s one user, tok/s total")
    print("  today (measured): 51.0 GiB, 3-4 sessions, 65, 185")
    for R in (8, 16, 32, 64):
        for pname, pick in (("best", best), ("balanced", balanced)):
            for kname, k in (("now", NOW), ("target", TARGET)):
                d = node(R, pick, k)
                print(f"  R={R:2d} {pname:8s} {kname:6s}: FLOP x{d['over']:6.2f} mem /{d['comp']:6.0f}  "
                      f"weights {d['weights']:5.1f}  sessions {d['sess']:5.1f}  "
                      f"one {d['one']:5.0f}  total {d['total']:5.0f}")
    print("\nprefill 45K (measured dense 14.3 s; linear share of prefill FLOP assumed 0.8)")
    for R in (8, 16, 32, 64):
        over = model(R, best)[1] / model(R, best)[0]
        naive = 14.3 * (0.2 + 0.8 * over)
        recon = 14.3 * (1 + 0.8 * R * R / 4096)
        print(f"  R={R:2d}: contraction {naive:6.1f} s   reconstruction (+2R^2 per weight per "
              f"4096-token chunk) {recon:5.1f} s")
