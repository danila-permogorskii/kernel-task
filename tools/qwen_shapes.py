"""The linear layers of Qwen3.8-27B as three-core tensor-ring shapes (tools/qwen_bench.py).

Feature sizes come from the model's config.json (kernel-design/physics/qwen3.8-27b_config.json):
hidden 5120, MLP 17408, 24 query heads x 256 (+ output gate), 4 KV heads x 256, Gated DeltaNet
layers with 16 key heads x 128 and 48 value heads x 128. Of the 64 layers, 16 are full attention
and 48 are Gated DeltaNet ("linear attention").

Mode choice per shape (fixed here so every GPU runs the same problem):
  - digits whose max/min is at most 2.5 (close to the cube root, as in qwen_estimate.py)
  - input j divisible by 4 (the kernel's vectorised stage 1), then the order with the fewest
    Tensor Core FLOP after padding R and the output digits to 16: i and r largest
Printed by `python tools/qwen_shapes.py`. The weights stay random (make_cores): these runs
measure speed and numerics on real sizes, not the quality of a compressed Qwen.
"""
from __future__ import annotations

import json
import pathlib

CONFIG = pathlib.Path(__file__).resolve().parents[1] / "kernel-design" / "physics" / "qwen3.8-27b_config.json"

# name: (input modes, output modes, layers that use it per forward pass)
SHAPES = {
    "mlp_gate_up": ((20, 16, 16), (16, 34, 32), 128),  # 5120 -> 17408, gate and up, 64 layers
    "mlp_down":    ((34, 32, 16), (16, 16, 20), 64),   # 17408 -> 5120
    "attn_q_gate": ((20, 16, 16), (16, 24, 32), 16),   # 5120 -> 12288 (24 x 256 x 2)
    "attn_kv":     ((20, 16, 16), (8, 8, 16), 32),     # 5120 -> 1024, k and v
    "o_proj":      ((24, 16, 16), (16, 16, 20), 64),   # 6144 -> 5120: attn o (16) + GDN out (48)
    "gdn_qkvz":    ((20, 16, 16), (16, 32, 32), 48),   # 5120 -> 16384 (2 x 2048 + 2 x 6144)
}


def prod(t):
    out = 1
    for v in t:
        out *= v
    return out


def check_against_config():
    """The shapes above must match the config's feature sizes (and the layer counts)."""
    c = json.loads(CONFIG.read_text())["text_config"]
    h, inter = c["hidden_size"], c["intermediate_size"]
    q_gate = c["num_attention_heads"] * c["head_dim"] * (2 if c["attn_output_gate"] else 1)
    kv = c["num_key_value_heads"] * c["head_dim"]
    kd = c["linear_num_key_heads"] * c["linear_key_head_dim"]
    vd = c["linear_num_value_heads"] * c["linear_value_head_dim"]
    n_full = c["layer_types"].count("full_attention")
    n_lin = c["layer_types"].count("linear_attention")
    want = {
        "mlp_gate_up": (h, inter, 2 * (n_full + n_lin)),
        "mlp_down": (inter, h, n_full + n_lin),
        "attn_q_gate": (h, q_gate, n_full),
        "attn_kv": (h, kv, 2 * n_full),
        "o_proj": (c["num_attention_heads"] * c["head_dim"], h, n_full + n_lin),
        "gdn_qkvz": (h, 2 * kd + 2 * vd, n_lin),
    }
    for name, (ins, outs, n) in SHAPES.items():
        got = (prod(ins), prod(outs), n)
        assert got == want[name], (name, got, want[name])
    # o_proj: attention o is 6144 -> 5120 and the GDN out projection is value dim -> 5120
    assert vd == c["num_attention_heads"] * c["head_dim"] == 6144


if __name__ == "__main__":
    check_against_config()
    for name, (ins, outs, n) in SHAPES.items():
        print(f"{name:12s} {prod(ins):6d} -> {prod(outs):6d}  x{n:3d}   modes {ins} -> {outs}")
