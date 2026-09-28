#!/usr/bin/env python3
"""Download the linear-layer weights of a few Qwen3.8-27B layers, and nothing else.

HTTP range requests against huggingface.co/Qwen/Qwen3.8-27B: the safetensors header of each
shard (8 bytes of length + JSON), then only the byte ranges of the wanted tensors. Saved as one
BF16 torch file per layer in --dir (kept off the repo: these are Qwen's weights, not ours).

Per layer: mlp gate / up / down; full attention q (5120 -> 12288, with the output gate), k, v, o;
Gated DeltaNet in_proj_qkv (5120 -> 10240) + in_proj_z (5120 -> 6144) stored as one 5120 -> 16384
matrix "gdn_qkvz" (the fused projection our shape table uses), out_proj.

    python tools/qwen_fetch_layers.py --layers 0,3,8,15,20,27,32,39,44,51,56,63 --dir ~/qwen_w
"""
from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path

import torch

REPO = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/"
PREFIX = "model.language_model.layers.{}."
WANT = {
    "mlp_gate": "mlp.gate_proj.weight", "mlp_up": "mlp.up_proj.weight",
    "mlp_down": "mlp.down_proj.weight",
    "attn_q_gate": "self_attn.q_proj.weight", "attn_k": "self_attn.k_proj.weight",
    "attn_v": "self_attn.v_proj.weight", "o_proj": "self_attn.o_proj.weight",
    "gdn_qkv": "linear_attn.in_proj_qkv.weight", "gdn_z": "linear_attn.in_proj_z.weight",
    "gdn_out": "linear_attn.out_proj.weight",
}


def get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def header(shard: str, cache: dict):
    if shard not in cache:
        n = struct.unpack("<Q", get(REPO + shard, 0, 7))[0]
        cache[shard] = (8 + n, json.loads(get(REPO + shard, 8, 8 + n - 1)))
    return cache[shard]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3,8,15,20,27,32,39,44,51,56,63")
    ap.add_argument("--dir", type=Path, default=Path.home() / "qwen_w")
    a = ap.parse_args()
    a.dir.mkdir(parents=True, exist_ok=True)
    index = json.loads(get(REPO + "model.safetensors.index.json"))["weight_map"]
    headers, total = {}, 0
    for layer in (int(v) for v in a.layers.split(",")):
        path = a.dir / f"layer{layer:02d}.pt"
        if path.exists():
            print(f"layer {layer}: have it")
            continue
        out = {}
        for short, suffix in WANT.items():
            name = PREFIX.format(layer) + suffix
            if name not in index:
                continue  # full-attention layers have no linear_attn and vice versa
            shard = index[name]
            base, hdr = header(shard, headers)
            meta = hdr[name]
            assert meta["dtype"] == "BF16", meta
            s, e = meta["data_offsets"]
            raw = get(REPO + shard, base + s, base + e - 1)
            assert len(raw) == e - s, (name, len(raw), e - s)
            t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(meta["shape"])
            out[short] = t.clone()
            total += len(raw)
            print(f"layer {layer:2d} {short:12s} {tuple(meta['shape'])}  {len(raw) / 2**20:7.1f} MiB",
                  flush=True)
        if "gdn_qkv" in out:  # one 16384 x 5120 input projection, as the serving shape table
            out["gdn_qkvz"] = torch.cat([out.pop("gdn_qkv"), out.pop("gdn_z")], 0)
        torch.save(out, path)
    print(f"downloaded {total / 2**30:.2f} GiB into {a.dir}")


if __name__ == "__main__":
    main()
