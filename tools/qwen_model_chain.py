#!/usr/bin/env python3
"""One decode step through ALL linear layers of Qwen3.8-27B, every layer with its own weights.

What it simulates (per layer, in config.json order, 64 layers):
    h -> rms_norm -> [full attention]   q+gate 5120->12288, k 5120->1024, v 5120->1024,
                                        o 6144->5120 (input: first 6144 of q+gate = the
                                        attention output's shape; attention math NOT run)
                     [linear attention] qkvz 5120->16384, ba 5120->96 (dense, too small to
                                        factor), out 6144->5120 (input: first 6144 of qkvz;
                                        the Gated DeltaNet recurrence NOT run)
      -> + residual -> rms_norm -> gate, up 5120->17408 -> silu(gate) * up -> down 17408->5120
      -> + residual
    final rms_norm -> lm_head 5120 -> 248320 (dense in every variant)

Variants: dense (a distinct random BF16 weight per layer, ~46 GB: the real model's weight
traffic) and ring R = 8 / 16 (a distinct set of cores per layer, our kernel, tilings from
tools/qwen_sweep.py if given). Unlike tools/qwen_bench.py, nothing stays warm in L2 between
two calls of the same layer: ~400 distinct layers are touched per step, as in the model.

Timing: CUDA events around 20 steps (eager) and a CUDA graph of one step (the launch
overhead a serving engine removes). Weights are random: timing only, no quality claim.

    python tools/qwen_model_chain.py --tokens 1,8,32 --tilings results/h100/qwen/sweep.json \\
        --out results/h100/qwen/model_chain.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_bench import best_tiling  # noqa: E402
from qwen_shapes import CONFIG, SHAPES  # noqa: E402
from lowbit import FP8Linear, INT4Linear  # noqa: E402

from factorized_inference import TRSpec, make_cores  # noqa: E402
from factorized_inference.tr_kernel import PreparedTRKernel  # noqa: E402

CFG = json.loads(CONFIG.read_text())["text_config"]
H, VOCAB = CFG["hidden_size"], CFG["vocab_size"]
DT = torch.bfloat16


class Model:
    """Weights of one variant. linear(name, layer) -> callable(x)."""

    def __init__(self, variant: str, rank: int | None, sweep: Path | None, tokens: list[int]):
        self.variant = variant
        self.layers = []
        g = torch.Generator(device="cuda").manual_seed(0)
        for li, kind in enumerate(CFG["layer_types"]):
            names = (["attn_q_gate", "attn_kv", "attn_kv", "o_proj"] if kind == "full_attention"
                     else ["gdn_qkvz", "o_proj"]) + ["mlp_gate_up", "mlp_gate_up", "mlp_down"]
            ops = [self.make(n, li * 16 + j, rank, sweep, tokens, g) for j, n in enumerate(names)]
            ba = (torch.randn(2 * CFG["linear_num_value_heads"], H, device="cuda", dtype=DT,
                              generator=g) / H ** 0.5) if kind != "full_attention" else None
            self.layers.append((kind, ops, ba))
        self.lm_head = torch.randn(VOCAB, H, device="cuda", dtype=DT, generator=g) / H ** 0.5
        self.norm_w = torch.ones(H, device="cuda", dtype=DT)

    def make(self, shape, seed, rank, sweep, tokens, g):
        ins, outs, _ = SHAPES[shape]
        spec = TRSpec(ins, outs, rank or 8)
        if self.variant in ("dense", "fp8", "int4"):
            w = torch.randn(spec.out_features, spec.in_features, device="cuda", dtype=DT,
                            generator=g) / spec.in_features ** 0.5
            if self.variant == "fp8":
                return FP8Linear(w)
            if self.variant == "int4":
                return INT4Linear(w)
            return lambda x, w=w: F.linear(x, w)
        cores = make_cores(spec, device="cuda", dtype=DT, seed=seed)
        run = PreparedTRKernel(cores, spec)
        for T in tokens:  # fix the tiling per token count now (choose_tiling reads the env)
            t = best_tiling(sweep, shape, rank, T) if sweep else None
            if t:
                os.environ.update(TR_KC=str(t[0]), TR_TT=str(t[1]), TR_QC=str(t[2]))
            run.tiling(T)
            for k in ("TR_KC", "TR_TT", "TR_QC"):
                os.environ.pop(k, None)
        return run

    def step(self, h):
        if self.variant.endswith("f"):
            return self.step_fused(h)
        nw = self.norm_w
        for kind, ops, ba in self.layers:
            x = F.rms_norm(h, (H,), nw, 1e-6)
            if kind == "full_attention":
                qg, _, _ = ops[0](x), ops[1](x), ops[2](x)
                h = h + ops[3](qg[:, :6144].contiguous())
            else:
                qkvz = ops[0](x)
                F.linear(x, ba)
                h = h + ops[1](qkvz[:, :6144].contiguous())
            x = F.rms_norm(h, (H,), nw, 1e-6)
            gate, up, down = ops[-3], ops[-2], ops[-1]
            h = h + down(F.silu(gate(x)) * up(x))
        return F.linear(F.rms_norm(h, (H,), nw, 1e-6), self.lm_head)


def _step_fused(self, h):
    """Ring with FP32 outputs where the next op is ours to fuse: o / out projections (residual
    add) and the MLP (silu * mul, residual add). q/k/v/qkvz keep the BF16 tail: their
    consumers (attention, DeltaNet) are not simulated."""
    from factorized_inference.tr_kernel import load_v3g

    ext = load_v3g()
    nw = self.norm_w
    for kind, ops, ba in self.layers:
        x = F.rms_norm(h, (H,), nw, 1e-6)
        if kind == "full_attention":
            qg, _, _ = ops[0](x), ops[1](x), ops[2](x)
            h = ext.consume_add(h, ops[3].accumulate(qg[:, :6144].contiguous()))
        else:
            qkvz = ops[0](x)
            F.linear(x, ba)
            h = ext.consume_add(h, ops[1].accumulate(qkvz[:, :6144].contiguous()))
        x = F.rms_norm(h, (H,), nw, 1e-6)
        act = ext.consume_silu_mul(ops[-3].accumulate(x), ops[-2].accumulate(x))
        h = ext.consume_add(h, ops[-1].accumulate(act))
    return F.linear(F.rms_norm(h, (H,), nw, 1e-6), self.lm_head)


Model.step_fused = _step_fused


def time_eager(model, h, steps=20, reps=5):
    for _ in range(3):
        model.step(h)
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(steps):
            model.step(h)
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e) / steps)
    return statistics.median(out)


def time_graph(model, h, steps=20, reps=5):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            model.step(h)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        model.step(h)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(steps):
            graph.replay()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e) / steps)
    ms = statistics.median(out)
    # energy: board power sampled every 50 ms (nvidia-smi) over ~3 s of back-to-back replays
    n = max(20, int(3000 / ms))
    torch.cuda.synchronize()
    mon = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                            "-i", "0", "-lms", "50"], stdout=subprocess.PIPE, text=True)
    time.sleep(0.3)
    t0 = time.time()
    for _ in range(n):
        graph.replay()
    torch.cuda.synchronize()
    t1 = time.time()
    time.sleep(0.1)
    mon.terminate()
    watts_all = [float(v) for v in mon.communicate()[0].split() if v.replace(".", "", 1).isdigit()]
    del graph
    watts = statistics.median(watts_all[len(watts_all) // 5:]) if watts_all else float("nan")
    return ms, watts * (t1 - t0) / n, watts


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="dense,fp8,int4,ring8,ring16")
    ap.add_argument("--tokens", default="1,8,32")
    ap.add_argument("--tilings", type=Path)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    os.environ["TR_DESIGN"] = "B"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tokens = [int(v) for v in a.tokens.split(",")]
    rows = []
    result = {"device": torch.cuda.get_device_name(0), "tilings": str(a.tilings), "rows": rows}
    for variant in a.variants.split(","):
        rank = int(variant[4:].rstrip("f")) if variant.startswith("ring") else None
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        model = Model(variant, rank, a.tilings, tokens)
        weights = torch.cuda.memory_allocated() - base
        for T in tokens:
            h = torch.randn(T, H, device="cuda", dtype=DT)
            eager = time_eager(model, h)
            graph, j_step, watts = (None, None, None) if a.no_graph else time_graph(model, h)
            rows.append({"variant": variant, "tokens": T, "weights_bytes": weights,
                         "eager_ms": eager, "graph_ms": graph, "joules_per_step": j_step,
                         "joules_per_token": j_step / T if j_step else None, "avg_watts": watts})
            print(f"{variant:7s} T={T:3d} weights {weights / 2**30:6.2f} GiB  eager {eager:7.2f} ms"
                  + (f"  graph {graph:7.2f} ms  -> {T / graph * 1e3:8.0f} tok/s  "
                     f"{j_step / T * 1e3:7.1f} mJ/token  {watts:5.0f} W (linear+lm_head only)"
                     if graph else ""), flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(result, indent=2) + "\n")
        del model
        torch.cuda.synchronize()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
