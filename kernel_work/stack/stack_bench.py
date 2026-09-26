"""Weight-stationary stack experiment (kernel-design/WEIGHT_STATIONARY.md).

L tensor-ring layers, alternating up (1920 -> 2880) and down (2880 -> 1920), batch-1 decode.
Every method runs the same chain of L different layers:

  dense_eager   L x F.linear with L different dense weights (HBM-bound, no L2 flattery)
  dense_graph   the same, captured in one CUDA graph (no launch overhead: the fair baseline)
  v2_graph      L x the v2 fused kernel (design B, 1 launch per layer), in one CUDA graph
  stack         ONE persistent kernel for all L layers, a grid barrier between layers
  barrier_ours  the stack kernel with no work: L + 1 of our grid barriers
  barrier_cg    the same with cooperative-groups grid.sync()

    python kernel_work/stack/stack_bench.py --check
    python kernel_work/stack/stack_bench.py --rank 8 --layers 8,32,128 --out results/h100/stack_r8.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch

from factorized_inference import TRSpec, make_cores, materialize_dense_weight, tr_forward_reference
from factorized_inference.tr_kernel import PreparedTRKernel, pack_cores

REPO = Path(__file__).resolve().parents[2]
UP = ((8, 12, 20), (12, 10, 24))
DOWN = ((12, 10, 24), (8, 12, 20))
DEFAULT_TILING = (2, 4, 2, 4)  # kc, qc for up layers; kc, qc for down layers
_exts = {}


VARIANTS = {  # build name -> (source, extra nvcc flags)
    "default": ("tr_stack.cu", []),
    "timing": ("tr_stack.cu", ["-DSTACK_TIMING"]),       # in-kernel timeline
    "rt": ("tr_stack.cu", ["-DSTACK_RUNTIME_TILING"]),   # tiling only known at run time
    "v3": ("tr_stack.cu", ["-DSTACK_V3"]),               # stages 2 -> 3 in registers (PTX mma)
    "v3timing": ("tr_stack.cu", ["-DSTACK_V3", "-DSTACK_TIMING"]),
    "v3a": ("tr_stack.cu", ["-DSTACK_V3", "-DSTACK_ASYNC"]),    # + cp.async cores
    "v3atiming": ("tr_stack.cu", ["-DSTACK_V3", "-DSTACK_ASYNC", "-DSTACK_TIMING"]),
    "v0": ("tr_stack_v0.cu", []),                        # first session's kernel, as measured
}


def load_stack_ext(variant="default"):
    if variant is True:
        variant = "split"
    elif variant is False:
        variant = "default"
    if variant not in _exts:
        shim = REPO / ".cuda_home"
        if "CUDA_HOME" not in os.environ and shim.exists():
            os.environ["CUDA_HOME"] = str(shim)
        from torch.utils import cpp_extension

        if cpp_extension.CUDA_HOME is None and "CUDA_HOME" in os.environ:
            cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]
        src, extra = VARIANTS[variant]
        name = "tr_stack_ext" if variant == "default" else f"tr_stack_ext_{variant}"
        build = REPO / "build" / name
        build.mkdir(parents=True, exist_ok=True)
        _exts[variant] = cpp_extension.load(
            name=name,
            sources=[str(Path(__file__).resolve().parent / src)],
            build_directory=str(build),
            extra_cuda_cflags=["-O3", "-lineinfo"] + extra,
        )
    return _exts[variant]


def make_layers(L: int, R: int, device="cuda", seed=100):
    """L different layers, alternating up / down; FP16 cores on the GPU."""
    layers = []
    for l in range(L):
        spec = TRSpec(*(UP if l % 2 == 0 else DOWN), rank=R)
        cores = make_cores(spec, device=device, dtype=torch.float16, seed=seed + l)
        layers.append((spec, cores))
    return layers


class Stack:
    """The persistent stack kernel with its packed cores and buffers."""

    def __init__(self, layers, R: int, tiling=DEFAULT_TILING, max_tokens: int = 4,
                 split_calls=False, variant=None):
        self.variant = variant or os.environ.get("STACK_VARIANT", "default")
        self.ext = load_stack_ext(self.variant)
        self.L, self.R, self.tiling = len(layers), R, list(tiling)
        packed = [[], []]
        for l, (spec, cores) in enumerate(layers):
            packed[l % 2].append(pack_cores(cores, spec))
        dev = layers[0][1][0].device
        self.A1, self.B2, self.C3 = [], [], []
        for typ in (0, 1):
            if packed[typ]:
                self.A1.append(torch.stack([p[0] for p in packed[typ]]).contiguous())
                self.B2.append(torch.stack([p[1] for p in packed[typ]]).contiguous())
                self.C3.append(torch.stack([p[2] for p in packed[typ]]).contiguous())
            else:  # a stack of one layer has no down layer; the pointer is never read
                for lst in (self.A1, self.B2, self.C3):
                    lst.append(torch.zeros(8, dtype=torch.float16, device=dev))
        self.buf = torch.zeros(3 * max_tokens * 2880, dtype=torch.float32, device=dev)
        # gen barrier + monotonic counter; then 4 dependency counters per layer (mode 9)
        self.sync = torch.zeros(4 + 4 * len(layers), dtype=torch.int32, device=dev)
        self.bar_base = 0     # monotonic barrier: counter value at the next launch
        self._grid = {}
        self.core_bytes = sum(t.numel() * 2 for t in (*self.A1, *self.B2, *self.C3))

    def __call__(self, x, mode: int = 0, grid: int = 0, flags: int = 0, tbuf=None):
        if self.variant == "v0":
            return self.ext.forward(x, self.A1, self.B2, self.C3, self.L, self.R, self.tiling,
                                    self.buf, self.sync)
        T = x.shape[0]
        if (T, mode) not in self._grid:  # the grid the kernel will use (resident blocks)
            info = self.info(T)
            self._grid[(T, mode)] = info["resident_blocks_prefetch" if mode in (4, 5, 8)
                                         else "resident_blocks"]
        g = self._grid[(T, mode)] if grid <= 0 else min(grid, self._grid[(T, mode)])
        need = g * (self.L + 1)
        if self.bar_base + need >= 2**31 - 1:  # restart the counter well before it wraps
            self.sync[2].zero_()
            self.bar_base = 0
        base = self.bar_base
        if mode in (5, 6, 7):
            self.bar_base += need
        elif mode == 9:  # two monotonic barriers (start, end); dependency counters from zero
            self.bar_base += 2 * g
            self.sync[4:].zero_()
        return self.ext.forward(x, self.A1, self.B2, self.C3, self.L, self.R, self.tiling,
                                self.buf, self.sync, mode, grid, flags,
                                0 if tbuf is None else tbuf.data_ptr(), base)

    def info(self, T: int):
        ext = load_stack_ext("default") if self.variant == "v0" else self.ext  # v0 has no info()
        smem, blocks, units_up, units_down, smem4, blocks4 = ext.info(self.R, T, self.tiling)
        return {"smem_bytes": smem, "resident_blocks": blocks,
                "units_up": units_up, "units_down": units_down,
                "smem_bytes_prefetch": smem4, "resident_blocks_prefetch": blocks4}


def reference_chain(x, layers):
    """FP32 chain through the unoptimized reference (the oracle)."""
    y = x.float()
    for spec, cores in layers:
        y = tr_forward_reference(y, [c.float() for c in cores], spec)
    return y


def errors(actual, expected):
    d = (actual.float() - expected).abs()
    return {"max_abs": d.max().item(),
            "rel_l2": (torch.linalg.vector_norm(actual.float() - expected)
                       / torch.linalg.vector_norm(expected)).item()}


def stream_us(fn, calls=20, reps=7) -> float:
    """CUDA events around `calls` back-to-back calls, median over reps (like the harness)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(calls):
            fn()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(out)


def graphed(fn, x):
    """Capture fn(x_static) into one CUDA graph; return a replay function."""
    static_x = x.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(static_x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(static_x)
    return g.replay


def check(R: int, T_list=(1, 2, 4), L_list=(1, 2, 3, 4, 8)):
    ok = True
    for L in L_list:
        layers = make_layers(L, R)
        st = Stack(layers, R)
        for T in T_list:
            x = torch.randn(T, 1920, device="cuda", dtype=torch.float16)
            ref = reference_chain(x, layers)
            for mode in (0, 3, 4, 5, 6):  # every work mode (barrier / prefetch variants)
                try:
                    for _ in range(2):  # twice: the buffers and the barrier must be reusable
                        y = st(x, mode)
                except RuntimeError as e:
                    if "does not fit" not in str(e):
                        raise
                    print(f"check R={R:2d} L={L:3d} T={T} mode {mode}  skipped: does not fit")
                    continue
                torch.cuda.synchronize()
                err = errors(y, ref)
                good = err["rel_l2"] < 1e-2 and y.shape == ref.shape
                ok &= good
                print(f"check R={R:2d} L={L:3d} T={T} mode {mode}  rel L2 {err['rel_l2']:.2e}  "
                      f"max abs {err['max_abs']:.2e}  {'ok' if good else 'FAIL'}", flush=True)
    return ok


def measure(R: int, L: int, T: int, dense_max_layers: int, tiling, peak_gbs, skip_baselines=False):
    layers = make_layers(L, R)
    x = torch.randn(T, 1920, device="cuda", dtype=torch.float16)
    row = {"rank": R, "layers": L, "tokens": T}

    st = Stack(layers, R, tiling)
    row["stack_info"] = st.info(T)
    row["core_bytes"] = st.core_bytes
    row["stack_err"] = errors(st(x), reference_chain(x, layers))
    row["stack_us"] = stream_us(lambda: st(x))
    row["stack_pf_err"] = errors(st(x, mode=4), reference_chain(x, layers))
    row["stack_pf_us"] = stream_us(lambda: st(x, mode=4))
    row["barrier_ours_us"] = stream_us(lambda: st(x, mode=1))
    try:
        row["barrier_cg_us"] = stream_us(lambda: st(x, mode=2))
        row["stack_cg_err"] = errors(st(x, mode=3), reference_chain(x, layers))
        row["stack_cg_us"] = stream_us(lambda: st(x, mode=3))
    except RuntimeError as err:
        row["barrier_cg_us"] = row["stack_cg_us"] = None
        print("  grid.sync not available:", err)
    if skip_baselines:
        return row

    os.environ["TR_DESIGN"] = "B"
    v2 = [PreparedTRKernel(cores, spec) for spec, cores in layers]

    def v2_chain(x_):
        for f in v2:
            x_ = f(x_)
        return x_

    row["v2_graph_us"] = stream_us(graphed(v2_chain, x))

    if L <= dense_max_layers:  # baseline only: dense weights exist solely in this branch
        Ws = [materialize_dense_weight(cores, spec) for spec, cores in layers]
        row["dense_bytes"] = sum(W.numel() * 2 for W in Ws)

        def dense_chain(x_):
            for W in Ws:
                x_ = torch.nn.functional.linear(x_, W)
            return x_

        row["dense_eager_us"] = stream_us(lambda: dense_chain(x))
        row["dense_graph_us"] = stream_us(graphed(dense_chain, x))
        if peak_gbs:
            row["dense_hbm_floor_us"] = row["dense_bytes"] / (peak_gbs * 1e9) * 1e6
        del Ws
        torch.cuda.empty_cache()
    return row


def sweep(R: int, T: int, L: int = 32, mode: int = 0):
    """Coordinate search over the (kc, qc) tiling of up, then down layers, at fixed L."""
    layers = make_layers(L, R)
    x = torch.randn(T, 1920, device="cuda", dtype=torch.float16)
    ref = reference_chain(x, layers)
    rows = []

    def run(tiling):
        try:
            st = Stack(layers, R, tiling)
            err = errors(st(x, mode), ref)["rel_l2"]
            us = stream_us(lambda: st(x, mode))
        except RuntimeError as e:  # does not fit (shared memory, register Y)
            print(f"  tiling {tiling}: skipped ({str(e).splitlines()[0][:60]})")
            return None
        info = st.info(T)
        rows.append({"tiling": tiling, "stack_us": us, "rel_l2": err, **info})
        print(f"  tiling {tiling}: {us:8.1f} µs  ({us / L:5.2f} µs/layer)  units "
              f"{info['units_up']}/{info['units_down']} on {info['resident_blocks']} blocks  "
              f"err {err:.1e}", flush=True)
        return us if err < 1e-2 else None

    best_up, best_down = (2, 4), (2, 6)
    best = None
    for kc in (1, 2, 4, 5):
        for qc in (2, 4, 5, 10):
            us = run((kc, qc, *best_down))
            if us is not None and (best is None or us < best):
                best, best_up = us, (kc, qc)
    for kc in (1, 2, 3, 4, 6):
        for qc in (3, 4, 6, 12):
            us = run((*best_up, kc, qc))
            if us is not None and us < best:
                best, best_down = us, (kc, qc)
    print(f"best R={R} T={T} mode {mode}: {(*best_up, *best_down)}  {best:.1f} µs for L={L}")
    return {"rank": R, "tokens": T, "layers": L, "mode": mode, "best": [*best_up, *best_down],
            "best_us": best, "rows": rows}


SLOTS = 16


def timeline(st, x, mode: int, sm_mhz: float):
    """One steady-state call of the STACK_TIMING build; per-phase µs averaged over layers."""
    import numpy as np
    T = x.shape[0]
    grid = st.info(T)["resident_blocks" + ("_prefetch" if mode in (4, 5) else "")]
    for _ in range(3):
        st(x, mode)
    tb = torch.zeros(st.L * grid * SLOTS, dtype=torch.int64, device=x.device)
    torch.cuda.synchronize()
    st(x, mode, tbuf=tb)
    torch.cuda.synchronize()
    a = tb.view(st.L, grid, SLOTS).cpu().numpy().astype(np.float64)
    cyc = 1.0 / sm_mhz  # µs per cycle
    out = {"mode": mode, "grid": grid, "layers": {}}
    for typ, name in ((0, "up"), (1, "down")):
        rows = []
        for l in range(2 + typ, st.L - 1, 2):  # skip the first layers and the last one
            L_ = a[l]
            has = L_[:, 4] != 0
            d = {
                "pre_unit": (L_[has, 4] - L_[has, 3]) * cyc,
                "cores_sync": (L_[has, 5] - L_[has, 4]) * cyc,
                "x_s1zero": (L_[has, 6] - L_[has, 5]) * cyc,
                "stage1": L_[has, 7] * cyc, "stage2": L_[has, 8] * cyc, "stage3": L_[has, 9] * cyc,
                "atomics": (L_[has, 11] - L_[has, 10]) * cyc,
                "to_arrive": (L_[has, 12] - L_[has, 11]) * cyc,
                "prefetch": (L_[has, 13] - L_[has, 12]) * cyc,
                "wait": (L_[has, 14] - L_[has, 13]) * cyc,
                "unit_total": (L_[has, 11] - L_[has, 4]) * cyc,
            }
            gstart, garr, grel = L_[:, 0], L_[:, 1], L_[:, 2]
            nxt = a[l + 1][:, 0].min()
            rows.append({
                **{k: float(v.mean()) for k, v in d.items()},
                "unit_total_max": float(d["unit_total"].max()),
                "span_start_to_last_arrive": float((garr.max() - gstart.min()) / 1000),
                "first_arrive": float((garr.min() - gstart.min()) / 1000),
                "release_latency": float((grel.min() - garr.max()) / 1000),
                "release_spread": float((grel.max() - grel.min()) / 1000),
                "start_spread": float((gstart.max() - gstart.min()) / 1000),
                "layer_time": float((nxt - gstart.min()) / 1000),
                "blocks_with_unit": int(has.sum()),
            })
        out["layers"][name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return out


FLAGS = {"no_x": 1, "no_stage1": 2, "no_stage2": 4, "no_stage3": 8, "no_atomics": 16,
         "no_cores": 32}


def ablate(R: int, tiling, T: int = 1, Ls=(8, 32, 128)):
    """µs per layer (slope over L) for each variant: build, mode, grid, ablation flags."""
    variants = [("", False, mode, 0, 0) for mode in (0, 3, 4, 5, 6)]
    variants.append(("1 block/SM", False, 5, -1, 0))
    for name, f in FLAGS.items():
        variants.append((name, False, 6, 0, f))
        variants.append((name, False, 5, 0, f))
    variants.append(("empty units", False, 6, 0, 63))
    variants.append(("empty units", False, 5, 0, 63))
    variants.append(("no units", False, 6, 0, 64))
    variants.append(("no units", False, 5, 0, 64))
    variants.append(("barriers only", False, 7, 0, 0))
    x = torch.randn(T, 1920, device="cuda", dtype=torch.float16)
    stacks = {}
    rows = []
    for name, split, mode, grid, flags in variants:
        pts = []
        for L in Ls:
            key = (L, split)
            if key not in stacks:
                stacks[key] = Stack(make_layers(L, R), R, tiling, split_calls=split)
            st = stacks[key]
            g = st.info(T)["resident_blocks" + ("_prefetch" if mode in (4, 5) else "")] // 2 \
                if grid == -1 else grid
            try:
                pts.append((L, stream_us(lambda: st(x, mode, g, flags))))
            except RuntimeError as e:
                print(f"  {name} mode {mode}: {str(e).splitlines()[0][:80]}")
                break
        if len(pts) == len(Ls):
            fit = slope([{"layers": L, "v": v} for L, v in pts], "v")
            rows.append({"variant": name or "full", "split_calls": split, "mode": mode,
                         "grid": g, "flags": flags, "points": pts, **fit})
            print(f"  {(name or 'full'):16s} split={int(split)} mode {mode} grid {g:4d}: "
                  f"{fit['us_per_layer']:6.2f} µs/layer", flush=True)
    return {"rank": R, "tokens": T, "tiling": list(tiling), "rows": rows}


def round2(R: int, tilings, T: int = 1, Ls=(8, 32, 128), sm_mhz: float = 1980.0):
    x = torch.randn(T, 1920, device="cuda", dtype=torch.float16)
    res = {"ab": [], "skeleton": [], "timeline": []}
    print("-- A/B: V0 vs current (same machine, same tiling)")
    for tiling in tilings:
        stacks = {}
        for L in Ls:
            layers = make_layers(L, R)
            for v in ("v0", "default"):
                stacks[(v, L)] = Stack(layers, R, tiling, variant=v)
            if L == Ls[0]:
                ref = reference_chain(x, layers)
                print(f"   v0 correctness rel L2 {errors(stacks[('v0', L)](x), ref)['rel_l2']:.1e}")
        for v, mode in (("v0", 0), ("default", 0), ("default", 3), ("default", 4),
                        ("default", 5), ("default", 6)):
            pts = [(L, stream_us(lambda: stacks[(v, L)](x, mode))) for L in Ls]
            fit = slope([{"layers": L, "u": u} for L, u in pts], "u")
            res["ab"].append({"tiling": list(tiling), "variant": v, "mode": mode, "points": pts, **fit})
            print(f"   tiling {tiling} {v:8s} mode {mode}: {fit['us_per_layer']:6.2f} µs/layer", flush=True)
        stacks.clear()
    print("-- skeleton: pure barriers, loop without units, loop with empty units")
    tiling = tilings[0]
    stacks = {L: Stack(make_layers(L, R), R, tiling) for L in Ls}
    for name, mode, flags in (("barrier generation (mode 1)", 1, 0),
                              ("barrier grid.sync (mode 2)", 2, 0),
                              ("barrier monotonic (mode 7)", 7, 0),
                              ("loop, no units, mode 6", 6, 64), ("loop, no units, mode 5", 5, 64),
                              ("empty units, mode 6", 6, 63), ("empty units, mode 5", 5, 63)):
        pts = [(L, stream_us(lambda: stacks[L](x, mode, 0, flags))) for L in Ls]
        fit = slope([{"layers": L, "u": u} for L, u in pts], "u")
        res["skeleton"].append({"name": name, "mode": mode, "flags": flags, **fit})
        print(f"   {name:28s} {fit['us_per_layer']:6.2f} µs/layer", flush=True)
    print("-- timeline (STACK_TIMING build), L = 32")
    for tiling in tilings:
        st = Stack(make_layers(32, R), R, tiling,
                   variant=os.environ.get("STACK_TIMING_VARIANT", "timing"))
        for mode in (0, 5, 6):
            tl = timeline(st, x, mode, sm_mhz)
            tl["tiling"] = list(tiling)
            res["timeline"].append(tl)
            for typ, d in tl["layers"].items():
                print(f"   tiling {tiling} mode {mode} {typ:4s}: " +
                      "  ".join(f"{k} {v:.2f}" for k, v in d.items()), flush=True)
    return res


def compare(R: int, tiling, variants=None, modes=None, Ls=(8, 32, 128)):
    modes = modes or tuple(int(m) for m in os.environ.get("COMPARE_MODES", "3,5,6").split(","))
    variants = variants or tuple(os.environ.get("COMPARE_VARIANTS", "rt,default,v3").split(","))
    """µs per layer for every (build variant, mode) on the same layers and input."""
    x = torch.randn(1, 1920, device="cuda", dtype=torch.float16)
    rows = []
    layer_sets = {L: make_layers(L, R) for L in Ls}
    ref8 = reference_chain(x, layer_sets[Ls[0]])
    for v in variants:
        stacks = {L: Stack(layer_sets[L], R, tiling, variant=v) for L in Ls}
        for mode in modes:
            err = errors(stacks[Ls[0]](x, mode), ref8)["rel_l2"]
            pts = [(L, stream_us(lambda: stacks[L](x, mode))) for L in Ls]
            fit = slope([{"layers": L, "u": u} for L, u in pts], "u")
            rows.append({"variant": v, "mode": mode, "rel_l2": err, "points": pts, **fit})
            print(f"   R={R} tiling {tiling} {v:8s} mode {mode}: {fit['us_per_layer']:6.2f} µs/layer"
                  f"   err {err:.1e}", flush=True)
    return {"rank": R, "tiling": list(tiling), "rows": rows}


def slope(rows, key):
    """Least-squares µs per layer over the measured L values."""
    pts = [(r["layers"], r[key]) for r in rows if r.get(key) is not None]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    return {"us_per_layer": sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx,
            "intercept_us": my - mx * sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="correctness against the FP32 chain")
    ap.add_argument("--sweep", action="store_true", help="search the tiling at L = 32")
    ap.add_argument("--skip-baselines", action="store_true", help="stack kernel only")
    ap.add_argument("--sweep-mode", type=int, default=0, help="kernel mode to tune (0 or 4)")
    ap.add_argument("--ablate", action="store_true", help="per-part timing of a unit")
    ap.add_argument("--round2", action="store_true", help="V0 A/B, skeleton, in-kernel timeline")
    ap.add_argument("--sm-mhz", type=float, default=1980.0)
    ap.add_argument("--compare", action="store_true", help="build variants x modes")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--layers", default="8,32,128")
    ap.add_argument("--tokens", default="1")
    ap.add_argument("--tiling", default=",".join(map(str, DEFAULT_TILING)))
    ap.add_argument("--dense-max-layers", type=int, default=128)
    ap.add_argument("--peak-gbs", type=float, default=None, help="HBM peak for the dense floor")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    name = torch.cuda.get_device_name(0)
    peak = args.peak_gbs or (3350.0 if "H100" in name else None)

    if args.check:
        ok = all([check(8), check(16)])
        print("ALL OK" if ok else "SOME CHECKS FAILED")
        raise SystemExit(0 if ok else 1)

    if args.sweep:
        res = [sweep(args.rank, int(T), mode=args.sweep_mode) for T in args.tokens.split(",")]
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"device": name, "sweeps": res}, indent=2))
            print("wrote", args.out)
        return

    tiling = tuple(int(v) for v in args.tiling.split(";")[0].split(","))
    if args.compare:
        res = compare(args.rank, tuple(int(v) for v in args.tiling.split(",")))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"device": name, **res}, indent=2))
            print("wrote", args.out)
        return
    if args.round2:
        tilings = [tuple(int(v) for v in s_.split(",")) for s_ in args.tiling.split(";")]
        res = round2(args.rank, tilings, int(args.tokens.split(",")[0]), sm_mhz=args.sm_mhz)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"device": name, **res}, indent=2))
            print("wrote", args.out)
        return
    if args.ablate:
        res = ablate(args.rank, tiling, int(args.tokens.split(",")[0]))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"device": name, **res}, indent=2))
            print("wrote", args.out)
        return
    rows = []
    for T in (int(v) for v in args.tokens.split(",")):
        for L in (int(v) for v in args.layers.split(",")):
            r = measure(args.rank, L, T, args.dense_max_layers, tiling, peak, args.skip_baselines)
            rows.append(r)
            fmt = lambda k: f"{r[k]:9.1f}" if r.get(k) is not None else "        -"  # noqa: E731
            print(f"R={args.rank:2d} T={T} L={L:4d} | stack {fmt('stack_us')} / cg {fmt('stack_cg_us')} / pf {fmt('stack_pf_us')} | v2_graph "
                  f"{fmt('v2_graph_us')} | dense_graph {fmt('dense_graph_us')} | dense_eager "
                  f"{fmt('dense_eager_us')} | floor {fmt('dense_hbm_floor_us')} | barriers "
                  f"{fmt('barrier_ours_us')} / cg {fmt('barrier_cg_us')} µs | "
                  f"err {r['stack_err']['rel_l2']:.1e}", flush=True)
    fits = {}
    for T in sorted({r["tokens"] for r in rows}):
        sub = [r for r in rows if r["tokens"] == T]
        fits[T] = {k: slope(sub, k) for k in ("stack_us", "stack_cg_us", "stack_pf_us", "v2_graph_us", "dense_graph_us",
                                               "dense_eager_us", "barrier_ours_us",
                                               "barrier_cg_us", "dense_hbm_floor_us")}
        for k, v in fits[T].items():
            if v:
                print(f"  T={T} {k:20s} {v['us_per_layer']:7.2f} µs/layer  "
                      f"(+ {v['intercept_us']:6.1f} µs fixed)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"device": name, "tiling": tiling, "peak_gbs": peak,
                                        "rows": rows, "fits": fits}, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
