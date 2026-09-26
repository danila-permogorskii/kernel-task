# 08_torch_machine.py — Rung 8: the model meets PyTorch and the real harness
import json
import pathlib
import time

import torch
from factorized_inference import TRSpec, dense_forward, make_cores, materialize_dense_weight, tr_forward_reference

#== STEP 1 ==#
dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = False                            # as benchmark.py:90-91
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.set_num_threads(1)                                                 # as benchmark.py:89
SPIN = 30_000_000                                                        # GPU cycles, ~20 ms


def us(s):
    return f"{s * 1e6:7.1f} µs" if s < 1e-3 else f"{s * 1e3:7.2f} ms"


@torch.inference_mode()                                                  # as benchmark.py:78
def split_time(f, n=20, reps=3):
    """Per-call HOST and GPU time of f(), with the GPU parked first (guide 06, step 1)."""
    f()
    torch.cuda.synchronize()
    host, gpu = float("inf"), float("inf")
    for _ in range(reps):
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(SPIN)                                          # park the GPU
        e1.record()
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        th = time.perf_counter() - t0
        e2.record()
        e2.synchronize()
        host = min(host, th / n)
        gpu = min(gpu, e1.elapsed_time(e2) / 1e3 / n)                  # elapsed_time is in ms
    return host, gpu


print("STEP 1 — the device and the parking kernel")
print(f"  {torch.cuda.get_device_name(dev)}, torch {torch.__version__}")
torch.cuda._sleep(1000)
e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e1.record()
torch.cuda._sleep(SPIN)
e2.record()
e2.synchronize()
print(f"  one spin on the GPU clock: {us(e1.elapsed_time(e2) / 1e3)}")

#== STEP 2 ==#
spec8 = TRSpec(rank=8)
cores8 = make_cores(spec8, device=dev, dtype=torch.float16, seed=0)
W8 = materialize_dense_weight(cores8, spec8)
x1 = torch.randn(1, spec8.in_features, device=dev, dtype=torch.float16)
xm = x1.reshape(1, *spec8.input_modes)
y = torch.zeros(1024, device=dev)

buttons = [("y.add_(1)", lambda: y.add_(1)),
           ("dense_forward", lambda: dense_forward(x1, W8)),
           ("one einsum (the first)", lambda: torch.einsum("tijk,apib->tjkapb", xm, cores8[0])),
           ("tr_forward_reference", lambda: tr_forward_reference(x1, cores8, spec8))]

print("\nSTEP 2 — PyTorch's button prices at R = 8, t = 1 (GPU parked)")
for name, f in buttons:
    h, g = split_time(f)
    print(f"  {name:24s} host {us(h)}   gpu {us(g)}")

#== STEP 3 ==#
def kernels_of(f):
    """(name, GPU seconds) of every kernel and copy that one call of f launches."""
    f()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        f()
        torch.cuda.synchronize()
    return [(e.name, e.device_time / 1e6) for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA]


def kind(kname):
    return "GEMM" if "gemm" in kname else "copy" if "elementwise" in kname else "other"


print("\nSTEP 3 — what one call launches on the GPU (R = 8, t = 1)")
for name in ("dense_forward", "tr_forward_reference"):
    ks = kernels_of(dict(buttons)[name])
    print(f"  {name}: {len(ks)} GPU operations, {us(sum(t for _, t in ks))} of GPU work")
    for kname, t in ks:
        print(f"    {us(t)}  {kind(kname):5s}  {kname[:48]}")

#== STEP 4 ==#
@torch.inference_mode()
def stream_time(f, n=20, reps=5):
    """What the harness reports: CUDA events around n back-to-back calls, median of reps."""
    f()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e1.record()
        for _ in range(n):
            f()
        e2.record()
        e2.synchronize()
        samples.append(e1.elapsed_time(e2) / 1e3 / n)
    return sorted(samples)[reps // 2]


cases = ((8, 1), (8, 8), (8, 32), (16, 1), (16, 32))
predicted = {}                                                           # step 5 reuses these

print("\nSTEP 4 — max(host, gpu) from the parked measurement, against the unparked stream")
print("   R   t  method       host        gpu    predicted     stream   ratio")
for R, t in cases:
    spec = TRSpec(rank=R)
    cores = make_cores(spec, device=dev, dtype=torch.float16, seed=0)
    W = materialize_dense_weight(cores, spec)
    x = torch.randn(t, spec.in_features, device=dev, dtype=torch.float16)
    for name, f in (("dense", lambda: dense_forward(x, W)),
                    ("reference", lambda: tr_forward_reference(x, cores, spec))):
        h, g = split_time(f)
        predicted[(R, t, name)] = (max(h, g), "host" if h > g else "GPU")
        ts = stream_time(f)
        print(f"  {R:2d}  {t:2d}  {name:9s} {us(h)} {us(g)}   {us(max(h, g))} {us(ts)}   {ts / max(h, g):5.2f}")

#== STEP 5 ==#
RESULTS = pathlib.Path("../../results/laptop")                          # written by the harness

print("\nSTEP 5 — the harness's own numbers against step 4's prediction")
print("   R   t  method     predicted   harness stream   ratio   limited by")
for R in (8, 16):
    report = json.loads((RESULTS / f"rank{R}.json").read_text())
    for case in report["cases"]:
        t = case["tokens"]
        for name, key in (("dense", "dense"), ("reference", "factorized_reference")):
            ts = case["methods"][key]["cuda_event_stream_median_ms"] / 1e3
            tp, side = predicted[(R, t, name)]
            print(f"  {R:2d}  {t:2d}  {name:9s} {us(tp)}      {us(ts)}    {ts / tp:5.2f}   {side}")
