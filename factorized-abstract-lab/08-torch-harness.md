# 08 — The model meets PyTorch and the real harness

## Purpose

**Problem.** Guides 06–07 tested the model on Julia code. The project is PyTorch code,
measured by `benchmarks/benchmark.py`. This guide measures what PyTorch's buttons cost,
looks inside one reference call with the profiler, predicts every row of the harness's
table from two measured numbers, and then checks the prediction against the harness's
own JSON.

It answers guide 05's central question with evidence: **at small t, is the reference
limited by the host or by the GPU?** That is the "explain the token-1 profiler evidence"
part of the project's README.

| This guide | Project |
|---|---|
| `split_time` | guide 06's parked measurement, with `torch.cuda._sleep` as the parking kernel |
| `stream_time` | `benchmark.py:164-173`, the `cuda_event_stream_*` numbers |
| `kernels_of` | the profiler trace that `--profile-dir` writes (`benchmark.py:186-199`) |
| `inference_mode`, 1 thread | `benchmark.py:78`, `benchmark.py:89` |

## Mental model

```
   one call of tr_forward_reference, t = 1

   HOST   ┃ einsum ~100 µs ┃ einsum ~100 µs ┃ einsum ~100 µs ┃ reshapes ┃   ≈ 350–400 µs
           (Python → dispatcher → plan the contraction → allocate → launch)

   GPU      ▮ ▮▮  ·  ▮ ▮ ▮  ·  ▮ ▮                                      ≈ 80 µs
            8 kernels, 27 µs of real work, the rest is gaps (g)

   time per call = max(host, gpu) = the host
```

The picture is guide 05's prediction with real numbers in it. Guide 05 guessed 9
kernels and `H = 5 µs` per op, a 45 µs host. The real host is about **8× that**.

### In the studio

The chain studio has a new engineer: PyTorch. Each `einsum` is not one button press but
a small ritual of working out which re-spoolings and desk passes to do, then pressing the
buttons. The hardware finishes each pass quickly and then waits. At t = 1 the session
length is set by the engineer's ritual, not by the hardware.

## Setup

This guide is Python. It uses the project's virtual environment (`.venv` in the project
root), where `factorized_inference` is installed. Type the file into `workspace/` and
run it from there:

**COMMAND** (PowerShell)

```
..\..\.venv\Scripts\python 08_torch_machine.py
```

**COMMAND** (Git Bash)

```
../../.venv/Scripts/python 08_torch_machine.py
```

**What failure looks like**

- `ModuleNotFoundError: No module named 'torch'` or `'factorized_inference'`: you ran
  the system Python, not the project's `.venv`.
- `USDT: … profiler_start` / `profiler_stop` lines on the screen from step 3 on: PyTorch's
  profiler logging to stderr. Harmless; ignore them.

## Step 1 — the parking kernel in PyTorch

**FILE — create `08_torch_machine.py`**

```python
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
```

Notes:

- `split_time` is guide 06's function, line for line. `torch.cuda._sleep(cycles)` is
  PyTorch's built-in spin kernel. The leading underscore means it is internal, but it
  exists for exactly this purpose: PyTorch's own tests use it to park the GPU.
- `n=20`, not guide 06's 200: one reference call costs ~0.4 ms of host time, and
  200 × 0.4 ms = 80 ms would outlast the 20 ms spin. Then the GPU would catch up with the
  host and the GPU clock would measure waiting. **The spin must last longer than
  enqueuing all `n` calls.**
- `e1.elapsed_time(e2)` returns **milliseconds** (CUDA.jl's `CUDA.elapsed` returned
  seconds). Hence `/ 1e3`.
- `@torch.inference_mode()` on `split_time` does what `benchmark.py:78` does to the whole
  worker: it switches off autograd bookkeeping. It saves ~7% of the reference's host time;
  without it, your host numbers would not match the harness's.
- The first three flag lines copy the harness's settings. `allow_fp16_reduced_precision_reduction
  = False` is the FP32 accumulation from guide 06, step 3.

**Expected output** (the device line is exact for this laptop; the spin is ~20 ms)

```
STEP 1 — the device and the parking kernel
  NVIDIA GeForce RTX 3050 Ti Laptop GPU, torch 2.14.0+cu132
  one spin on the GPU clock:   22.10 ms
```

Measured over four runs: 20.23–22.10 ms. The laptop's GPU clock moves with power and
temperature; the spin only has to be long, not exact.

## Step 2 — PyTorch's button prices

**FILE — append to `08_torch_machine.py`**

```python
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
```

Notes:

- `make_cores`, `materialize_dense_weight`, `dense_forward` and `tr_forward_reference`
  come from the project itself (`src/factorized_inference/reference.py`). Same seed as
  the harness (`seed=0`).
- `xm` is `x` reshaped to its digits, the first line of `tr_forward_reference`. The
  third button is the reference's first einsum on its own.

**Predict first:** guide 06 measured a Julia kernel launch at ~7 µs host. What will one
`einsum` cost on the host? And the whole reference call?

**Expected output** (machine-dependent)

```
STEP 2 — PyTorch's button prices at R = 8, t = 1 (GPU parked)
  y.add_(1)                host    12.7 µs   gpu     8.4 µs
  dense_forward            host    26.3 µs   gpu    66.9 µs
  one einsum (the first)   host    94.9 µs   gpu    18.5 µs
  tr_forward_reference     host   342.9 µs   gpu    77.7 µs
```

Measured over five runs: `add_` host 11–14 µs; `dense_forward` host 26–35 µs, gpu
66–67 µs; one einsum host 95–104 µs; the reference host 343–410 µs, gpu 78–91 µs.

**Read it.**

- A PyTorch op with nothing to decide (`add_`) costs ~12 µs on the host: about 2× a raw
  Julia launch. That is Python plus PyTorch's dispatcher.
- **One einsum costs ~100 µs on the host.** It parses the string, decides how to permute
  and reshape both operands, allocates the output, and launches several kernels.
- **The reference costs ~4.5× more host time than GPU time.** At t = 1 the GPU is idle
  most of the call.
- `dense_forward` is one op: 26–35 µs host, 66 µs GPU. It is GPU-bound, reading the 11 MB
  W at ~170 GB/s.

## Step 3 — look inside one call with the profiler

**FILE — append to `08_torch_machine.py`**

```python
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
```

Notes:

- `torch.profiler.profile(activities=[…CUDA])` records every kernel the GPU runs inside
  the `with` block. `e.device_type == …CUDA` keeps only the GPU events; `e.device_time`
  is in **microseconds**, hence `/ 1e6`.
- Kernel names are C++ **mangled** names (`_ZN2at6native18elementwise_kernel…`). `kind`
  sorts them by a substring: `gemm` is a matrix multiply; `elementwise_kernel` is
  PyTorch's generic copy, which is how its permutes run.
- The harness writes the same information, with timestamps, to
  `traces/…/factorized_reference_rank8_tokens1.json` when you pass `--profile-dir`
  (step 5). This step is the numbers from that trace.

**Expected output** (the kernel list is exact for torch 2.14 on an sm_86 GPU; the times
are machine-dependent)

```
STEP 3 — what one call launches on the GPU (R = 8, t = 1)
  dense_forward: 1 GPU operations,    63.8 µs of GPU work
       63.8 µs  other  _Z17gemv2T_kernel_valIii6__halfS0_S0_fLi128ELi16
  tr_forward_reference: 8 GPU operations,    27.5 µs of GPU work
        1.9 µs  copy   _ZN2at6native18elementwise_kernelILi128ELi4EZNS0
        3.3 µs  GEMM   _ZN7cutlass7Kernel2I65cutlass_80_wmma_tensorop_f
        3.7 µs  copy   _ZN2at6native18elementwise_kernelILi128ELi4EZNS0
        1.8 µs  copy   _ZN2at6native18elementwise_kernelILi128ELi4EZNS0
        5.2 µs  GEMM   sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize64
        4.5 µs  copy   _ZN2at6native18elementwise_kernelILi128ELi4EZNS0
        2.2 µs  copy   _ZN2at6native18elementwise_kernelILi128ELi4EZNS0
        5.0 µs  GEMM   sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize32
```

**Read it.**

- **8 GPU operations: 3 GEMMs and 5 copies.** Guide 05 assumed 9 (3 GEMMs, 6 copies).
  The assumption was close; now it is a measurement.
- **27.5 µs of real GPU work**, but step 2 measured ~80 µs of GPU time per call. The
  difference, ~50 µs over 8 kernels, is ~6–7 µs of gap per kernel: guide 06's `g`.
- Dense is one kernel, `gemv2T_kernel`: a matrix-*vector* product, because t = 1. Your
  GEMV work from `batch1-cdna`, in cuBLAS's version.

## Step 4 — predict with max(host, gpu), then measure the stream

**FILE — append to `08_torch_machine.py`**

```python
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
```

Notes:

- For each case, `split_time` gives the two sides separately (GPU parked). The
  prediction is guide 05's rule, `max(host, gpu)`. `stream_time` then runs the calls
  **unparked**, exactly as the harness does.
- This tests the **composition rule** alone. The parts are measured; the question is
  whether "the slower side sets the pace" is the right way to combine them.
- `sorted(samples)[reps // 2]` is the median, as the harness reports.
- `predicted` keeps each prediction and its limiting side for step 5.

**Expected output** (machine-dependent)

```
STEP 4 — max(host, gpu) from the parked measurement, against the unparked stream
   R   t  method       host        gpu    predicted     stream   ratio
   8   1  dense        31.7 µs    64.1 µs      64.1 µs    66.3 µs    1.03
   8   1  reference   383.9 µs    87.3 µs     383.9 µs   420.3 µs    1.09
   8   8  dense        35.0 µs    64.3 µs      64.3 µs    66.1 µs    1.03
   8   8  reference   394.8 µs   159.1 µs     394.8 µs   418.4 µs    1.06
   8  32  dense        33.8 µs    94.6 µs      94.6 µs    97.0 µs    1.03
   8  32  reference   385.7 µs   537.7 µs     537.7 µs   546.6 µs    1.02
  16   1  dense        32.9 µs    64.9 µs      64.9 µs    67.8 µs    1.05
  16   1  reference   373.8 µs   110.6 µs     373.8 µs   407.9 µs    1.09
  16  32  dense        34.1 µs    94.1 µs      94.1 µs    96.5 µs    1.03
  16  32  reference   421.4 µs    2.40 ms      2.40 ms    2.43 ms    1.01
```

Measured over five runs: every ratio between 1.01 and 1.13.

**Read it.** The rule holds everywhere, host-bound or GPU-bound. It is 1–5% off when the
GPU is the limit and 6–13% off when the host is: back-to-back calls also pay a little
Python loop and event overhead that `split_time` spreads differently.

**The reference switches sides.** At t = 1 and t = 8 it is host-bound (~390 µs). At
t = 32 it becomes GPU-bound (0.54 ms at R = 8, 2.4 ms at R = 16). The two sides cross
between t = 8 and t = 32 at R = 8, and between t = 1 and t = 32 at R = 16.

## Step 5 — the harness's own numbers

First run the harness, from the **project root** (two folders up from `workspace/`). It
starts a fresh process for every (method, t) case, so it takes about a minute per rank.

**COMMAND** (each is one line; works in PowerShell and Git Bash)

```
cd ../..
.venv/Scripts/python benchmarks/benchmark.py --device cuda:0 --dtype float16 --rank 8 --tokens 1,8,32 --output results/laptop/rank8.json --profile-dir traces/laptop/rank8
.venv/Scripts/python benchmarks/benchmark.py --device cuda:0 --dtype float16 --rank 16 --tokens 1,32 --output results/laptop/rank16.json
cd factorized-abstract-lab/workspace
```

`--device cuda:0`, not the README's `--device cuda`: with the pinned torch 2.14,
`torch.cuda.set_device("cuda")` needs a device index and fails (`benchmark.py:88`). The
harness accepts `cuda:N` (`benchmark.py:82`). Note this in your report; it will happen on
the H100 too.

**Expected output** (the harness's last lines for rank 8; machine-dependent)

```
tokens=32 dense: host=0.1417 ms; CUDA stream=0.10931199789047241 ms; steady allocated peak=19886080 bytes
tokens=32 factorized_reference: host=0.6716 ms; CUDA stream=0.5791744232177735 ms; steady allocated peak=43612160 bytes
tokens=32 factorized_optimized: host=0.6755 ms; CUDA stream=0.5779967784881592 ms; steady allocated peak=43612160 bytes
```

`factorized_optimized` equals the reference because `submission.py` still falls back to
it. That row becomes your kernel.

Then add the comparison to the script:

**FILE — append to `08_torch_machine.py`**

```python
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
```

Notes:

- `results/` and `traces/` are in the project's `.gitignore`, so these laptop files will
  not end up in a commit.
- The script reads `../../results/laptop/`, relative to `workspace/`. Run it from
  `workspace/`.

**COMMAND**

```
../../.venv/Scripts/python 08_torch_machine.py
```

**Expected output** (step 5 part; machine-dependent)

```
STEP 5 — the harness's own numbers against step 4's prediction
   R   t  method     predicted   harness stream   ratio   limited by
   8   1  dense        64.1 µs         68.2 µs     1.06   GPU
   8   1  reference   383.9 µs        343.3 µs     0.89   host
   8   8  dense        64.3 µs         68.8 µs     1.07   GPU
   8   8  reference   394.8 µs        333.7 µs     0.85   host
   8  32  dense        94.6 µs        109.3 µs     1.16   GPU
   8  32  reference   537.7 µs        579.2 µs     1.08   GPU
  16   1  dense        64.9 µs         68.0 µs     1.05   GPU
  16   1  reference   373.8 µs        328.1 µs     0.88   host
  16  32  dense        94.1 µs        109.6 µs     1.17   GPU
  16  32  reference    2.40 ms         2.43 ms     1.01   GPU
```

**Read it.**

- **GPU-bound rows: 1.01–1.17.** Dense at t = 32 is the loosest (109 vs 94 µs). The
  harness's process has its own allocator state and input; exercise 2 is about it.
- **Host-bound rows: 0.85–0.89.** The harness is ~12% *faster* on the host than this
  script. Host time on a laptop moves with CPU boost clocks, and the harness runs each
  case in a fresh, otherwise idle process. Host-bound numbers are the noisiest numbers in
  the whole lab; always report them as a range.
- **The limiting side is right in all ten rows.** That is the robust result: the model
  says *why* each row costs what it costs. Numbers within ±17% are the precision.

## What this means for the kernel you will write

On this laptop, the uncaptured reference at t = 1 costs ~340 µs, and **all of it is set
by the host**: the GPU needs only ~80 µs of it. A custom kernel launched as **one** op pays one launch (tens of µs of host), plus
its own GPU time. The GPU bytes a fused "cut the ring" kernel must move are x, the three
cores (0.09 MB at R = 8) and y: well under 1 µs at 180 GB/s. So on this laptop, at t = 1:

```
   reference     ≈ 340 µs   host-bound (3 einsums)
   dense         ≈  68 µs   GPU-bound (reads 11 MB)
   fused kernel  ≈ max(one launch, g + tiny work)  →  tens of µs     ← a prediction, not a measurement
```

Guide 05 predicted the same ordering for the H100. The laptop now agrees on every
ingredient it could measure. What remains unmeasured is your kernel's own `eff` at
R = 16, t = 32, where guide 05 says dense wins.

## What is *not* true

- *"The reference is slow because the GPU is slow."* At t = 1 the GPU does 27 µs of work
  per call. The call takes ~340 µs because PyTorch's host path takes that long.
- *"CUDA-event stream time is GPU time."* It measures the GPU stream, and when the host
  is slower, the stream is idle waiting for it. That is why it equals `max(host, gpu)`.
  The harness says this itself: "not kernel-only time" (`BENCHMARK_NOTES.md`).
- *"`torch.compile` or a CUDA graph would fix it, so the kernel is unnecessary."* A graph
  removes the host cost (guide 05, step 5a), but the project requires uncaptured results
  and a custom kernel, and graph results must be reported separately.
- *"These laptop numbers are the H100 answer."* The H100 has ~19× the bandwidth and a
  faster host path under Linux. What transfers is the method (park, two clocks, profile,
  predict, compare), which side limits which case, and the kernel count per call.

## Exercises

1. Open `traces/laptop/rank8/factorized_reference_rank8_tokens1.json` in
   <https://ui.perfetto.dev>. Find the three `aten::einsum` blocks on the CPU row and the
   8 kernels on the GPU row. Measure one gap between two kernels by eye. Does it match `g`?
2. Dense at t = 32: the harness measures 109 µs, your script 94–97 µs. Read the
   harness's worker (`benchmark.py:79-127`) next to step 4 and list every difference in
   how `x` and `W` are created and used. Pick the one you think explains 15%, predict,
   and test it by changing step 4.
3. Add a fifth button to step 2: `torch.mm(x1, W8.t())`. Compare its host price with
   `dense_forward` (which calls `F.linear`). Which one does less work on the host, and why
   might a kernel author care?
4. Predict, then measure: set `n=200` in `split_time` and rerun step 2. Which row's GPU
   number changes, and why? (Guide 06's exercise 1, from the other side.)

## Checklist

- [ ] steps 1–5 run; the device line and step 3's kernel list match
- [ ] I can say what one `einsum` costs on the host, and why it is ~100 µs
- [ ] I can name the 8 GPU operations of one reference call by kind, and compare them
      with guide 05's assumption
- [ ] I can explain why the stream time equals `max(host, gpu)`, and show it holds on
      the harness's own numbers
- [ ] I can say at which t the reference switches from host-bound to GPU-bound, for each R
- [ ] I can state what a one-launch custom kernel should gain at t = 1, and what is
      still unmeasured
- [ ] in studio terms: why the session length at t = 1 is set by the engineer
