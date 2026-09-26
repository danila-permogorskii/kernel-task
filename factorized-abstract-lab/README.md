# Factorized-inference abstract lab

A distilled version of `factorized-inference-assignment`. The GPU, PyTorch and the
benchmark harness are stripped away; what is left is the **method**, in problems small
enough to check by hand.

## The whole project in one sentence

> Compute **y = M·x** where M is never given as a table, only as a *recipe* of small
> pieces. Choose the **order** of operations and **where the intermediates live**.

## The studio: one picture for the whole ladder

Every guide is told twice: once in matrices, once as a **sound studio**. The studio is
the mental model; the matrices are how it is written down.

```
 MICROPHONES        MIXER V         GROUPS          MIXER U          SPEAKERS
      x        ──►  (knobs)   ──►     z       ──►   (knobs)   ──►       y
   many (n)                        few (r)                           many (n)

 Many inputs ──V──► few groups ──U──► many outputs          y = U·(V·x)
```

A number in a matrix is a **volume knob**: how much of one input goes into one output.
The whole studio, microphones to speakers, is `M`. Owning `M` as one board means
`n × n` knobs; owning the two desks means `2·n·r` knobs.

The same studio grows with the ladder. Each guide adds one piece of equipment:

```
 guide  studio                                        the maths
 ─────  ────────────────────────────────────────────  ──────────────────────────────
 00     the stopwatch and the tape counter            best_time, alloc_bytes
 01     two desks with a few GROUPS between them      M = U·V, route through z = V·x
 02     mics and speakers stand on a GRID; one small  M = B ⊗ A, index = digits
        desk per grid direction
 03     desks joined by a multi-line GROUP BUS that   ring: W = tr(A·B), links a, b
        loops back to the first desk                  R² intermediate, cut the ring
 04     three desks round the loop, many SONGS        three cores, tokens t
 05     the ENGINEER pressing buttons, the TAPE       host H, kernel gap g, HBM
        machine between desks                         bandwidth, fused kernel
```

The dictionary, used in every guide:

| Studio | Maths | Project |
|---|---|---|
| microphone signals | input `x`, digits `i, j, k` | `x[t, 1920]`, `input_modes` (8, 12, 20) |
| speaker signals | output `y`, digits `p, q, r` | `output_modes` (12, 10, 24) |
| a volume knob | one number of a matrix or core | one FP16 weight |
| the full board | dense `M` / `W` | `materialize_dense_weight` |
| audio groups | the rank link `z`, `a`, `b`, `c` | `rank` R = 8 or 16 |
| a group bus of R lines | a link index running between cores | the `a, b, c` letters in the einsums |
| songs played through the same desk | tokens `t` | `tokens` 1, 8, 32 |
| bouncing tracks to tape and back | storing `T1`, `T2` in slow memory | HBM traffic of the reference |
| live routing inside the desk | fused kernel, slices on-chip | the `submission.py` kernel |
| the engineer pressing buttons | host dispatch, cost `H` per op | Python + PyTorch launch overhead |

**Where the analogy stops.** Real knobs only turn the volume down or up. These knobs can
also be **negative** (`randn`), which a studio would call "flip the phase". A studio
also mixes continuously in time; here each song is one vector, mixed once.

## The ladder

Each guide adds exactly one ingredient. Do them in order; every later guide assumes the
earlier mental model.

```
 guide  file              new ingredient                     project counterpart
 ─────  ────────────────  ─────────────────────────────────  ──────────────────────────────
 00     common.jl         measuring: best time, allocations  benchmark.py timing helpers
 01     01_lowrank.jl     ORDER of operations; never build M "never construct the dense W"
 02     02_kronecker.jl   an index is a number made of DIGITS x.reshape(t, 8, 12, 20)
 03     03_ring2.jl       rank LINKS that close a RING;      the R² in T1, T2;
                          cut the ring                       the kernel idea
 04     04_ring3.jl       three cores + tokens: the project  reference.py:122-124
                          in miniature
 05     05_machine.jl     an ABSTRACT MACHINE that predicts  the results table in
                          the benchmark table                REPORT_TEMPLATE.md
 06     06_gpu_machine.jl MEASURE the machine on a real GPU: the token-1 profiler
                          H, g, BW, F, eff; test the kernel  evidence; the harness's
                          formula on the reference's GEMMs   FP32 accumulation
 07     07_predict_calls.jl PREDICT whole calls, measure,    reference.py:122-124 as
                          LOCALISE the miss, repair the      GEMMs + permutes; the
                          model with one parameter           permute cost
 08     08_torch_machine.py (Python) PyTorch's host cost,    benchmark.py, its JSON and
                          the profiler, max(host, gpu) on    its token-1 trace
                          the harness's own numbers
```

Guides 06 and 07 need the GPU environment in `julia-gpu/` and run as
`julia --project=../julia-gpu <file>.jl` from `workspace/`. Guide 08 runs with the
project's own Python: `../../.venv/Scripts/python 08_torch_machine.py`.

```
   rung 1          rung 2              rung 3                rung 4               rung 5
  U·(V·x)   →   A⊗B on a grid   →   ring of 2 cores   →   ring of 3 cores   →   cost model
  order          digits              links, R², cut        the real shapes       time = max(host, gpu)
```

## How to use a guide

1. Read **Purpose** and the **mental model** picture first. Do not type yet.
2. For every step: read it, then **predict** the output, then type the block, then run.
3. Compare with **Expected output**. Where they differ, read **What failure looks like**.
4. Do the **Exercises** before moving on. They are where the model sets.

Every code block is labelled:

- **FILE — append to `name.jl`**: text that goes into a file, below what is already there.
- **COMMAND**: typed at a terminal prompt (PowerShell or Git Bash).

Each step appends to the same file, so the file grows and every run replays all earlier
steps. That is deliberate: you always see the whole story up to where you are.

## How these guides were checked

Every step was run in a scratch copy before it was written down, including each step on
its own (the file cut off after step 1, after step 2, and so on). Expected outputs are
copied from those runs.

- Machine: AMD Ryzen 7 5800H, Windows 11, Julia 1.13.0, OpenBLAS with 8 threads.
- GPU (guide 06 on): NVIDIA RTX 3050 Ti Laptop, 4 GB, driver 617.14 (CUDA 13.4),
  CUDA.jl 6.4, cuBLAS 13.8. Under Windows, so kernel launch costs are higher than on Linux.
- **Exact** outputs: sizes, counts, FLOPs, anything printed from a formula. Yours must match.
- **Approximate** outputs: errors like `9.09e-13` depend on the random numbers and BLAS;
  anything below `1e-10` is "equal".
- **Machine-dependent** outputs: times. Two runs on the same laptop moved by up to 2×
  (for example 760 µs vs 943 µs). Compare **ratios**, not absolute values.

## Folder layout when you are done

```
factorized-abstract-lab/
  README.md            this file
  00-setup.md … 08-torch-harness.md   the guides
  common.jl            you type it in guide 00
  01_lowrank.jl … 07_predict_calls.jl, 08_torch_machine.py   you type them in guides 01–08
  julia-gpu/           the CUDA.jl environment for guides 06–07 (already set up)
```
