# CUDA graphs, a chain of distinct layers, a cold L2

**Status:** script ready (`tools/graph_bench.py`, session `tools/h100_graphs.sh`), checked on
the laptop (correctness only). Predictions below were written **before** the H100 run
(2026-09-27).

## 1. The questions

The report's weak spots, in the order the researchers are likely to press on them:

| # | question | why it matters |
|---|---|---|
| 1 | Under CUDA graphs, who wins one layer? | Part of our t = 1 win is the shorter host path (G5 §6). Graphs remove the host path. |
| 2 | With many different layers, as in a model? | The harness re-reads one 11 MB W, which stays in the 50 MB L2 and flatters dense. |
| 3 | How much does dense lose with a cold L2? | The same question, one layer, kernel time only. |
| 4 | Is the ~32 MiB of the dense and reference rows the cuBLAS workspace? | The report says "probably, not verified". |
| 5 | Do the final numbers repeat on another machine? | Every earlier comparison was within one instance. |

## 2. How each is measured (all methods under the same rules)

- **single:** G = 20 back-to-back calls, (a) eager from Python, (b) captured in one CUDA
  graph and replayed. µs per call. The captured output is checked against the FP64 oracle.
- **chain:** L distinct instances of the assignment's operator (different random cores per
  layer, seeds 1000 + l), one after another in one stream, eager and in one graph. µs per
  layer. L = 1, 8, 32, 128; R = 8, 16; T = 1, 8. Every layer reads the same x: nodes captured
  from one stream run strictly in order, so this costs the same per layer as a
  data-dependent chain, and it needs no new "down" (2880 → 1920) kernel.
- **cold:** profiler kernel time of one call after a 256 MB read-modify-write (evicts L2),
  next to the same without the flush. The flush kernel is excluded by name.
- **cublas:** PyTorch-allocated bytes around the first `F.linear` and after
  `torch._C._cuda_clearCublasWorkspaces()`, in a fresh process. Laptop (sm_86): 8.125 MiB,
  freed by the clear call; the reference's einsum allocates the same; our kernel 0.012 MiB.

## 3. Predictions *(model, before measuring)*

| question | dense | ours B | ours A | reference | reasoning |
|---|---|---|---|---|---|
| single, graph, R8 t1 | 5.5–6.5 | 7.0–7.5 | 7–8 | 20–25 | kernels 5.0 / 6.7 / ~6 + gaps; graphs take the host path away, **dense likely wins** |
| single, graph, R16 t32 | 5–6 | ~120 | ~120 | ~200 | GPU-bound either way; unchanged |
| chain L ≥ 32, graph, R8 t1, µs/layer | 7–8 | 7.0–7.5 | 8–9 | 20–25 | stack study: cuBLAS graph chain 7.7 µs/layer with W from HBM; HBM floor 3.30 |
| chain L ≥ 32, graph, R8 t8 | 7.5–8.5 | ~22 | ~23 | — | ring arithmetic grows with t; dense wins |
| cold L2, R8 t1, kernel µs | 6.5–8 | 6.8–7.2 | — | — | 11 MB from HBM ≥ 3.3 µs; our 89 KB of cores barely matter |
| cuBLAS workspace, H100 | 32 MiB | 0 | 0 | 32 MiB | Hopper default `CUBLAS_WORKSPACE_CONFIG` = 4 MiB × 8 |
| repeat harness, R8 t1 call | 10.2–10.7 | 7.8–8.3 | — | — | three earlier instances: 10.3–10.7 / 7.9–8.0 |

**What would change the story:**

- chain graph with B ≤ dense at L ≥ 32: the ring layer is at least as fast as dense in a
  realistic setting **without** the host-path advantage and without a megakernel. That is
  the strongest honest claim available.
- chain graph with B clearly > dense (say > 9 µs): the t = 1 win is purely the host path,
  and the report must say so.

## 4. Results

H100 SXM5, one instance (the first of 2026-09-27, 217.177.40.33; `results/h100/graphs/`, `session.log`).
Code: the submitted kernel (commit `b0a73ed`, V3 at t = 1, WMMA kernel for t > 1), design B
unless marked. Every captured output passed the FP64 oracle at the harness tolerance.

### 4.1 cuBLAS workspace (`cublas.json`)

| | MiB |
|---|---|
| first `F.linear` (after dropping the workspace `materialize` created) | +32.0 |
| freed by `torch._C._cuda_clearCublasWorkspaces()` | 32.0 |
| first reference call (einsum → cuBLAS) | +32.0 |
| first call of our kernel (design B workspace + counters) | +0.012 |

**Verified:** the ~32 MiB in the dense and reference rows is the cuBLAS workspace (Hopper
default). Dense's resident 42.55 MiB = 10.5 MiB weight + 32 MiB workspace.

### 4.2 One layer, eager vs one CUDA graph (`single.json`), µs per call

| case | dense eager | dense graph | ours B eager | ours B graph | ours A graph | ref graph |
|---|---|---|---|---|---|---|
| R8 t1  | 9.91 | **5.23** | 8.25 | 6.96 | 7.29 | 21.23 |
| R8 t8  | 9.83 | **5.09** | 23.61 | 22.03 | 22.47 | 31.61 |
| R8 t32 | 9.53 | **5.39** | 49.37 | 47.89 | 47.00 | 60.80 |
| R16 t1 | 9.70 | **5.40** | 11.01 | 9.80 | 10.73 | 26.62 |
| R16 t32 | 9.67 | **5.66** | 124.99 | 121.96 | 120.74 | 199.76 |

- **Prediction held:** under graphs dense wins one layer at t = 1 (5.2 vs 7.0 µs at R8). Our
  call-level win in the required (uncaptured) table is the shorter host path, as the report
  says. Graphs cut dense by 4.7 µs and ours by 1.3 µs.
- The reference gains most from graphs (140 → 21 µs): it is host-bound, as profiled.

### 4.3 Kernel time, warm vs cold L2 (`cold.json`), µs per call

| case | dense warm | dense cold | ours B warm | ours B cold |
|---|---|---|---|---|
| R8 t1  | 4.97 | 7.87 | 6.84 | **7.17** |
| R8 t8  | 4.82 | 7.84 | 22.09 | 23.89 |
| R16 t1 | 4.93 | 7.79 | 9.81 | 10.22 |
| R16 t32 | 5.02 | 8.08 | 125.93 | 128.79 |

- Dense pays **+2.9 µs** when its 11 MB weight is not in L2 (7.9 µs: 1.4 TB/s, 42% of HBM
  peak). Ours pays +0.3–0.4 µs at t = 1.
- With a cold L2, our R8 t1 kernel (7.17) is **faster than dense's (7.87)**, kernel against
  kernel. The repeated-call benchmark hides this.

### 4.4 L distinct layers in one stream (`chain.json`), µs per layer

| R, t | method | eager L=128 | graph L=1 | graph L=8 | graph L=32 | graph L=128 |
|---|---|---|---|---|---|---|
| R8 t1 | dense | 9.38 | 9.22 | 7.12 | 6.81 | **6.73** |
| R8 t1 | ours B | **7.90** | 10.69 | 7.41 | 6.89 | 7.00 |
| R8 t1 | ours A | 11.70 | 10.91 | 7.68 | 7.31 | 7.57 |
| R8 t1 | ref | 140.08 | 25.18 | 21.36 | 21.11 | 21.11 |
| R16 t1 | dense | 9.36 | 8.80 | 7.13 | 6.80 | **6.69** |
| R16 t1 | ours B | 10.80 | 12.74 | 9.71 | 9.41 | 9.82 |
| R8 t8 | dense | 9.48 | 8.45 | 7.26 | 6.93 | 6.82 |
| R8 t8 | ours B | 24.43 | 25.41 | 22.14 | 21.80 | 23.14 |
| R16 t8 | dense | 9.28 | 8.42 | 7.21 | 6.94 | 6.83 |
| R16 t8 | ours B | 52.40 | 53.57 | 50.29 | 50.06 | 50.11 |

Dense HBM floor: 11.06 MB / 3.35 TB/s = 3.30 µs per layer.

- **R8 t1, graphs, 128 distinct layers: dense 6.73 vs ours 7.00, a tie within 4%.** Dense
  did better than predicted (7–8): 11 MB per 6.73 µs = 1.64 TB/s, 49% of HBM peak. Probably
  back-to-back GEMVs in a graph overlap one layer's tail with the next one's loads (not
  verified: that would need an Nsight Systems timeline).
- **Without graphs, 128 layers: ours 7.90 vs dense 9.38 (1.19× faster).** Eager is what the
  README's required table measures.
- R16 t1: dense wins by 1.47× under graphs. More tokens: dense wins by 3–7×.

### 4.5 README harness again, design B (`repeat_B/`), another machine

| case | dense | ours B | earlier final session (dense / ours) |
|---|---|---|---|
| R8 t1 | 9.90 | **8.28** | 10.3 / 7.9 |
| R8 t8 | 10.16 | 23.36 | 10.4 / 23.3 |
| R8 t32 | 9.71 | 49.73 | 10.6 / 49.6 |
| R16 t1 | 10.33 | 10.62 | 10.4 / 10.5 |
| R16 t32 | 9.77 | 126.23 | 10.3 / 126.2 |

Reproduces: R8 t1 1.2× faster than dense (was 1.3×), R16 t1 a tie.

### 4.6 What this changes in the report

1. The t = 1 win is a **host-path** win for one layer. Say it plainly; under graphs dense
   wins one layer (5.2 vs 7.0).
2. In the realistic setting (many distinct layers, weights not in L2, graphs) the ring at
   R8 **ties** dense (7.00 vs 6.73 per layer) while holding 124× less weight memory, and
   **wins** without graphs (7.90 vs 9.38).
3. The 32 MiB is verified cuBLAS workspace.
