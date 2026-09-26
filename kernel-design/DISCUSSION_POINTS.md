# Discussion points for the team

Points to raise in the 45-minute discussion. Each one is a decision plus its consequences.

---

## 1. How the pieces are added: 3 launches (A) vs 1 launch (B)

**Decision (2026-09-26):** build **A** first. Build **B** only if H100 hours or time remain.

```
   A: atomics + convert                    B: "last block finishes the job"
   ┃ memset ┃ kernel ┃ convert ┃           ┃ kernel ... last block converts + clears ┃
   ≈ 3 launch floors at t = 1              ≈ 1 launch floor at t = 1
   stateless: every call independent       persistent workspace + counter between calls
```

### What other teams would see in production

Assumed model for scale: 32 layers × 4 factorized linears = 128 calls per decoded token.
A = 384 launches per token, B = 128. Saving ≈ 256 × ~2 µs ≈ 0.5 ms per token without CUDA
graphs, or about 2–5% of a 10–30 ms decode step. *(estimate, to be measured)*

| Team | A (3 launches) | B (1 launch, persistent state) |
|---|---|---|
| Serving / runtime | more launches and CPU per token; **stateless**, safe on many streams | lower decode latency; **one workspace per stream required**, since shared state breaks multi-stream, overlapped micro-batches and parallel speculative decoding |
| Reliability / on-call | failures are loud (crash or error) | an aborted call leaves the counter dirty, so **the next call is silently wrong**; needs a reset path and a health check |
| Kernel maintainers | three simple kernels | memory-ordering code (fence + counter): bugs under load or on new GPUs; needs stress tests and an expert reviewer |
| ML / evaluation | atomics are nondeterministic (last bits vary) | same |
| Compiler (`torch.compile`) | pure function, easy to register | hidden mutable state must be declared, or calls may be reordered or merged |
| Capacity / finance | the big lever is compression: 30–120× less weight memory → bigger batch / KV cache / cheaper GPUs | same, plus a few % of decode throughput |

### Business conclusion

```
   value ▲
         │ ██████████████  compression itself: memory freed → cost per token ↓↓
         │ ████            one custom op instead of 8 kernels + 3 einsums
         │ █               A → B: a few % of decode latency (less under CUDA graphs)
         └────────────────────────────────────────────────►
                           risk / engineering cost:  A low     B medium–high
```

- B's gain is **second-order**, and it shrinks further under CUDA graphs, which production
  servers normally use.
- B's risk is **first-order**: a silent wrong answer costs more than a few % of throughput.
- Production path: ship A; put B behind a flag with a stress test and a workspace per
  stream; promote B only if measurements *in the full system, with CUDA graphs* still show
  a gain. If exact repeatability is a product requirement, use the deterministic
  partials option instead.
- This answers the researchers' question: *what would you measure before claiming an
  improvement in a complete inference system?*

---

## 2. Which case is the headline

- **t = 1 is the headline** (decode, the realistic regime). Real work is ~0.04 µs, against
  a fixed launch cost of ~2–5 µs and a Python call of ~10–30 µs. No hardware ceiling is
  reachable; the metric is **µs above the fixed floors** (an empty kernel, an empty op).
- **R = 16, t = 32 is the throughput case**: ~13,700 FLOP/byte, far past the H100 ridge
  (~295), so compute-bound. Metric: **% of FP16 Tensor Core peak**. Dense reads 11 MB in
  ~3.3 µs; we need ≥ 9 µs even at 100% of peak, so **dense wins, and we can prove why**.
