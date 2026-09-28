# A100 runbook: the kernel on Qwen3.8-27B shapes, BF16

Same flow as [H100-RUNBOOK.md](H100-RUNBOOK.md) (Claude drives over SSH, the instance never
talks to GitHub, you commit from the laptop, you delete the instance). What is new:

- **Shapes:** the six distinct linear layers of Qwen3.8-27B (`tools/qwen_shapes.py`, checked
  against `kernel-design/physics/qwen3.8-27b_config.json`). The weights are random: these runs
  measure speed, memory and numerics on real sizes, not the quality of a compressed model.

  ```
  mlp_gate_up    5120 -> 17408  x128   modes (20,16,16) -> (16,34,32)
  mlp_down      17408 ->  5120  x 64   modes (34,32,16) -> (16,16,20)
  attn_q_gate    5120 -> 12288  x 16   modes (20,16,16) -> (16,24,32)
  attn_kv        5120 ->  1024  x 32   modes (20,16,16) -> (8,8,16)
  o_proj         6144 ->  5120  x 64   modes (24,16,16) -> (16,16,20)
  gdn_qkvz       5120 -> 16384  x 48   modes (20,16,16) -> (16,32,32)
  ```
- **BF16:** the generic WMMA kernel is now a template on the element type (FP16 / BF16).
  V3 / V3T stay FP16 on the assignment's modes, so on Qwen shapes **only the generic kernel
  runs** — the one the assignment used before V3.
- **Tiling sweep on the card:** `choose_tiling` was written for the small assignment modes. On
  the real shapes it often picks qc = 1 (stage 1 recomputed for every q). The session measures
  every tiling that fits and re-runs ours with the fastest one.

## Session plan (~1 h)

| Step | Command (Claude runs it from WSL) | Time | Stop if |
|---|---|---|---|
| 0 | `export GPU_HOST=root@<ip>`; `bash tools/remote.sh check` | 1 min | no GPU / not an A100 |
| 1 | `bash tools/remote.sh sync`, then `run bash tools/h100_setup.sh` (GPU-independent) | ~10 min | driver older than CUDA 13, build fails |
| 2 | `run bash tools/qwen_session.sh` (correctness first, then everything) | ~50 min | any `FAIL` in step 1 of the log |
| 3 | `bash tools/remote.sh pull`; read `results/a100/qwen/summary.md` together | 5 min | — |
| 4 | you: delete the instance | — | — |

Switches if time is short: `SKIP_SWEEP=1` (−25 min), `SKIP_HARNESS=1`, `SKIP_NCU=1`,
`TOKENS=1,8,32`.

## What the files answer

| File | Question |
|---|---|
| `results/a100/qwen/check_bf16.txt` | Is BF16 correct on every shape, R = 8 and 16, designs A and B? Our error next to dense BF16's |
| `results/a100/qwen/bf16.json` | Harness protocol: dense vs torch reference vs ours, T = 1, 8, 32, 128 |
| `results/a100/qwen/sweep.json` | How much the tiling rule leaves on the table, per shape / R / T |
| `results/a100/qwen/bf16_tuned.json` | Ours with the best measured tiling |
| `results/a100/qwen/summary.md` | The tables: µs, dense / ours, dense GB/s (% of HBM), our TFLOP/s, errors, memory, one decode step |
| `results/a100/harness_fp16/` | The assignment's README commands on the A100 (for the A100-vs-H100 comparison) |
| `profiles/a100/qwen_*.ncu-rep` | Where the time goes in the generic kernel on the biggest layer |

## What the laptop already says (RTX 3050 Ti — correctness only, timings are indicative)

BF16 is correct everywhere (`tools/check_bf16.py`: our relative L2 error ≈ 2.9e-3, dense BF16
≈ 3.1e-3, so the error is the BF16 output rounding). At T = 1 the ring beats dense by moving
~1000x fewer bytes. For T ≥ 8 the generic kernel is **slower than the torch reference** on
these shapes: per-block redundancy (stage 1 recomputed per q chunk, B reloaded per block)
that was cheap on the 1920 -> 2880 assignment layer grows with the modes. Expect the A100 to
show the same shape of result; the sweep says how much of it is tiling and how much is design.

## After the session (you, on the laptop)

```bash
cd ~/kernel-task
git add src tools kernel-guides results/a100 traces/a100 profiles/a100
git commit -m "Qwen3.8-27B shapes in BF16: kernel template, bench/sweep/session tools, A100 results"
git push
```
