# H100 runbook

How a paid session runs. Budget: **$25 at $3.5/h ≈ 7 GPU-hours**, and the meter runs from
the moment the instance exists until it is **deleted**.

## Who does what

```
   LAPTOP (WSL, ~/kernel-task)                      H100 (Verda, ~/kernel-task)
   ───────────────────────────                      ───────────────────────────
   you:    create instance, give Claude the login
   Claude: remote.sh sync ───── rsync working tree ──►  (code arrives, no GitHub needed)
   Claude: remote.sh run ... ── ssh ────────────────►  setup / check / measure / experiment
   Claude: remote.sh pull ◄──── rsync evidence ──────  results/h100 traces/h100 profiles/h100
   you:    git add / commit / push  (from the laptop)
   you:    DELETE the instance
```

The instance never talks to GitHub. Code goes up and evidence comes down with `rsync`, so
no keys or tokens are ever stored on it.

## Before creating the instance (free)

1. In the Verda console, add your **WSL** public key (`cat ~/.ssh/id_ed25519.pub` in the
   Ubuntu terminal) to the instance's SSH keys.
2. Pick an image with a recent NVIDIA driver (CUDA 13.0 or newer). The setup script checks
   this and stops early if the driver is too old: that costs a minute, not an hour.

## Session 1 plan (~1.5 h, ~$5)

| Step | Command (Claude runs it from WSL) | Time | Stop if |
|---|---|---|---|
| 0 | `export H100=root@<ip>`; `bash tools/remote.sh check` | 1 min | no GPU / wrong GPU |
| 1 | `bash tools/remote.sh sync` then `run bash tools/h100_setup.sh` | ~10 min | driver too old, build fails |
| 2 | (inside setup) `pytest -q`, `tools/check_kernel.py` | 1 min | **any** correctness failure: stop and fix on the laptop |
| 3 | `run bash tools/h100_session.sh` | ~25 min | — |
| 4 | `run python tools/h100_experiments.py --threads --stages --out results/h100/experiments.json` | ~15 min | — |
| 5 | `bash tools/remote.sh pull`; read the numbers together | 5 min | — |
| 6 | decide: tune now (kernel edits on the laptop, sync, re-measure) or delete and think | — | **no clear next step: delete the instance** |

Rule: **thinking happens with the instance deleted.** Debugging on the paid box only when
the fix is a quick edit and re-run.

## What each script produces

| File | What it answers |
|---|---|
| `results/h100/{A,B}/rank{8,16}.json` | the README's required harness runs, one set per design |
| `traces/h100/{A,B}/*.json` | token-1 profiler traces: where our kernel appears (open in https://ui.perfetto.dev) |
| `results/h100/kernels.json` | floors (empty call, empty kernel), kernel-only µs, % of FP16 Tensor Core peak and HBM peak, for ref / dense / A / B / torch.compile |
| `results/h100/sweep.json` | kernel time for each (kc, tt) tiling |
| `results/h100/experiments.json` | time with 128–1024 threads per block; time with each stage removed |
| `profiles/h100/*.ncu-rep` | Nsight Compute reports (only if the image ships `ncu`) |
| `results/h100/environment.txt`, `gpu.txt` | pip freeze, nvidia-smi, for the report |

## After the session (you, on the laptop)

```bash
cd ~/kernel-task
git add results/h100 traces/h100 profiles/h100
git commit -m "H100 session 1: harness A/B, kernel measurements"
git push
```

Then check the Verda console once more that the instance is **deleted**, not just stopped.
