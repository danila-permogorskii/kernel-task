# K0: a Linux workbench and the CPU oracle

**Goal:** (1) the laptop behaves exactly like the H100 box: Linux, a clone from GitHub, one
venv; (2) a CPU file that computes `y` as a **sum of independent pieces**. Every kernel we
write later is checked against it.

**Time:** ~1 h. **Cost:** free (laptop).

```
   where K0 sits

   K0  workbench + oracle (CPU, FP64)       ◄── you are here
   K1  Triton: one piece, one stage at a time (laptop GPU)
   K2  Triton: all pieces, atomics = design A, through prepare_optimized, pytest
   K3  H100 session 1: measure the machine, run the harness
   K4  CUDA C++ v0 = design A ...  (Tensor Cores, tuning, B if time allows)
```

---

## Part 1: the workbench (WSL)

Why WSL: the H100 at Verda is Linux, so we debug the exact same commands here, for free.

```
   Windows copy (OneDrive)  ──git push──►  GitHub  ──git clone──►  WSL ~/kernel-task
        (stops being used                                         (your working copy
         after this step)                                          from now on)
```

### Step 1: push the new files from Windows

Two new files exist in the Windows copy: this guide and `kernel-design/DISCUSSION_POINTS.md`.
In the Windows project folder (Git Bash or PowerShell):

```bash
git add -A
git commit -m "K0 guide and discussion points"
git push
```

### Step 2: system packages in WSL

Open the Ubuntu terminal (`wsl -d Ubuntu-26.04`):

```bash
sudo apt update
sudo apt install -y python3.14-venv git
```

Checked: without `python3.14-venv`, `python3 -m venv` fails with "ensurepip is not available".

### Step 3: a GitHub key for WSL, then clone

WSL has its own home directory, so it needs its own SSH key.

```bash
ssh-keygen -t ed25519 -C "wsl-laptop" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

Copy the printed line to GitHub → Settings → SSH and GPG keys → New SSH key. Then:

```bash
ssh -T git@github.com            # expect: "Hi danila-permogorskii! You've successfully authenticated..."
cd ~
git clone git@github.com:danila-permogorskii/kernel-task.git
cd kernel-task
```

Why the home directory and not `/mnt/c/...`: files under `/mnt/c` go through a slow
Windows bridge, and the OneDrive path has spaces and Cyrillic letters.

### Step 4: the venv, the same way it will be done on the H100

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
pip install -e '.[test]'
```

On Linux, Triton comes with torch as a dependency; you do not install it separately.
Checked on the PyTorch index: `torch-2.14.0+cu132` and `triton-3.8.0` exist for Python 3.14 on Linux.

### Step 5: check the bench

```bash
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0))"
pytest -q
```

Expect:

```
2.14.0+cu132 3.8.0 NVIDIA GeForce RTX 3050 Ti Laptop GPU
21 passed
```

If `pytest` reports skips, CUDA was not seen: say so in chat before going on.

> *Not run by me:* steps 2–5 (they install into your WSL). The wheel availability and the
> `ensurepip` error were checked.

---

## Part 2: the oracle, one piece at a time

### The picture

The ring gives `y` as one big sum. We cut it into **pieces**, one per `(a, k)`:

```
   y[t,p,q,r] = Σ_a Σ_k  piece(a, k)[t,p,q,r]

   one piece (a, k):

   x[t, :, :, k] ──stage 1──► S1[t,j,p,b] ──stage 2──► S2[t,p,q,c] ──stage 3──► part of y[t,p,q,r]
                  × A[a]                    × B (all of it)           × C[:,:,k,a]
                  sum over i                sum over j, b             sum over c
```

Why this is legal: `a` and `k` are not summed until stage 3. Until then they just ride
along, so fixing them splits the work into independent problems.
*(Read later: Zhao 2016, §2 and Theorem 2.1, in `kernel-design/sources/`.)*

In studio terms: each piece is one **take**, and the final mix is the sum of all takes.

```
   how many takes?          R = 8:  8 × 20 = 160 pieces     R = 16: 16 × 20 = 320 pieces
   on the GPU:              one piece = one block of threads  (design A)
```

### Step 6: the skeleton and one piece

Create `kernel_work/pieces_cpu.py`:

```python
"""CPU oracle for the kernel: y as a sum of independent (a, k) pieces.

Every later kernel (Triton, CUDA v0, v1) is checked against piece_gemm().
"""
from __future__ import annotations

import torch

from factorized_inference import (
    TRSpec, make_cores, materialize_dense_weight, dense_forward,
)


# ---------- step 6: one piece, written as three einsums ----------------------
def piece_einsum(x, A, B, C, a, k):
    """Partial y[t, p, q, r] from one (a, k) piece. x is [T, ni, nj, nk]."""
    xk = x[:, :, :, k]                                   # [T, i, j]
    S1 = torch.einsum("tij,pib->tjpb", xk, A[a])         # stage 1: sum over i
    S2 = torch.einsum("tjpb,bqjc->tpqc", S1, B)          # stage 2: sum over j, b
    return torch.einsum("tpqc,cr->tpqr", S2, C[:, :, k, a])  # stage 3: sum over c
```

Read the three einsum strings against the picture above: each one names exactly the
indices that disappear (`i`, then `j b`, then `c`).

### Step 7: sum the pieces and compare with the dense oracle

Append:

```python
def forward_pieces(x2d, cores, spec, piece=piece_einsum):
    """Sum all R * nk pieces. x2d is [T, in_features] -> [T, out_features]."""
    A, B, C = cores
    ni, nj, nk = spec.input_modes
    P, Q, Rr = spec.output_modes
    T = x2d.shape[0]
    x = x2d.reshape(T, ni, nj, nk)
    y = torch.zeros(T, P, Q, Rr, dtype=x2d.dtype)
    for a in range(spec.rank):
        for k in range(nk):
            y += piece(x, A, B, C, a, k)
    return y.reshape(T, P * Q * Rr)


# ---------- check against the dense oracle -----------------------------------
def check(spec, tokens=3, seed=0, piece=piece_einsum):
    cores = tuple(c.double() for c in make_cores(spec, seed=seed))
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(tokens, spec.in_features, generator=g, dtype=torch.float64)
    expected = dense_forward(x, materialize_dense_weight(cores, spec))  # oracle only
    got = forward_pieces(x, cores, spec, piece)
    err = (got - expected).abs().max().item()
    print(f"{piece.__name__:13s} modes={spec.input_modes}->{spec.output_modes} "
          f"R={spec.rank:2d} pieces={spec.rank * spec.input_modes[2]:3d}  max|err|={err:.1e}")
    return err


if __name__ == "__main__":
    specs = [
        TRSpec((2, 3, 4), (5, 6, 7), 3),      # small, every mode different
    ]
    for piece in (piece_einsum,):
        for spec in specs:
            assert check(spec, piece=piece) < 1e-10
    print("all pieces agree with the dense oracle")
```

The dense W appears **only** inside `check()`: it is the oracle, which the rules allow.
The pieces themselves never build it.

Run it:

```bash
python kernel_work/pieces_cpu.py
```

Expect:

```
piece_einsum  modes=(2, 3, 4)->(5, 6, 7) R= 3 pieces= 12  max|err|=1.8e-15
all pieces agree with the dense oracle
```

The exact error may differ in the last digit on your machine; anything ~1e-15 is FP64 round-off.

**Why small modes first, all different:** if two modes were equal (say `ni = nj`), swapping
`i` and `j` by mistake would still give the right shape and could hide the bug.

### Step 8: break it on purpose

In `piece_einsum`, change `C[:, :, k, a]` to `C[:, :, k, 0]` and run again. You should see
the `assert` fail with a large error. Put it back.

This proves the check actually catches a wrong index. Do this once for every new oracle.

### Step 9: the real workload

In the `specs` list, add the two real cases:

```python
    specs = [
        TRSpec((2, 3, 4), (5, 6, 7), 3),      # small, every mode different
        TRSpec(rank=8),                       # the real workload
        TRSpec(rank=16),
    ]
```

Expect three lines, the new ones with `pieces=160` and `pieces=320`, errors ~1e-14.

---

## Part 3: the piece as the kernel sees it

A GPU does not run einsums. Inside a block, each stage will be a **2D matrix product**
`[M × K] @ [K × N]` (that is what Tensor Cores and Triton's `tl.dot` do). So we rewrite the
piece in exactly that form. This function *is* the blueprint of the kernel.

### The picture: three GEMMs, and the layouts chain

```
   stage 1    X [(t,j) × i]      @  A1 [i × (p,b)]       =  S1 [(t,j) × (p,b)]
                                                               │
                                          re-layout ◄──────────┘   (in the kernel: write S1
                                              │                     into shared memory in
                                              ▼                     the new order: free)
   stage 2    S1 [(t,p) × (j,b)] @  B2 [(j,b) × (q,c)]   =  S2 [(t,p) × (q,c)]
                                                               │
                                          same bytes, read as ─┘   [(t,p,q) × c]: free
                                              │
                                              ▼
   stage 3    S2 [(t,p,q) × c]   @  C3 [c × r]           =  Y  [(t,p,q) × r]  = y[t,p,q,r]

   sizes for R = 8, T tokens:      M          K          N
                     stage 1     12·T         8         96
                     stage 2     12·T        96         80     ◄── ~74% of all FLOPs
                     stage 3    120·T         8         24
```

In the reference, each "re-layout" is a separate copy through HBM (a separate GPU kernel).
In our kernel it costs nothing: the result is simply *written* in the order the next stage
reads it. In studio terms: the reference records to tape and re-spools between desks; we
route the cable straight to the next desk's input.

`A1`, `B2`, `C3` depend only on the cores, so `prepare_optimized` builds them **once**
(same number of values as the cores: packing, not a dense W).

### Step 10: write `piece_gemm`

Insert this **above** the `# ---------- check` line:

```python
# ---------- step 10: the same piece as three 2D matrix products --------------
def piece_gemm(x, A, B, C, a, k):
    """Same result as piece_einsum, but every stage is  [M x K] @ [K x N]."""
    T, ni, nj, nk = x.shape
    R, P, _, _ = A.shape
    _, Q, _, _ = B.shape
    _, Rr, _, _ = C.shape

    # stage 1:  X[(t,j), i] @ A1[i, (p,b)]  ->  S1[(t,j), (p,b)]
    X = x[:, :, :, k].permute(0, 2, 1).reshape(T * nj, ni)
    A1 = A[a].permute(1, 0, 2).reshape(ni, P * R)
    S1 = X @ A1

    # "write it where the next stage reads it":  [(t,j),(p,b)] -> [(t,p),(j,b)]
    S1 = S1.reshape(T, nj, P, R).permute(0, 2, 1, 3).reshape(T * P, nj * R)

    # stage 2:  S1[(t,p), (j,b)] @ B2[(j,b), (q,c)]  ->  S2[(t,p), (q,c)]
    B2 = B.permute(2, 0, 1, 3).reshape(nj * R, Q * R)
    S2 = S1 @ B2

    # free reshape: [(t,p),(q,c)] is already [(t,p,q), c] in memory
    S2 = S2.reshape(T * P * Q, R)

    # stage 3:  S2[(t,p,q), c] @ C3[c, r]  ->  Y[(t,p,q), r]  == y[t, p, q, r]
    C3 = C[:, :, k, a]
    Y = S2 @ C3
    return Y.reshape(T, P, Q, Rr)
```

How to read a `permute(...).reshape(...)` line: `permute` puts the indices in the order
you want the rows and columns to be; `reshape` then glues neighbours together. For `X`:
`x[:, :, :, k]` is `[t, i, j]` → permute to `[t, j, i]` → glue `(t, j)` into rows, `i`
stays the column.

### Step 11: check the blueprint

Change the loop in `__main__` to run both pieces:

```python
    for piece in (piece_einsum, piece_gemm):
```

Run. Expect six lines (the three from step 9, then the same three for `piece_gemm`),
all errors ~1e-14, then `all pieces agree with the dense oracle`. On my run:

```
piece_gemm    modes=(2, 3, 4)->(5, 6, 7) R= 3 pieces= 12  max|err|=1.8e-15
piece_gemm    modes=(8, 12, 20)->(12, 10, 24) R= 8 pieces=160  max|err|=9.8e-15
piece_gemm    modes=(8, 12, 20)->(12, 10, 24) R=16 pieces=320  max|err|=8.0e-15
```

### Step 12: save your work

```bash
git add kernel_work/pieces_cpu.py
git commit -m "K0: CPU piece oracle (einsum and 3-GEMM forms)"
git push
```

---

## ▶ YOUR CALL (answer in chat before K1)

**1. Where does S1 get re-laid out?** In the kernel, S1 lives in shared memory between stage
1 and stage 2. Either:

```
   (a) stage 1 WRITES S1 in [(t,p) × (j,b)] order        stage 2 reads it straight
   (b) stage 1 writes S1 in its natural [(t,j) × (p,b)]   stage 2 reads it with a stride
```

Which would you pick, and why? Hint: think about which side is simpler to get right in
Triton, where you describe *blocks* of indices, not single addresses.

**2. Stage 1 has K = 8 (sum over `i`).** Tensor Cores and `tl.dot` want K ≥ 16. Options:
pad `i` with zeros up to 16 (half the work is wasted, but stage 1 is only ~7% of FLOPs), or
do stage 1 without `tl.dot` (plain multiply-adds). Your instinct?

There is no wrong answer here; K1 will measure whichever you pick.
