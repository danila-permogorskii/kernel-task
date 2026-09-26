# 00 — Setup and the measuring stick

## Purpose

Every later guide asks two questions of a computation: **how long does it take** and
**how much memory does it create**. This guide builds the two tools that answer them,
in a file every later guide includes.

This is the same job `benchmarks/benchmark.py` does in the project (`timed_call`, the
memory counters), reduced to ten lines.

## Mental model

```
   a computation f()
        │
        ├── best_time(f)    → run once to compile, then 30 times; keep the FASTEST
        │
        └── alloc_bytes(f)  → run once to compile, then count bytes allocated by ONE call
```

**Why the fastest and not the median?** Noise on an idle laptop only ever *adds* time:
a background process, a cache miss, a clock change. The minimum is the closest thing to
the true cost of the work. The project's harness reports the **median** instead, because
it wants typical behaviour including noise. Both are defensible; they answer different
questions. You made the same choice in `batch1-cdna/03-dlops` ("minimum of five runs,
since this is a latency floor").

**Why a warm-up call?** Julia compiles a function the first time it runs. The first call
measures the compiler, not the code. The project's harness has the same rule:
`first_call_ms` is recorded separately from the steady timings.

### In the studio

The lab tells every idea as a sound studio (README, "The studio"). Before any mixing, the
engineer needs two instruments:

```
   stopwatch     → best_time    how long one song takes through the desk
   tape counter  → alloc_bytes  how much new tape one song used up
                                (every intermediate track + the final mix)
```

The first song after switching the desk on is a sound check (the warm-up call); it is
never timed.

## Environment

**COMMAND**

```
julia --version
```

**Expected output**

```
julia version 1.13.0
```

Any 1.10 or later works. Only the standard library is used (`LinearAlgebra`, `Printf`,
`Random`); nothing needs installing.

**COMMAND** — create the folder and enter it (Git Bash shown; in PowerShell use `cd` the same way)

```
cd "C:/Users/79021/OneDrive - РУТ (МИИТ)/Рабочий стол/factorized-inference-assignment/factorized-abstract-lab"
```

Run every guide **as a fresh process** with `julia file.jl`. Do not `include` the files
into one long-lived REPL: the guides use `const`, and redefining a `const` with a new
value in the same session is an error.

## Step 1 — the helpers

**FILE — create `common.jl`**

```julia
# common.jl — measurement helpers shared by every guide
using Printf

"Call f once (compile + warm up), then return the fastest of `reps` runs, in seconds."
function best_time(f; reps = 30)
    f()
    best = Inf
    for _ in 1:reps
        best = min(best, @elapsed f())
    end
    return best
end

"Bytes allocated by one call of f, measured after a warm-up call."
function alloc_bytes(f)
    f()
    return @allocated f()
end

"Human-readable byte count."
human(b) = b < 1024   ? @sprintf("%d B", b) :
           b < 1024^2 ? @sprintf("%.1f KiB", b / 1024) :
                        @sprintf("%.1f MiB", b / 1024^2)

"Human-readable time from seconds."
us(s) = s < 1e-3 ? @sprintf("%.1f µs", s * 1e6) : @sprintf("%.2f ms", s * 1e3)
```

Line by line:

- `f` is any function with no arguments. You will pass anonymous functions like
  `() -> good(U, V, x)`.
- `@elapsed expr` runs `expr` and returns the seconds it took.
- `@allocated expr` runs `expr` and returns the bytes of heap memory it allocated. In
  these guides, **allocated bytes = intermediates + output that the computation had to
  create**. That is the "fridge traffic" from the project, made countable.
- `human` and `us` only format numbers.

**COMMAND** — a smoke test

```
julia -e 'include("common.jl"); println(human(3000), " ", us(2.5e-6))'
```

**Expected output**

```
2.9 KiB 2.5 µs
```

**What failure looks like**

- `could not open file ... common.jl`: you are not in the lab folder. `cd` into it.
- `UndefVarError: @sprintf`: the `using Printf` line is missing.

## Checklist

- [ ] `julia --version` works
- [ ] `common.jl` exists in the lab folder
- [ ] the smoke test prints `2.9 KiB 2.5 µs`
- [ ] I can say why a warm-up call is needed, and why min vs median is a choice
- [ ] I can say what the stopwatch and the tape counter measure in studio terms
