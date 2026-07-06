# Benchmark protocol — a frozen, reproducible experiment (docs/DESIGN.md §9)

The benchmark is an *experiment*, not a number: any result must be reproducible from
its artifact alone. This document is the protocol the paper points to. Each piece is
built CPU-first and validated; the device numbers are produced in scripted rented-GPU
sessions (`scripts/gpu_session.sh`), never on a shared CI runner.

## 1. What every artifact records (provenance)

`bench/_provenance.py` stamps **every** artifact — kernel throughput, public-path
end-to-end, accuracy, sampler, the Walrus baseline, the crossover sweep — with the same
block, so a result is reproducible from the file alone:

* `commit` — `git rev-parse --short HEAD` of the exact tree that ran.
* `container_digest` — the **pinned image digest** the instance was launched from
  (`nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:…`); fixes the toolchain.
* `hostname`, `captured_utc`.

The rented box has no `.git` (rsync excludes it) and the image is digest-pinned, so the
host captures both into `COMMIT_SHA` / `CONTAINER_DIGEST` before upload and
`gpu_session.sh` exports them as `GBS_COMMIT` / `GBS_CONTAINER_DIGEST` (see
`envs/rental_4090.md`). Missing values are recorded as `null`, never faked.

## 2. Warm-up policy

The large cold-start IQRs must be understood before any number is a headline. Policy:
**each timing cell runs untimed warm-up first** (steady-state GPU clocks, primed caches
/ kernel JIT), then the timed repeats report **median + IQR** (never best-of-N as the
headline). Implemented in `bench_kernels.cu` (2 untimed launches/cell),
`bench.throughput_end_to_end` (`--warmup`, recorded in `params`), `bench.sampler_throughput`,
`bench.walrus_baseline`, and `bench.crossover`. The IQR is reported, not hidden; if it is
large after warm-up, that is itself the finding.

## 3. Input regimes (three, all realistic-to-adversarial)

`bench/_inputs.py` exposes **one shared generator**, `bench_batch(func, dim, batch, regime,
seed)`; `throughput_end_to_end`, `walrus_baseline`, and `crossover` all draw from it with the
same seed convention, so **GPU, CPU, and The Walrus are timed on identical matrices** per
`(func, dim, regime)`. The three regimes per function:

* **physical** — pure, well-conditioned (Haar interferometer; `B = U tanh(r) U^T`;
  small-norm real `O`). FP64 is accurate, DD agrees.
* **loss / mixed-state** — matrices from a Gaussian state after a uniform loss channel
  (mixed: `det(Q) > 1`): the A-matrix block `X(I − Q⁻¹)` for the (loop) hafnian, `O = I −
  Q⁻¹` for the torontonian. The regime a real lossy experiment produces. (The permanent's
  loss analog is a *sub-unitary* — the top-left block of a larger Haar unitary, i.e. the
  linear map of a lossy interferometer.)
* **adversarial** — a tunable cancellation family that drives the FP64 error up while DD
  holds near machine precision: this is the *measured* FP64↔DD boundary (§6).

## 4. The GPU set (sweep two cards, not one)

Run the session on **two** GPUs across runs so the result is not a single-card artifact:

* **RTX 4090** (Ada, `sm_89`) — the consumer card where FP64 is throttled (~1/64 FP32);
  the DD-vs-FP64 story is most relevant here. (`envs/rental_4090.md`.)
* **One first-class-FP64 GPU** — where FP64 is ~1/2 FP32 (vs the 4090's ~1/64). **A100/H100
  are not required**; a **V100** (`sm_70`) is the cheapest, most-available such card and the
  recommended choice — **P100** (`sm_60`), **A30** / **A100** (`sm_80`), **H100** (`sm_90`)
  all qualify (full table + cost in `envs/rental_datacenter.md`). Provision a CUDA **devel**
  image (record its digest) and `bash scripts/gpu_session.sh <arch>`; the bench records the
  GPU model/driver/clocks via `nvidia-smi`, so the two cards' artifacts are directly
  comparable. Any first-class-FP64 card makes the point that the FP64↔DD crossover is not a
  single-architecture artifact; **this is the one piece that needs renting a second GPU.**

## 5. Same-instance The Walrus baseline

The comparison must be apples-to-apples on **one** machine. `bench.walrus_baseline` times
The Walrus (the canonical CPU reference) for the four functions on the **same instance**
as our kernels, with the identical hygiene (warm-up, median + IQR, randomized order,
provenance). It draws from the **same shared generator** as `throughput_end_to_end` and
`crossover` (§3), so the GPU, CPU, and Walrus numbers are **same-input**; `gpu_session.sh`
installs `thewalrus` and runs it. This fixes the "our GPU vs The Walrus on a laptop"
apples-to-oranges gap; the crossover (§6) then ties each point to its achieved accuracy.

## 6. Batch-size sweep + crossover figures

`bench.crossover` sweeps the public end-to-end path over batch sizes
(`256,1024,4096,16384,…`) and records the GPU per-eval rate against two ~flat baselines on
the **same shared workload** — our CPU reference and the **same-instance The Walrus**
(measured once; one-at-a-time, so batch-independent) — with the **crossover batch** reported
**vs CPU and vs Walrus** (the smallest batch where the GPU median overtakes each). The GPU's
fixed H2D/launch/D2H overhead means small batches favour the baselines; large batches favour
the GPU. Each series is tagged with its **achieved FP64 error** (vs mpmath) and **precision
tier**, so a throughput point never travels without the accuracy it was bought at.
`bench.plot_crossover` renders the log-log curves per function (matplotlib; emits CSV without
it). No composite "winner" score; raw per-cell data is retained.

**Honesty gate.** `throughput_end_to_end` cross-checks that the GPU and CPU backends agree
(FP64 tolerance) on every well-conditioned (physical) cell and **exits non-zero on a
disagreement**; `gpu_session.sh` aborts the session, so no timing is ever published from a
backend mismatch (a kernel/binding bug). Adversarial inputs may legitimately diverge and are
an accuracy-study input, not a gate.

## 7. Reproducing a run

```bash
# host: capture provenance, push (excludes .git + local-only files)
git rev-parse --short HEAD > COMMIT_SHA
echo '<image>@sha256:<digest>' > CONTAINER_DIGEST
rsync -az --exclude .git --exclude .venv --exclude 'core/build' --exclude 'bindings/build*' \
      --exclude 'private/' \
      ./ <user>@<box>:~/GBSKernels/
# box: one command -- CPU pre-flight -> nvcc build -> all gates -> kernel throughput
#      -> accuracy (3 regimes) -> e2e -> sampler -> same-instance Walrus baseline
ssh <box> 'cd ~/GBSKernels && bash scripts/gpu_session.sh <arch>'   # 89=4090, 80=A100, 90=H100
rsync -az <box>:~/GBSKernels/results/ ./results/          # append-only artifacts
uv run python -m bench.crossover --batches 256,1024,4096,16384   # (on the box, in-session)
uv run python -m bench.plot_crossover results/throughput/crossover_*.json -o results/fig/
```

Every artifact under `results/` is append-only and self-describing (provenance + params +
raw rows). A reviewer re-launches the pinned image at the recorded commit and re-runs.
