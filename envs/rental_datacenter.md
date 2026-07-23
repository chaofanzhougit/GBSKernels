# Rental configuration — a first-class-FP64 GPU (V100 / A30 / A100 / H100 / P100)

The §9 benchmark sweeps **two** cards: the RTX 4090 (consumer, FP64 throttled ~1/64
FP32 — where the DD-vs-FP64 story is sharpest) and **one GPU with first-class FP64**
(~1/2 FP32), so the throughput and the FP64↔DD crossover are not a single-architecture
artifact. This is that second card.

**A100/H100 are NOT required** — the only requirement is *first-class FP64*. If A100/H100
are unavailable, a **V100 is the best substitute**: the same 1:2 FP64:FP32 ratio, abundant
and cheap on vast.ai / Lambda, fully supported (CUDA 12.4, `sm_70`). Any of these qualify:

| GPU | Arch | `gpu_session.sh <arch>` | FP64:FP32 | availability / cost |
|---|---|---|---|---|
| **V100** (recommended) | Volta `sm_70` | `70` | 1:2 | abundant, ~$0.2–0.6/hr |
| **P100** | Pascal `sm_60` | `60` | 1:2 | abundant, ~$0.1–0.4/hr |
| **A30** | Ampere `sm_80` | `80` | 1:2 | common, ~$0.3–0.7/hr |
| **A100** | Ampere `sm_80` | `80` | 1:2 | scarcer, ~$0.8–1.5/hr |
| **H100** | Hopper `sm_90` | `90` | 1:2 | scarce, ~$2–3/hr |

Avoid the FP64-**throttled** datacenter cards (A40, L40 / L40S, RTX A6000, RTX 6000 Ada —
all GA102/AD102, 1:64 like the 4090): they add nothing the 4090 doesn't already show.

Everything is wired — provision the instance below, give me the SSH string, and the session
runs identically to the 4090 (`scripts/gpu_session.sh` auto-detects the arch).

## vast.ai / Lambda / RunPod create-instance — exact field values

| Form field | Value |
|---|---|
| **Image Path:Tag** | `nvidia/cuda:12.4.1-devel-ubuntu22.04` (same as the 4090; a **devel** tag — has `nvcc` + headers) |
| **Launch Mode** | **SSH** (so the repo can be rsync'd up and `gpu_session.sh` run) |
| **On-start / Bash commands** | `apt-get update && apt-get install -y --no-install-recommends build-essential cmake ninja-build git python3 python3-dev python3-pip rsync ca-certificates` |
| **Extra Filters (V100, recommended)** | `verified=true rentable=true num_gpus=1 gpu_name=V100 cuda_max_good>=12.4 disk_space>=40 reliability>=0.98 inet_down>=200` |
| **Extra Filters (alternatives)** | …same with `gpu_name=` one of `P100` / `A30` / `A100` / `H100` |
| **Disk Space** | `40` (GB) |

Notes:
- **The on-start `apt` line is now optional**: `scripts/gpu_session.sh` self-bootstraps the
  build prereqs it needs if missing — **cmake / ninja / git / rsync** (devel images ship
  `nvcc`+`gcc` but often not these; the 2026-06-24 A100 box lacked cmake/ninja/pip) **and**
  pip + nanobind/numpy/mpmath/thewalrus. So a **bare** `*-devel` image works; setting on-start
  just front-loads the install. (`nvcc` itself can't be apt-installed — that's why the image
  must be a `-devel` tag.)
- **Arch is automatic**: V100 `sm_70`, P100 `sm_60`, A30/A100 `sm_80`, H100 `sm_90`. The
  session reads `nvidia-smi --query-gpu=compute_cap` and builds for it; force it with e.g.
  `bash scripts/gpu_session.sh 70` (V100) / `60` (P100) / `80` (A30/A100) / `90` (H100).
- CUDA 12.4 covers `sm_60`/`sm_70`/`sm_80`/`sm_90` (Pascal/Volta are deprecated-but-present in
  CUDA 12.x). H100 needs CUDA ≥ 12.0; V100/P100/A30 work on any 12.x devel image. The kernels
  use only FP64 + cuComplex + `__shfl_*_sync` (sm_60+), so every card above runs them.

## Get the repo there, then run everything in one command

Identical to the 4090 flow (`envs/rental_4090.md`), with the **pinned container digest**
captured so every artifact records it (frozen experiment — `docs/benchmark_protocol.md`):

```bash
# host: capture provenance, then push (excludes .git + local-only files)
git rev-parse --short HEAD > COMMIT_SHA
echo 'nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:<paste-digest-from-the-instance>' > CONTAINER_DIGEST
rsync -az --exclude .venv --exclude .git --exclude '__pycache__' \
      --exclude 'core/build' --exclude 'bindings/build*' --exclude '.pytest_cache' \
      --exclude 'private/' \
      ./  <user>@<box>:~/GBSKernels/

# box: one command -- CPU pre-flight -> nvcc build (sm_80/sm_90) -> all gates ->
#   kernel throughput -> accuracy (3 regimes) -> e2e -> sampler -> SAME-INSTANCE
#   The Walrus baseline. Self-bootstraps Python deps.
ssh <box> 'cd ~/GBSKernels && bash scripts/gpu_session.sh'      # or `... 80` / `... 90`

# back: copy the append-only artifacts (they carry GPU model + digest + commit)
rsync -az <box>:~/GBSKernels/results/ ./results/
# then, in-session or after: the batch-size crossover sweep (run on the box for real numbers)
ssh <box> 'cd ~/GBSKernels && python3 -m bench.crossover --batches 256,1024,4096,16384 --repeats 7'
```

The artifacts record `gpu.name` (A100/H100), driver, clocks, the container digest, and
the commit — directly comparable to the 4090's. That completes the two-card frozen
experiment for the paper.

## Cost expectation

V100 ≈ **$0.2–0.6/hr**, P100 ≈ **$0.1–0.4/hr**, A30 ≈ **$0.3–0.7/hr**, A100 ≈ **$0.8–1.5/hr**,
H100 ≈ **$2–3/hr**. The complete tagged workflow includes long CPU baselines and
sampler sweeps, so budget **3–4 hours**, not only the build/gate time. Keep an
independent watchdog and terminate only after `results/` is copied and verified.
