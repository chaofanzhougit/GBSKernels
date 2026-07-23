# Rental configuration — single RTX 4090 (smoke + measurement in one box)

One Ada 4090 covers everything: the kernels are tiny (~17 MB working set for a
4096×16×16 batch) so correctness gates, throughput, the certified-numerics
tightness figure, sampler sweeps, and same-instance crossover run on one GPU.
Budget about three hours after SSH is ready, plus startup and result-retrieval
margin; the complete 2026-07-23 v0.2.0 run took about three hours on-device.
The 4090 posts the *faster* `tor_single` frontier (26 clicks 0.74 s).

## Turnkey: one command from access to results

Once the instance is up (spec below), the whole session is driven from the
workstation by `scripts/launch_session.sh` — it resolves + pins the container
digest, rsyncs the tree (excluding `private/`, `.git`,
`data/`), runs `gpu_session.sh` **detached under nohup** (so a dropped local
connection can't kill the run), streams the log, and copies `results/` back:

```bash
git status --short && git rev-parse HEAD   # confirm the intended clean commit
bash scripts/launch_session.sh -p <port> <user>@<host> 89
```

`89` = sm_89 (Ada 4090). That single session now runs, in order: CPU pre-flight
→ nvcc build → **all differential gates** (FP64 + DD + certified + certified
sieve incl. the M=32 PNR-base case + recursive tor + session) → build+smoke the
nanobind extension → kernel-only throughput (incl. `tor_single` frontier) →
GPU-vs-mpmath accuracy → 3-regime e2e → sampler throughput + sweep + R4 A/B +
Walrus baseline + crossover → **tightness distributions**. Borealis validation
is a separate workflow; `gpu_session.sh` does not download or run that dataset.

Everything self-reports the pinned commit + container digest. When it finishes,
verify and copy `results/`, then **terminate the instance manually**. The launcher
does not call a provider API or propagate the detached runner's exit code, so
check the remote `_session.log` for `=== DONE ===` and no `[ABORT]` first. Use an
independent provider-side or local watchdog while the detached job is billing.

## vast.ai create-instance form — exact field values

| Form field | Value |
|---|---|
| **Image Path:Tag** | `nvidia/cuda:12.4.1-devel-ubuntu22.04` |
| **Docker Options** | *(leave empty/default)* |
| **Launch Mode** | **SSH** (so the repo can be rsync'd up and `gpu_session.sh` run) |
| **On-start / Bash commands** | `apt-get update && apt-get install -y --no-install-recommends build-essential cmake ninja-build git python3 python3-dev python3-pip rsync ca-certificates && pip3 install --break-system-packages nanobind numpy mpmath` |
| **Extra Filters** | `verified=true rentable=true num_gpus=1 gpu_name=RTX_4090 cuda_max_good>=12.4 disk_space>=50 reliability>=0.98 inet_down>=200` |
| **Disk Space** | `50` (GB) — CUDA devel image, dependencies, build products, results, and headroom |

Notes:
- The image is a **devel** tag (has `nvcc` + headers + `g++`); a *runtime*/*base*
  tag would not compile. `cuda_max_good>=12.4` ensures the host driver supports
  the 12.4 toolkit. If offers are scarce, drop both to **12.2**
  (`nvidia/cuda:12.2.2-devel-ubuntu22.04` + `cuda_max_good>=12.2`) — the code only
  needs CUDA ≥ 11.8 (first with Ada `sm_89` support), nothing 12.4-specific.
- On-start installs what the session needs: `make`/`g++` (build-essential) for
  cmake + the CPU pre-flight, `cmake`/`ninja`, `git`, `python3`, `rsync`, and the
  Python deps `nanobind numpy mpmath` (+ `thewalrus` for the same-instance
  baseline). `gpu_session.sh` also self-heals these if the image lacks them.
- The kernel-only throughput driver is stdlib-only, but the **accuracy study,
  sampler, and tightness figure need numpy + mpmath + the built nanobind
  extension** — all handled on-box; no
  `uv sync` required (the session exports `GBSKERNELS_EXT_DIR=bindings/build`).
- Official v0.2.0 bindings require Python 3.12 or newer. Ubuntu 22.04's default
  Python 3.10 is sufficient for the C++ gates but not for a release-grade Python
  smoke; install Python 3.12 or use a maintained image that already provides it.

## Instance spec

| Field | Value | Why |
|---|---|---|
| **GPU** | 1× NVIDIA RTX 4090 (Ada, **compute capability 8.9 → `sm_89`**) | the target card; FP64 is throttled (~1/64 FP32) but that is *intended* — the DD-vs-FP64 story is most relevant on consumer cards |
| **GPU count** | 1 | no multi-GPU / NVLink (docs/DESIGN.md §4) |
| **CUDA toolkit** | **≥ 11.8, recommend 12.x** (devel) | 11.8 is the first toolkit with Ada/`sm_89` support; we need **nvcc** (a *devel* image, not runtime/base) |
| **Host driver** | matched to the toolkit: ≥ 520 (CUDA 11.8), ≥ 535 (12.2), ≥ 550 (12.4) | the container uses the host driver; pick a toolkit ≤ what the host driver supports |
| **OS** | Ubuntu 22.04 (or 20.04) x86-64 | matches the toolchain; the code is portable C++17/CUDA |
| **Disk** | **≥ 30 GB** | CUDA devel image (~8–10 GB) + build + headroom |
| **RAM** | ≥ 8 GB | building + the Python driver; compute is tiny |
| **vCPU** | ≥ 4 | parallel `cmake --build -j` |
| **Access** | SSH enabled (key) | so the repo can be rsync'd up and the session driven/run |

## Launch template (the one field that matters)

Pick a **CUDA devel** image as the instance template — it ships `nvcc` + headers:

```
nvidia/cuda:12.4.1-devel-ubuntu22.04
```

(or `12.2.2-devel-ubuntu22.04` / `11.8.0-devel-ubuntu22.04` if the host driver is
older — all build `sm_89` fine; the code has no hard CUDA-12 dependency). This is
the same base as `envs/Dockerfile`.

## One-time setup on the box (if the image lacks them)

The CUDA devel image has nvcc + a C++ compiler; add cmake, git, and uv:

```bash
apt-get update && apt-get install -y --no-install-recommends cmake ninja-build git ca-certificates curl
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
```

## Get the repo there, then run everything in one command

```bash
# from the host workstation: push the tree to the box. EXCLUDE local-only files
# (rsync doesn't honor .gitignore) and the build dirs. Capture the commit
# AND the pinned container digest first (the box has no .git) so EVERY artifact
# records both (frozen, reproducible experiment -- docs/DESIGN.md §9).
git rev-parse --short HEAD > COMMIT_SHA
# the digest of the EXACT image you launch the instance from (see "Image Path:Tag"
# above) -- e.g. `docker buildx imagetools inspect <image> --format '{{.Manifest.Digest}}'`
# or copy it from the vast.ai instance details. Pins the toolchain.
echo 'nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:<paste-digest>' > CONTAINER_DIGEST
rsync -az --exclude .venv --exclude .git --exclude '__pycache__' \
      --exclude 'core/build' --exclude 'bindings/build*' --exclude '.pytest_cache' \
      --exclude 'private/' \
      ./  <user>@<box>:~/GBSKernels/

# on the box: gpu_session.sh exports GBS_COMMIT from COMMIT_SHA and installs
# numpy/mpmath if absent (so the accuracy + real-device e2e artifacts are produced).
cd ~/GBSKernels && bash scripts/gpu_session.sh                 # auto-detects sm_89
# -> CPU pre-flight -> nvcc build -> ALL differential gates (FP64 + DD + coop + warp
#    + host_api + session) -> build+smoke the nanobind extension -> kernel-only
#    throughput -> GPU-vs-mpmath accuracy (FP64 AND DD) + public-path e2e throughput
# -> writes results/ ; copy it back:
rsync -az  <user>@<box>:~/GBSKernels/results/  ./results/
```

## Cost expectation

4090 on-demand offers are commonly **$0.30–0.70/hr**. Budget **3–4 hours** for
the full tagged workflow plus startup/retrieval, or roughly **$0.90–2.80**. The
2026-07-23 v0.2.0 session cost $1.40 at $0.3472/hr including provisioning and
retrieval. The real cost risk is leaving the box running: keep a watchdog and
terminate only after a checksum-verified result copy.

## Provider field-name cheat-sheet

- **vast.ai**: "Docker image" = the CUDA devel tag above; "Disk Space" ≥ 30 GB;
  filter GPU = "RTX 4090", count 1; enable SSH.
- **RunPod**: pick a "CUDA 12.4 devel" / Ubuntu template or set a custom image;
  Container Disk ≥ 30 GB; GPU = "1× RTX 4090"; deploy with SSH.
- **Lambda / others**: choose a 4090 instance with a CUDA 12.x image; ensure nvcc
  is present (`nvcc --version`); ≥ 30 GB disk.
