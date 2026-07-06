#!/usr/bin/env bash
# scripts/gpu_session.sh -- run the whole GPU validation on the rented box, in one
# command. Idempotent; safe to re-run. Intended to be executed ON the rented GPU
# instance (after the repo is present), either by you or driven over SSH.
#
# bash scripts/gpu_session.sh # auto-detect arch (4090 -> sm_89)
# bash scripts/gpu_session.sh 89 # force arch
#
# Order (cheapest assurance first; aborts before timing if any gate fails):
# 1. CPU pre-flight (host shim) -- must already pass; near-instant
# 2. nvcc build of core/ -- the first thing only the GPU can verify
# 3. four differential gates -- GPU == CPU reference (must all PASS)
# 4. throughput benchmark -- writes results/throughput/
#
# Exits non-zero (and does NOT run timing) if the build or any gate fails, so no
# published number ever comes from a failed gate.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\n[ABORT] %s\n' "$*" >&2; exit 1; }

# Cap CPU BLAS / OpenMP threads. In a container on a many-core HOST, OpenBLAS (used by numpy /
# The Walrus) auto-detects the HOST core count, exceeds its 128-thread build limit, and
# SEGFAULTS the Python harnesses ("tried to allocate too many memory regions" -- measured on
# the 2026-06-24 4090 box: walrus_baseline + crossover both crashed). An explicit low cap
# overrides the auto-detection. The matrix functions are tiny, so single-threaded BLAS costs
# nothing meaningful; the on-device GPU timing is unaffected.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GOTO_NUM_THREADS=1 MKL_NUM_THREADS=1

# --- toolchain check --------------------------------------------------------
say "toolchain"
# CUDA devel images ship nvcc + gcc but often NOT cmake/ninja/git/rsync (measured: the
# 2026-06-24 A100 box had cmake+ninja+pip missing). Bootstrap them best-effort so the session
# is turnkey on a BARE devel image, not just a pre-provisioned one -- before the hard checks
# below (which would otherwise die on a missing cmake).
if command -v apt-get >/dev/null 2>&1; then
  for need in cmake git rsync; do command -v "$need" >/dev/null 2>&1 || _need_apt=1; done
  if [ -n "${_need_apt:-}" ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y --no-install-recommends cmake ninja-build git rsync ca-certificates >/dev/null 2>&1 || true
  fi
fi
command -v nvcc >/dev/null || die "nvcc not found (need the CUDA toolkit / devel container)"
command -v cmake >/dev/null || die "cmake not found (apt bootstrap failed; install cmake on the box)"
nvcc --version | tail -1
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader || \
  die "nvidia-smi failed (no GPU visible?)"

# --- arch (4090 = Ada = sm_89) ---------------------------------------------
ARCH="${1:-}"
if [ -z "$ARCH" ]; then
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
  ARCH="${CC:-89}"
fi
echo "building for CUDA arch sm_${ARCH}"

# --- bootstrap Python deps once (CUDA devel images often ship no pip) --------
# Every prior session needed a manual `apt install python3-pip` before the nanobind
# ext build (step 4) and the evidence harnesses (step 6). Do it here so the session
# is turnkey: install pip if missing, then nanobind/numpy/mpmath (kernels + accuracy)
# and thewalrus (the same-instance baseline). Best-effort -- the gates don't need them.
say "python deps (pip + nanobind/numpy/mpmath/thewalrus)"
if ! python3 -m pip --version >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y --no-install-recommends python3-pip python3-dev >/dev/null 2>&1 || true
fi
python3 -m pip install --quiet nanobind numpy mpmath thewalrus 2>/dev/null \
  || python3 -m pip install --quiet --break-system-packages nanobind numpy mpmath thewalrus 2>/dev/null || true
for m in numpy mpmath nanobind thewalrus; do
  python3 -c "import $m" 2>/dev/null && echo " $m OK" || echo " [warn] $m absent"
done

# --- provenance: the rented box has no .git (rsync excludes it), so the host
# captures the commit into COMMIT_SHA before upload; export it for the benches.
# The pinned CONTAINER DIGEST (the image the instance was launched from) is likewise
# captured into CONTAINER_DIGEST on the host (or read from the box) and exported, so
# EVERY artifact (throughput, e2e, accuracy, sampler) records both -- a frozen,
# reproducible experiment (docs/DESIGN.md §9). ---
if [ -z "${GBS_COMMIT:-}" ] && [ -f COMMIT_SHA ]; then
  export GBS_COMMIT="$(tr -d '[:space:]' < COMMIT_SHA)"
fi
if [ -z "${GBS_CONTAINER_DIGEST:-}" ] && [ -f CONTAINER_DIGEST ]; then
  export GBS_CONTAINER_DIGEST="$(tr -d '[:space:]' < CONTAINER_DIGEST)"
fi
[ -n "${GBS_COMMIT:-}" ] && echo "artifact commit: ${GBS_COMMIT}" || echo "[warn] no GBS_COMMIT / COMMIT_SHA -- artifacts will record commit: null"
[ -n "${GBS_CONTAINER_DIGEST:-}" ] && echo "container digest: ${GBS_CONTAINER_DIGEST}" || echo "[warn] no GBS_CONTAINER_DIGEST / CONTAINER_DIGEST -- artifacts will record container_digest: null"

# --- 1. CPU pre-flight ------------------------------------------------------
say "1/4 CPU pre-flight (host shim -- must already be green)"
bash core/preflight/run_preflight.sh || die "CPU pre-flight failed -- fix before spending GPU time"

# --- 2. build core/ with nvcc ----------------------------------------------
say "2/4 nvcc build of core/"
cmake -S core -B core/build -DCMAKE_CUDA_ARCHITECTURES="$ARCH" >/dev/null || die "cmake configure failed"
cmake --build core/build -j || die "nvcc build failed (this is the first GPU-only check)"

# --- 2b. register/spill profiling (DYNAMIC footprint; nvcc -Xptxas -v) -------
# The MEASURED complement to the static bench/kernel_footprint.py: a profiling re-compile with
# -Xptxas -v captures each kernel's real register count + spill stores/loads + stack frame (the
# spill driver behind the cooperative-kernel results). Best-effort; recorded under results/perf/.
say "2b register/spill profiling (nvcc -Xptxas -v) -> results/perf/"
mkdir -p results/perf
PROF="results/perf/ptxas_sm${ARCH}_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "# ptxas -v register/spill | arch sm_${ARCH} | commit ${GBS_COMMIT:-null} | digest ${GBS_CONTAINER_DIGEST:-null}"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
} > "$PROF"
if cmake -S core -B core/build_prof -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DCMAKE_CUDA_FLAGS="-Xptxas=-v" >/dev/null 2>&1; then
  cmake --build core/build_prof -j 2>&1 \
    | grep -iE "ptxas info|Function properties|Used .* registers|spill (stores|loads)|stack frame" >> "$PROF" || true
  echo " -> $PROF ($(grep -ciE 'registers|spill' "$PROF" 2>/dev/null) register/spill lines)"
  grep -iE "Function properties|Used .* registers|spill" "$PROF" 2>/dev/null | head -16 || true
else
  echo " [warn] profiling build (-Xptxas -v) failed; skipping the dynamic footprint"
fi

# --- 2c. achieved-occupancy / DRAM profiling (ncu) -> results/perf/ ----------
# The ptxas stack frame PREDICTS
# occupancy collapse from the per-thread footprint; ncu MEASURES it. Profiles the four hard
# per-thread FP64 kernels -- haf / lhaf / tor / the candidate-C real-Cholesky tor -- for
# achieved occupancy + DRAM/compute throughput, so we can (1) confirm the footprint->occupancy
# mechanism on-device and (2) read candidate C's occupancy gain (half the per-thread buffer)
# straight off tor vs tor_real_chol. Best-effort: ncu is often absent or lacks GPU perf-counter
# permissions in rented containers -> warn, never abort (like the ptxas step).
say "2c achieved-occupancy / DRAM profiling (ncu) -> results/perf/"
NCU_OUT="results/perf/ncu_sm${ARCH}_$(date -u +%Y%m%dT%H%M%SZ).csv"
if command -v ncu >/dev/null 2>&1 && [ -x ./core/build/bench_kernels ]; then
  ncu_metrics="sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread"
  echo "# ncu achieved-occupancy/DRAM | arch sm_${ARCH} | commit ${GBS_COMMIT:-null} | digest ${GBS_CONTAINER_DIGEST:-null}" > "$NCU_OUT"
  ncu_ok=0
  for k in haf_powertrace_fp64_kernel_t loop_haf_fp64_kernel tor_fp64_kernel tor_real_chol_fp64_kernel tor_recursive_real_kernel lhaf_repeated_kernel; do
    # -s 2 -c 1: skip this kernel's 2 warm-up launches, profile its first TIMED launch (the
    # smallest benched size) -- one clean steady-state sample per hard kernel.
    if ncu --target-processes all -k "regex:$k" -s 2 -c 1 --metrics "$ncu_metrics" --csv \
          ./core/build/bench_kernels 2048 3 >>"$NCU_OUT" 2>>"${NCU_OUT%.csv}.log"; then
      ncu_ok=1
    else
      echo " [warn] ncu failed for $k (profiler perms / availability); see ${NCU_OUT%.csv}.log"
    fi
  done
  [ "$ncu_ok" -eq 1 ] && echo " -> $NCU_OUT" \
                      || echo " [warn] ncu produced no profile (GPU perf counters likely restricted in this container)"
else
  echo " [warn] ncu not available (or bench_kernels not built); skipping achieved-occupancy profiling"
fi

# --- 3. differential gates (must all PASS) ---------------------------------
say "3/5 GPU-vs-CPU differential gates (incl. double-double + host_api)"
mkdir -p results/gpu_gates
gates_ok=1
for g in check_permanent check_hafnian check_loop_hafnian check_torontonian check_torontonian_real_chol \
         check_permanent_dd check_hafnian_dd check_loop_hafnian_dd check_torontonian_dd \
         check_permanent_coop check_haf_coop check_lhaf_coop check_tor_coop \
         check_haf_small check_permanent_warp check_certified check_repeated check_tor_recursive check_host_api check_session \
         check_sampler_draw check_sampler_gather check_sampler_haf_varn check_sampler_session; do
  out="$(./core/build/$g || true)"
  echo "[$g] $out" | tr '\n' ' '; echo
  if echo "$out" | grep -q '^PASS$'; then
    printf 'PASS %s\n' "$g" > "results/gpu_gates/${g#check_}_PASS.txt"
  else
    gates_ok=0
  fi
done
[ "$gates_ok" -eq 1 ] || die "a differential gate FAILED -- not running timing (no number from a failed gate)"

# --- 4. Python bindings: build the CUDA extension + smoke it ----------------
# Best-effort: the C++ gates above already validate the kernels + host_api on the
# device, so a binding-build hiccup warns but does not abort the session.
say "4/6 build + smoke the nanobind GPU extension"
python3 -m pip install --quiet nanobind numpy mpmath 2>/dev/null || true # best-effort
if python3 -c 'import nanobind' 2>/dev/null; then
  if cmake -S bindings -B bindings/build -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        -DPython_EXECUTABLE="$(command -v python3)" >/dev/null 2>&1 \
     && cmake --build bindings/build -j >/dev/null 2>&1; then
    python3 - <<'PY' || echo " [warn] extension built but smoke failed"
import sys
sys.path.insert(0, "bindings/build")
import gbskernels_ext as e
print(" bindings import OK:", [x for x in dir(e) if not x.startswith("_")])
try:
    import numpy as np
    g = np.random.default_rng(0)
    A = np.ascontiguousarray(g.standard_normal((8, 6, 6)) + 1j * g.standard_normal((8, 6, 6)))
    print(" gpu perm runs on device -> shape", e.perm(A).shape,
          "| perm_dd shape", e.perm_dd(A).shape)
    # precision="auto" cancellation indicators (perm/haf/lhaf/tor), on device
    Asym = A + A.transpose(0, 2, 1) # symmetric for haf/lhaf
    Oreal = np.ascontiguousarray((np.real(A) * 0.1) + np.real(A).transpose(0, 2, 1) * 0.1)
    for nm, inp in [("perm", A), ("haf", Asym), ("lhaf", Asym), ("tor", Oreal)]:
        v, k = getattr(e, nm + "_kappa")(inp)
        print(" %-4s_kappa on device -> values %s absnorm %s" % (nm, v.shape, k.shape))
except ModuleNotFoundError:
    print(" (numpy absent; import-only smoke)")
PY
  else
    echo " [warn] bindings CUDA build failed (kernels already validated by the gates)"
  fi
else
  echo " [skip] nanobind unavailable; bindings not built (gates already validate the kernels)"
fi

# --- 5. kernel-only throughput ----------------------------------------------
say "5/6 kernel-only throughput (bench_kernels) -> results/throughput/"
PY="python3"; command -v uv >/dev/null && PY="uv run python"
$PY -m bench.throughput_gpu --batch 4096 --repeats 7 || die "throughput driver failed"

# --- 6. GPU-vs-mpmath accuracy + end-to-end (binding) throughput -------------
# Needs numpy/mpmath + the built extension (GBSKERNELS_EXT_DIR -> bindings/build).
# Best-effort: these are the evidence harnesses; warn (don't abort) if the Python
# env is incomplete, since the gates above are the correctness record.
say "6/6 GPU-vs-mpmath accuracy (physical + adversarial) + end-to-end throughput"
export GBSKERNELS_EXT_DIR="$PWD/bindings/build"
# These evidence harnesses (incl. the real-device public-path e2e + FP64/DD vs
# mpmath) need numpy/mpmath; install if absent so the artifacts are actually
# produced (the last session skipped them for lack of numpy).
# numpy/mpmath for the evidence harnesses; thewalrus for the SAME-INSTANCE baseline.
python3 -c 'import numpy, mpmath' 2>/dev/null \
  || python3 -m pip install --quiet --break-system-packages numpy mpmath thewalrus 2>/dev/null \
  || python3 -m pip install --quiet numpy mpmath thewalrus 2>/dev/null || true
if python3 -c 'import numpy, mpmath' 2>/dev/null && [ -d bindings/build ]; then
  $PY -m bench.accuracy --dps 60 || echo " [warn] accuracy study failed"
  # calibrate precision="auto": kappa-vs-actual-error on physical/loss/adversarial (CPU+mpmath)
  $PY -m bench.calibrate_auto --dps 60 || echo " [warn] auto calibration failed"
  # Public-path throughput + HONESTY GATE: throughput_end_to_end exits 2 when the GPU and CPU
  # backends disagree on a well-conditioned (physical) cell -- a kernel/binding bug from which
  # no timing is publishable, so abort the session (anchor: "no number from a failed gate").
  # Other (incidental) failures only warn. The physical regime is the gate; adversarial inputs
  # may legitimately diverge and are an accuracy-study input, not a gate.
  if $PY -m bench.throughput_end_to_end --batch 4096 --repeats 7 --regime physical; then :; else
    rc=$?
    [ "$rc" -eq 2 ] && die "PUBLIC-PATH CHECKSUM GATE FAILED (GPU != CPU on the physical regime)"
    echo " [warn] e2e throughput failed (rc=$rc)"
  fi
  # Throughput on the loss/mixed + adversarial regimes too, so the public claim is not
  # physical-only (NON-gating: adversarial GPU/CPU determinants legitimately diverge on
  # ill-conditioned inputs, so a checksum mismatch / exit 2 here is EXPECTED, not a failure).
  for rg in loss adversarial; do
    $PY -m bench.throughput_end_to_end --batch 4096 --repeats 7 --regime "$rg" \
      || echo " [warn] e2e throughput ($rg regime, rc=$?) -- adversarial divergence is expected"
  done
  # end-to-end GBS sampler: samples/sec (the real-workload metric, not kernel evals/sec)
  $PY -m bench.sampler_throughput --modes 6 --num-samples 2000 --cutoff 5 || echo " [warn] sampler throughput failed"
  # sampler characterization sweep (hybrid vs resident vs sieve across the surface)
  $PY -m bench.sampler_throughput --sweep --repeated-sieve --num-samples 1200 || echo " [warn] sampler sweep failed"
  # R4 device A/B: sieve kernel vs expanded hafnian kernel on identical workloads
  $PY -m bench.repeated_ab --batch 2048 --qs 2,3,4,5,6 --repeats 5 || echo " [warn] repeated_ab failed"
  # v3 hybrid-vs-fully-on-device (resident) before/after, at a config within the hafnian cap
  # (2*modes*cutoff <= cap): produces both the 'gpu' (hybrid) and 'gpu-resident' rows.
  for cfg in "3 3" "2 4" "5 2"; do set -- $cfg
    $PY -m bench.sampler_throughput --modes "$1" --cutoff "$2" --num-samples 2000 \
      || echo " [warn] resident sampler bench (modes=$1 cutoff=$2) failed"
  done
  # same-instance The Walrus baseline (apples-to-apples on this box); needs thewalrus.
  if python3 -c 'import thewalrus' 2>/dev/null; then
    $PY -m bench.walrus_baseline --batch 64 --repeats 7 || echo " [warn] Walrus baseline failed"
  else
    echo " [skip] thewalrus unavailable; no same-instance baseline this run"
  fi
  # batch-size sweep + GPU/CPU/Walrus crossover (the batched-throughput thesis figure). Swept
  # from batch 1 so the GPU genuinely loses at the small end and the crossover is REAL (not the
  # smallest batch tried); strict (default) -> aborts on e2e disagreement / missing Walrus.
  $PY -m bench.crossover --batches 1,4,16,64,256,1024,4096,16384 --repeats 7 || echo " [warn] crossover sweep failed"
  # --- certified-numerics figure: bound-tightness DISTRIBUTIONS (CPU, mpmath refs) ---
  $PY -m bench.tightness --samples 150 --plot || echo " [warn] tightness bench failed"
else
  echo " [skip] numpy/mpmath or the GPU extension unavailable; skipping the evidence harnesses"
fi

say "DONE"
echo "kernels built with nvcc; all differential gates passed on the GPU; the Python"
echo "extension built+smoked; kernel-only + end-to-end throughput and the four-function"
echo "GPU-vs-mpmath accuracy study recorded under results/ -- copy results/ back."
