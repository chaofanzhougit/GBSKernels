#!/usr/bin/env bash
# core/preflight/run_preflight.sh
#
# CPU pre-flight for the CUDA kernels (docs/DESIGN.md ): compile
# each kernel + its differential gate as plain C++ against the cuda_shim headers
# and RUN it on the host, so syntax/type/logic errors are caught *before* a paid
# rented-GPU session. The kernel sources are unchanged from the real GPU build;
# only the launch macro (GBS_LAUNCH_1D) and the shim headers differ, so a host
# PASS means the algorithm is correct and the GPU session only has to confirm
# device compilation + execution.
#
# Exits 0 iff all four kernels build and pass. Used by tests/test_cuda_preflight.py
# and CI. Requires a C++17 host compiler (clang++ or g++); no CUDA toolchain.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE="$(cd "$HERE/.." && pwd)"
SHIM="$HERE/cuda"
CXX="${CXX:-clang++}"
command -v "$CXX" >/dev/null 2>&1 || CXX=g++
command -v "$CXX" >/dev/null 2>&1 || { echo "no C++ compiler (clang++/g++) found"; exit 127; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# kernel : gate
KERNELS=("permanent:check_permanent"
         "hafnian:check_hafnian"
         "loop_hafnian:check_loop_hafnian"
         "torontonian:check_torontonian"
         "torontonian:check_torontonian_real_chol"
         "sampler_draw:check_sampler_draw"
         "sampler_gather:check_sampler_gather"
         "hafnian:check_sampler_haf_varn"
         "repeated:check_repeated"
         "tor_recursive:check_tor_recursive")

fail=0
for pair in "${KERNELS[@]}"; do
  kern="${pair%%:*}"; gate="${pair##*:}"
  bin="$WORK/pf_$kern"
  if ! "$CXX" -std=c++17 -O1 -I "$SHIM" -I "$CORE" -x c++ \
        -include "$SHIM/cuda_runtime.h" \
        "$CORE/$kern.cu" "$CORE/$gate.cu" -o "$bin" 2>"$WORK/$kern.log"; then
    echo "[$kern] COMPILE FAILED"; sed 's/^/ /' "$WORK/$kern.log" | head -20; fail=1; continue
  fi
  out="$("$bin" || true)"
  if echo "$out" | grep -q '^PASS$'; then
    echo "[$kern] $(echo "$out" | head -1) -> PASS (host)"
  else
    echo "[$kern] FAILED:"; echo "$out" | sed 's/^/ /'; fail=1
  fi
done

# Certified gate: links the certified kernels against the plain
# fp64 kernels + a long-double enclosure reference.
if "$CXX" -std=c++17 -O2 -I "$SHIM" -I "$CORE" -x c++ \
      -include "$SHIM/cuda_runtime.h" \
      "$CORE/certified.cu" "$CORE/certified_dd.cu" "$CORE/permanent.cu" "$CORE/permanent_dd.cu" "$CORE/hafnian.cu" "$CORE/hafnian_dd.cu" "$CORE/loop_hafnian.cu" "$CORE/check_certified.cu" \
      -o "$WORK/pf_certified" 2>"$WORK/certified.log"; then
  out="$("$WORK/pf_certified" || true)"
  if echo "$out" | grep -q '^PASS$'; then
    echo "[certified] enclosure + value-consistency + tightness -> PASS (host)"
  else
    echo "[certified] FAILED:"; echo "$out" | sed 's/^/ /'; fail=1
  fi
else
  echo "[certified] COMPILE FAILED"; sed 's/^/ /' "$WORK/certified.log" | head -20; fail=1
fi

# Double-double permanent gate (links the FP64 + DD permanent kernels). Validates
# the DD tier on host: DD holds at machine precision where FP64 cancels.
if "$CXX" -std=c++17 -O2 -I "$SHIM" -I "$CORE" -x c++ \
      -include "$SHIM/cuda_runtime.h" \
      "$CORE/permanent.cu" "$CORE/permanent_dd.cu" "$CORE/check_permanent_dd.cu" \
      -o "$WORK/pf_perm_dd" 2>"$WORK/perm_dd.log"; then
  out="$("$WORK/pf_perm_dd" || true)"
  if echo "$out" | grep -q '^PASS$'; then
    echo "[permanent_dd] DD holds where FP64 cancels -> PASS (host)"
  else
    echo "[permanent_dd] FAILED:"; echo "$out" | sed 's/^/ /'; fail=1
  fi
else
  echo "[permanent_dd] COMPILE FAILED"; sed 's/^/ /' "$WORK/perm_dd.log" | head -20; fail=1
fi

# Double-double hafnian gate (the hardest kernel in DD): power-trace in DD holds
# at machine precision where the FP64 subset sum cancels.
if "$CXX" -std=c++17 -O2 -I "$SHIM" -I "$CORE" -x c++ \
      -include "$SHIM/cuda_runtime.h" \
      "$CORE/hafnian.cu" "$CORE/hafnian_dd.cu" "$CORE/check_hafnian_dd.cu" \
      -o "$WORK/pf_haf_dd" 2>"$WORK/haf_dd.log"; then
  out="$("$WORK/pf_haf_dd" || true)"
  if echo "$out" | grep -q '^PASS$'; then
    echo "[hafnian_dd] DD holds where FP64 cancels -> PASS (host)"
  else
    echo "[hafnian_dd] FAILED:"; echo "$out" | sed 's/^/ /'; fail=1
  fi
else
  echo "[hafnian_dd] COMPILE FAILED"; sed 's/^/ /' "$WORK/haf_dd.log" | head -20; fail=1
fi

# Double-double loop-hafnian + torontonian gates.
for trio in "loop_hafnian:loop_hafnian_dd:check_loop_hafnian_dd:loop_hafnian_dd" \
            "torontonian:torontonian_dd:check_torontonian_dd:torontonian_dd"; do
  k1="${trio%%:*}"; rest="${trio#*:}"; k2="${rest%%:*}"; rest="${rest#*:}"
  gate="${rest%%:*}"; tag="${rest#*:}"
  if "$CXX" -std=c++17 -O2 -I "$SHIM" -I "$CORE" -x c++ -include "$SHIM/cuda_runtime.h" \
        "$CORE/$k1.cu" "$CORE/$k2.cu" "$CORE/$gate.cu" -o "$WORK/pf_$tag" 2>"$WORK/$tag.log" \
     && "$WORK/pf_$tag" | grep -q '^PASS$'; then
    echo "[$tag] DD holds where FP64 cancels -> PASS (host)"
  else
    echo "[$tag] FAILED"; sed 's/^/ /' "$WORK/$tag.log" | head -20; fail=1
  fi
done

# Host-API plumbing (the layer the Python bindings call): smoke test all wrappers
# against known closed-form values.
if "$CXX" -std=c++17 -O2 -I "$SHIM" -I "$CORE" -x c++ \
      -include "$SHIM/cuda_runtime.h" \
      "$CORE/permanent.cu" "$CORE/permanent_coop.cu" "$CORE/permanent_dd.cu" "$CORE/hafnian.cu" "$CORE/hafnian_dd.cu" "$CORE/loop_hafnian_dd.cu" "$CORE/torontonian_dd.cu" \
      "$CORE/loop_hafnian.cu" "$CORE/torontonian.cu" "$CORE/certified.cu" "$CORE/certified_dd.cu" "$CORE/repeated.cu" "$CORE/tor_recursive.cu" "$CORE/host_api.cu" "$CORE/check_host_api.cu" \
      -o "$WORK/pf_hostapi" 2>"$WORK/hostapi.log" \
   && "$WORK/pf_hostapi" | grep -q '^PASS$'; then
  echo "[host_api] all five host wrappers -> PASS (host)"
else
  echo "[host_api] FAILED"; sed 's/^/ /' "$WORK/hostapi.log" | head -20; fail=1
fi

# v3 on-device sampler -- the RESIDENT chain. The device orchestration (per-mode loop +
# compact/scatter/extract helpers) must equal a host orchestration using the SAME gather /
# variable-N hafnian / draw kernels and cuRAND states -- isolating the new orchestration.
if "$CXX" -std=c++17 -O2 -I "$SHIM" -I "$CORE" -x c++ -include "$SHIM/cuda_runtime.h" \
      "$CORE/sampler_session.cu" "$CORE/sampler_gather.cu" "$CORE/sampler_draw.cu" \
      "$CORE/hafnian.cu" "$CORE/check_sampler_session.cu" \
      -o "$WORK/pf_sampler_session" 2>"$WORK/sampler_session.log" \
   && "$WORK/pf_sampler_session" | grep -q '^PASS$'; then
  echo "[sampler_session] resident chain == host orchestration (device loop + helpers) -> PASS (host)"
else
  echo "[sampler_session] FAILED"; sed 's/^/ /' "$WORK/sampler_session.log" | head -20; fail=1
fi

# Cooperative permanent (perf): the warp/block-cooperative map/reduce permanent
# (groups split the 2^(n-1) Glynn sum) must equal the independent host Glynn across
# n and cooperation widths -- the first kernel of the headline perf mapping.
if "$CXX" -std=c++17 -O2 -I "$SHIM" -I "$CORE" -x c++ \
      -include "$SHIM/cuda_runtime.h" \
      "$CORE/permanent_coop.cu" "$CORE/check_permanent_coop.cu" \
      -o "$WORK/pf_perm_coop" 2>"$WORK/perm_coop.log" \
   && "$WORK/pf_perm_coop" | grep -q '^PASS$'; then
  echo "[permanent_coop] cooperative == independent Glynn (groups 1/8/32) -> PASS (host)"
else
  echo "[permanent_coop] FAILED"; sed 's/^/ /' "$WORK/perm_coop.log" | head -20; fail=1
fi

# Size-specialized hafnian (perf research): the small-buffer-cap kernel must equal the
# full-cap per-thread kernel for N <= HAF_SMALL_N (only the footprint differs).
if "$CXX" -std=c++17 -O2 -I "$SHIM" -I "$CORE" -x c++ -include "$SHIM/cuda_runtime.h" \
      "$CORE/hafnian.cu" "$CORE/check_haf_small.cu" -o "$WORK/pf_haf_small" 2>"$WORK/haf_small.log" \
   && "$WORK/pf_haf_small" | grep -q '^PASS$'; then
  echo "[haf_small] size-specialized == full-cap hafnian (N<=12) -> PASS (host)"
else
  echo "[haf_small] FAILED"; sed 's/^/ /' "$WORK/haf_small.log" | head -20; fail=1
fi

# Cooperative hafnian / loop hafnian / torontonian: the map/reduce variant (groups
# split the 2^(N/2) subset sum) must equal the per-thread kernel it regroups.
for pair in "hafnian:check_haf_coop:haf_coop" \
            "loop_hafnian:check_lhaf_coop:lhaf_coop" \
            "torontonian:check_tor_coop:tor_coop"; do
  k="${pair%%:*}"; rest="${pair#*:}"; g="${rest%%:*}"; tag="${rest##*:}"
  if "$CXX" -std=c++17 -O2 -I "$SHIM" -I "$CORE" -x c++ -include "$SHIM/cuda_runtime.h" \
        "$CORE/$k.cu" "$CORE/$g.cu" -o "$WORK/pf_$tag" 2>"$WORK/$tag.log" \
     && "$WORK/pf_$tag" | grep -q '^PASS$'; then
    echo "[$tag] cooperative == per-thread (groups 1/8/32) -> PASS (host)"
  else
    echo "[$tag] FAILED"; sed 's/^/ /' "$WORK/$tag.log" | head -20; fail=1
  fi
done

# Device-resident session (docs/device_resident_contract.md): the session reuses
# device buffers across differently-sized buckets (reallocs witness) and returns
# the same values as the one-shot host API.
if "$CXX" -std=c++17 -O2 -I "$SHIM" -I "$CORE" -x c++ \
      -include "$SHIM/cuda_runtime.h" \
      "$CORE/permanent.cu" "$CORE/permanent_coop.cu" "$CORE/permanent_dd.cu" "$CORE/hafnian.cu" "$CORE/hafnian_dd.cu" "$CORE/loop_hafnian_dd.cu" "$CORE/torontonian_dd.cu" \
      "$CORE/loop_hafnian.cu" "$CORE/torontonian.cu" "$CORE/certified.cu" "$CORE/certified_dd.cu" "$CORE/repeated.cu" "$CORE/tor_recursive.cu" "$CORE/host_api.cu" "$CORE/check_session.cu" \
      -o "$WORK/pf_session" 2>"$WORK/session.log" \
   && "$WORK/pf_session" | grep -q '^PASS$'; then
  echo "[session] device-resident buffer reuse + correctness -> PASS (host)"
else
  echo "[session] FAILED"; sed 's/^/ /' "$WORK/session.log" | head -20; fail=1
fi

# Also compile-check the throughput harness (links all four kernels). Timing on
# host is meaningless, but a clean build catches errors before the GPU session.
if "$CXX" -std=c++17 -O1 -I "$SHIM" -I "$CORE" -x c++ \
      -include "$SHIM/cuda_runtime.h" \
      "$CORE/permanent.cu" "$CORE/permanent_coop.cu" "$CORE/permanent_dd.cu" "$CORE/hafnian.cu" "$CORE/hafnian_dd.cu" "$CORE/loop_hafnian_dd.cu" "$CORE/torontonian_dd.cu" \
      "$CORE/loop_hafnian.cu" "$CORE/torontonian.cu" "$CORE/tor_recursive.cu" "$CORE/certified.cu" "$CORE/certified_dd.cu" "$CORE/bench_kernels.cu" -o "$WORK/pf_bench" 2>"$WORK/bench.log" \
   && "$WORK/pf_bench" 2 1 >/dev/null 2>&1; then
  echo "[bench_kernels] compiles & runs on host -> PASS (host)"
else
  echo "[bench_kernels] COMPILE/RUN FAILED"; sed 's/^/ /' "$WORK/bench.log" | head -20; fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "ALL FOUR CUDA KERNELS PASS ON HOST (CPU pre-flight green)"
fi
exit "$fail"
