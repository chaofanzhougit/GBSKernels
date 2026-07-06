# core/ -- CUDA C++ kernels. Built only in a rented-GPU session (docs/DESIGN.md §10);
# there is no local NVIDIA toolchain in the dev environment.
#
# cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89 # 89=Ada/4090, 90=H100
# cmake --build build -j
# ./build/check_permanent # GPU-vs-CPU differential gate (must PASS)

cmake_minimum_required(VERSION 3.18) # 3.18 = first with CMAKE_CUDA_ARCHITECTURES; Ubuntu 22.04 ships 3.22
project(gbskernels_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90) # cover smoke + measurement tiers
endif()

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Differential checks: each GPU kernel vs an independent host reference.
add_executable(check_permanent permanent.cu check_permanent.cu)
set_target_properties(check_permanent PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(check_hafnian hafnian.cu check_hafnian.cu)
set_target_properties(check_hafnian PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(check_loop_hafnian loop_hafnian.cu check_loop_hafnian.cu)
set_target_properties(check_loop_hafnian PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(check_torontonian torontonian.cu check_torontonian.cu)
set_target_properties(check_torontonian PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# real-Cholesky torontonian (physical real O) -- candidate C; gate vs the complex-LU kernel.
add_executable(check_torontonian_real_chol torontonian.cu check_torontonian_real_chol.cu)
set_target_properties(check_torontonian_real_chol PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Certified kernels: fp64 value + rigorous bound; gate = value
# consistency with the plain kernels + long-double enclosure + tightness.
add_executable(check_certified certified.cu certified_dd.cu permanent.cu permanent_dd.cu hafnian.cu hafnian_dd.cu loop_hafnian.cu check_certified.cu)
set_target_properties(check_certified PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Repeated-row loop-hafnian sieve: gate vs a naive expanded
# loop hafnian AND a long-double host sieve.
add_executable(check_repeated repeated.cu check_repeated.cu)
set_target_properties(check_repeated PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Recursive prefix-Cholesky torontonian: gate vs
# long-double direct subset determinants + the off-domain NaN contract.
add_executable(check_tor_recursive tor_recursive.cu check_tor_recursive.cu)
set_target_properties(check_tor_recursive PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Cooperative permanent (perf): warp/block map/reduce vs independent host Glynn.
add_executable(check_permanent_coop permanent_coop.cu check_permanent_coop.cu)
set_target_properties(check_permanent_coop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Cooperative hafnian / loop hafnian / torontonian: map/reduce vs the per-thread
# kernel they regroup (which is itself validated vs an independent reference).
add_executable(check_haf_coop hafnian.cu check_haf_coop.cu)
set_target_properties(check_haf_coop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Size-specialized hafnian (perf research): small-cap == full-cap.
add_executable(check_haf_small hafnian.cu check_haf_small.cu)
set_target_properties(check_haf_small PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(check_lhaf_coop loop_hafnian.cu check_lhaf_coop.cu)
set_target_properties(check_lhaf_coop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(check_tor_coop torontonian.cu check_tor_coop.cu)
set_target_properties(check_tor_coop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# on-device sampler: the conditional DRAW kernel
# (normalise + inverse-CDF + cuRAND) vs the multinomial its weights define.
add_executable(check_sampler_draw sampler_draw.cu check_sampler_draw.cu)
set_target_properties(check_sampler_draw PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# v3 on-device sampler: the conditional submatrix GATHER kernel vs hand-derived np.ix_ indices.
add_executable(check_sampler_gather sampler_gather.cu check_sampler_gather.cu)
set_target_properties(check_sampler_gather PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# v3 on-device sampler: the variable-N hafnian (ragged gather output) vs the single-size kernel.
add_executable(check_sampler_haf_varn hafnian.cu check_sampler_haf_varn.cu)
set_target_properties(check_sampler_haf_varn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# v3 on-device sampler: the RESIDENT chain (gather -> varn-haf -> draw per mode, no per-mode D2H)
# vs a host orchestration using the same kernels.
add_executable(check_sampler_session sampler_session.cu sampler_gather.cu sampler_draw.cu
               hafnian.cu check_sampler_session.cu)
set_target_properties(check_sampler_session PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Fused warp permanent (__shfl reduction): GPU-ONLY -- not in the host pre-flight
# (the shim cannot emulate warp shuffles); validated only here on the device.
add_executable(check_permanent_warp permanent_warp.cu check_permanent_warp.cu)
set_target_properties(check_permanent_warp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Double-double precision tier (docs/DESIGN.md §6): permanent + hafnian, each + gate.
add_executable(check_permanent_dd permanent.cu permanent_dd.cu check_permanent_dd.cu)
set_target_properties(check_permanent_dd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(check_hafnian_dd hafnian.cu hafnian_dd.cu check_hafnian_dd.cu)
set_target_properties(check_hafnian_dd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(check_loop_hafnian_dd loop_hafnian.cu loop_hafnian_dd.cu check_loop_hafnian_dd.cu)
set_target_properties(check_loop_hafnian_dd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(check_torontonian_dd torontonian.cu torontonian_dd.cu check_torontonian_dd.cu)
set_target_properties(check_torontonian_dd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# All DD kernels share this source set with the FP64 ones.
# permanent_coop.cu is in the shared set: host_api.cu dispatches large permanents
# to it, so every target linking host_api also needs it.
set(ALL_KERNELS permanent.cu permanent_coop.cu permanent_dd.cu hafnian.cu hafnian_dd.cu
    loop_hafnian.cu loop_hafnian_dd.cu torontonian.cu torontonian_dd.cu
    certified.cu certified_dd.cu repeated.cu tor_recursive.cu)

# Host-API smoke test (the layer the Python bindings call).
add_executable(check_host_api ${ALL_KERNELS} host_api.cu check_host_api.cu)
set_target_properties(check_host_api PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Device-resident session gate (docs/device_resident_contract.md): buffer reuse
# across differently-sized buckets + equality with the one-shot host API.
add_executable(check_session ${ALL_KERNELS} host_api.cu check_session.cu)
set_target_properties(check_session PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Throughput timing harness (FP64 four kernels + the four DD kernels + the
# cooperative permanent + the fused warp permanent); driven by bench/throughput_gpu.py.
# GBS_BENCH_WARP pulls in the GPU-only warp permanent (the host pre-flight build
# omits both, so it stays host-shim-buildable).
add_executable(bench_kernels ${ALL_KERNELS} permanent_warp.cu bench_kernels.cu)
target_compile_definitions(bench_kernels PRIVATE GBS_BENCH_WARP)
set_target_properties(bench_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

enable_testing()
add_test(NAME permanent_gpu_equals_cpu COMMAND check_permanent)
add_test(NAME hafnian_gpu_equals_cpu COMMAND check_hafnian)
add_test(NAME loop_hafnian_gpu_equals_cpu COMMAND check_loop_hafnian)
add_test(NAME torontonian_gpu_equals_cpu COMMAND check_torontonian)
add_test(NAME permanent_coop_equals_glynn COMMAND check_permanent_coop)
add_test(NAME haf_coop_equals_perthread COMMAND check_haf_coop)
add_test(NAME haf_small_equals_full COMMAND check_haf_small)
add_test(NAME lhaf_coop_equals_perthread COMMAND check_lhaf_coop)
add_test(NAME tor_coop_equals_perthread COMMAND check_tor_coop)
add_test(NAME permanent_warp_equals_glynn COMMAND check_permanent_warp)
add_test(NAME permanent_dd_gpu COMMAND check_permanent_dd)
add_test(NAME hafnian_dd_gpu COMMAND check_hafnian_dd)
add_test(NAME loop_hafnian_dd_gpu COMMAND check_loop_hafnian_dd)
add_test(NAME torontonian_dd_gpu COMMAND check_torontonian_dd)
add_test(NAME host_api_smoke COMMAND check_host_api)
add_test(NAME session_reuse_and_correctness COMMAND check_session)
add_test(NAME sampler_draw_distribution COMMAND check_sampler_draw)
add_test(NAME sampler_gather_indices COMMAND check_sampler_gather)
add_test(NAME sampler_haf_varn_equals_single COMMAND check_sampler_haf_varn)
add_test(NAME sampler_session_chain COMMAND check_sampler_session)
