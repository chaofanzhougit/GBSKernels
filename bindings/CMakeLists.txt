# bindings/ -- nanobind Python extension over the CUDA kernels.
#
# Two build modes:
# * real GPU build (default): nvcc compiles the kernels; produces a CUDA
# extension. Run in a rented-GPU session.
# cmake -S bindings -B bindings/build -DCMAKE_CUDA_ARCHITECTURES=89
# cmake --build bindings/build -j
# * host pre-flight (-DGBS_HOST_SHIM=ON): the kernels compile as plain C++
# against core/preflight/cuda (grid-emulated on CPU), so the *Python ->
# kernel* path can be imported and validated WITHOUT a GPU.
# cmake -S bindings -B bindings/build_host -DGBS_HOST_SHIM=ON
# cmake --build bindings/build_host -j

cmake_minimum_required(VERSION 3.18)

option(GBS_HOST_SHIM "Build the kernels as host C++ against the CPU shim (no CUDA)" OFF)

if(GBS_HOST_SHIM)
  project(gbskernels_ext LANGUAGES CXX)
else()
  project(gbskernels_ext LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# Locate nanobind via its installed Python package.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

set(CORE "${CMAKE_CURRENT_SOURCE_DIR}/../core")
set(KERNELS
  "${CORE}/permanent.cu" "${CORE}/permanent_coop.cu" "${CORE}/permanent_dd.cu"
  "${CORE}/hafnian.cu" "${CORE}/hafnian_dd.cu"
  "${CORE}/loop_hafnian.cu" "${CORE}/loop_hafnian_dd.cu"
  "${CORE}/torontonian.cu" "${CORE}/torontonian_dd.cu" "${CORE}/certified.cu" "${CORE}/certified_dd.cu" "${CORE}/repeated.cu" "${CORE}/tor_recursive.cu" "${CORE}/host_api.cu"
  # v3 fully on-device sampler (draw + gather + resident chain; hafnian.cu has the varn kernel)
  "${CORE}/sampler_draw.cu" "${CORE}/sampler_gather.cu" "${CORE}/sampler_session.cu")

nanobind_add_module(gbskernels_ext gbskernels_ext.cpp ${KERNELS})
target_include_directories(gbskernels_ext PRIVATE "${CORE}")

if(GBS_HOST_SHIM)
  # Compile the kernels (only) as C++ against the shim headers. -x c++ forces
  # clang to treat the .cu files as C++ (not CUDA); -include pulls the shim
  # runtime (as nvcc implicitly would). Applied per-source so nanobind's own
  # translation units are untouched.
  set(SHIM "${CORE}/preflight/cuda")
  set_source_files_properties(${KERNELS} PROPERTIES
    LANGUAGE CXX
    COMPILE_OPTIONS "-x;c++;-include;${SHIM}/cuda_runtime.h")
  target_include_directories(gbskernels_ext PRIVATE "${SHIM}")
  # The binary records its own provenance: host-shim builds must never be
  # reported as a real GPU run (honest "dd_backend" labelling, docs/DESIGN.md §6/§8).
  target_compile_definitions(gbskernels_ext PRIVATE GBS_HOST_SHIM=1)
else()
  set_target_properties(gbskernels_ext PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  find_package(CUDAToolkit REQUIRED)
  target_include_directories(gbskernels_ext PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
endif()
