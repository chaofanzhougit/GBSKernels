# Pinned container for rented-GPU sessions (docs/DESIGN.md §9/sec.10).
#
# Reproducible CUDA + Python 3.12 image: builds core/ (the four kernels + their
# differential gates + the throughput harness) and provides the uv-managed Python
# env for the driver scripts. Every published number records the image digest it
# came from. Pin by digest in production (see envs/README.md).
#
# docker build -f envs/Dockerfile -t gbskernels-gpu .
# docker run --gpus all --rm -v "$PWD/results:/work/results" gbskernels-gpu \
# bash -lc 'cmake -S core -B core/build -DCMAKE_CUDA_ARCHITECTURES=89 \
# && cmake --build core/build -j \
# && ./core/build/check_permanent && ./core/build/check_hafnian \
# && ./core/build/check_loop_hafnian && ./core/build/check_torontonian \
# && uv run python -m bench.throughput_gpu --batch 4096'

# CUDA 12.4 devel on Ubuntu 22.04 (nvcc + headers). Pin by digest in real runs.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build git ca-certificates curl \
        python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# uv for the Python layer (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.11.20 /uv /usr/local/bin/uv

WORKDIR /work
COPY . /work

# Resolve the Python env at build time (CPU-only deps; the GPU work is C++/CUDA).
RUN uv sync --frozen || uv sync

# CPU pre-flight at build time: the four kernels must pass on host before the
# image is considered good (fails the build otherwise).
RUN bash core/preflight/run_preflight.sh

# Build the CUDA targets (kernels + gates + bench) and the nanobind extension.
# Correctness gates + throughput are invoked by the run command (see header /
# scripts/gpu_session.sh) so results/ can be mounted.
RUN cmake -S core -B core/build -DCMAKE_CUDA_ARCHITECTURES="70;80;89;90" \
    && cmake --build core/build -j
RUN python3 -m pip install --no-cache-dir nanobind \
    && cmake -S bindings -B bindings/build -DCMAKE_CUDA_ARCHITECTURES="70;80;89;90" \
         -DPython_EXECUTABLE="$(command -v python3)" \
    && cmake --build bindings/build -j

CMD ["bash"]
