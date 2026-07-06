# envs/ — container definition for GPU sessions

`Dockerfile` is a reproducible CUDA 12.4 + Python 3.12 image that builds `core/`
(the four kernels, their differential gates, and the throughput harness) and
carries the uv-managed Python environment. The CPU pre-flight
(`core/preflight/run_preflight.sh`) runs at build time, so an image that cannot
pass the kernels on the host is never produced.

```bash
docker build -f envs/Dockerfile -t gbskernels-gpu .
docker run --gpus all --rm -v "$PWD/results:/work/results" gbskernels-gpu \
  bash -lc './core/build/check_permanent && ./core/build/check_hafnian \
            && ./core/build/check_loop_hafnian && ./core/build/check_torontonian'
```

For any run whose numbers will be published, pin the base image and uv by digest
(not tag) and record the resulting image digest alongside the `results/`
artifacts ([`docs/benchmark_protocol.md`](../docs/benchmark_protocol.md)). The
CUDA architectures default to `70;80;89;90`; narrow to your card to reduce build
time. `rental_4090.md` and `rental_datacenter.md` document provisioning a
single-GPU session end to end.
