# envs/ — container definition for GPU sessions

`Dockerfile` is a versioned CUDA 12.4 + Python 3.12 development image that builds
`core/` (the four kernels, their differential gates, and the throughput harness)
and carries the uv-managed Python environment. Its convenience references are
tags; published runs use the digest-pinned procedure below. The CPU pre-flight
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

## Publication-session dependencies

`publication-requirements.txt` is the direct, exact-pin supplement used only by
the prospective archive-bound publication session. It is not a change to the
pure-Python wheel's runtime requirements, and it is not a complete lock file by
itself:

| Package | Pin | Publication role |
|---|---:|---|
| `build` | `1.5.0` | Build the wheel from a separate extraction of the immutable release archive. |
| `nanobind` | `2.13.0` | Configure and build the recorded CUDA extension. |
| `numpy` | `2.4.6` | Canonical binary64 inputs and corpus serialization. |
| `mpmath` | `1.3.0` | Existing arbitrary-precision references and verification dependencies. |
| `python-flint` | `0.9.0` | Independent Arb interval reference for the enclosure campaign. |
| `thewalrus` | `0.22.0` | Recursive CPU torontonian baseline. |
| `piquasso` | `8.0.1` | Independent recursive CPU torontonian baseline. |

The on-box driver derives exact registry constraints and the sdist-only package
set from the immutable `uv.lock`, installs fixed bootstrap tooling, builds those
sdists without build isolation, then installs the direct requirements under the
lock constraints. It records installer reports, `pip freeze --all`, operating
system packages, tool versions, and `pip check`. Thus both direct and resolved
transitive versions are retained with the build evidence.

The adapter resolves its small set of Ubuntu bootstrap packages at run time.
Their exact installed versions and the complete bootstrap log are retained and
hash-bound, but the base-image digest alone cannot reproduce that APT resolution
after the repository changes. Exact environment replay therefore requires a
derived digest-pinned image containing those recorded packages (or an archived
package snapshot); the present workflow supports source-to-binary audit and does
not claim bit-for-bit future reconstruction of the bootstrap layer.

The repository `Dockerfile` remains a development convenience. The publication
launcher instead requires a caller-supplied `repository@sha256:<digest>` image;
the generated Vast adapter currently assumes a Debian-family CUDA image with
`apt-get`, and the on-box driver additionally requires Python 3.12, `nvcc`,
`cuobjdump`, `nvidia-smi`, CMake, Ninja, a C++ compiler, `tar`, `sha256sum`, and
`dpkg-query`. The driver refuses a missing tool, a tag-only image identity, an
in-tree build or output, or a source/archive hash mismatch. These requirements
describe the not-yet-executed publication harness and do not assert that its GPU
campaign has passed.
