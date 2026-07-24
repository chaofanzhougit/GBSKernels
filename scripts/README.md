# scripts/ — GPU-session orchestration

Tooling to run the on-device validation and benchmarks on a rented CUDA host,
CPU-first by construction.

- **`gpu_session.sh`** — the on-box runner. In one command it runs the CPU
  pre-flight, builds `core/` with `nvcc`, and runs every mandatory differential
  gate before publishable throughput timing. Compiler and profiler diagnostics
  collected immediately after the build remain provisional until those gates
  pass. In ordinary benchmark modes, optional evidence harnesses may warn when
  unavailable. In release `validate` mode, the nanobind build/smoke, adversarial
  enclosure with physical-family coverage, and Jiuzhang Gate C probe are all
  hard gates; success writes a semantic manifest with hashes for all evidence.
- **`launch_session.sh`** — drives a session from a workstation: it captures the
  full commit and container digest, copies the working tree to the host
  (excluding local-only files), runs `gpu_session.sh` there, and copies
  `results/` back. Release `validate` mode refuses tracked changes, staged
  changes, or untracked non-ignored files before upload.
- **`session.py`** — an experimental local manifest wrapper around pluggable
  provision/terminate commands. Its safety contract is active (`--dry-run`,
  `--confirm`, CPU gate, cost caps), but manifest cells execute in the local
  checkout; use `launch_session.sh` for the implemented remote-host workflow.

```bash
bash scripts/gpu_session.sh                         # on the CUDA host
bash scripts/launch_session.sh -p PORT USER@HOST 89 # from a workstation
bash scripts/launch_session.sh -p PORT USER@HOST 89 IMAGE@sha256:DIGEST validate
```

Replace `PORT` and `USER@HOST` with the SSH endpoint for the rented host.
