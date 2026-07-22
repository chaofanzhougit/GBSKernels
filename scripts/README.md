# scripts/ — GPU-session orchestration

Tooling to run the on-device validation and benchmarks on a rented CUDA host,
CPU-first by construction.

- **`gpu_session.sh`** — the on-box runner. In one command it runs the CPU
  pre-flight, builds `core/` with `nvcc`, and runs every mandatory differential
  gate before publishable throughput timing. Compiler and profiler diagnostics
  collected immediately after the build remain provisional until those gates
  pass. The nanobind build/smoke and several Python evidence harnesses are best-
  effort and report warnings when unavailable; the core build and gates remain
  hard failures, so publishable timing never follows a failed correctness gate.
- **`launch_session.sh`** — drives a session from a workstation: it captures the
  commit and container digest, copies the working tree to the host (excluding
  local-only files), runs `gpu_session.sh` there, and copies `results/` back.
- **`session.py`** — an experimental local manifest wrapper around pluggable
  provision/terminate commands. Its safety contract is active (`--dry-run`,
  `--confirm`, CPU gate, cost caps), but manifest cells execute in the local
  checkout; use `launch_session.sh` for the implemented remote-host workflow.

```bash
bash scripts/gpu_session.sh                         # on the CUDA host
bash scripts/launch_session.sh -p PORT USER@HOST 89 # from a workstation
```

Replace `PORT` and `USER@HOST` with the SSH endpoint for the rented host.
