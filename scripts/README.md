# scripts/ — GPU-session orchestration

Tooling to run the on-device validation and benchmarks on a rented CUDA host,
CPU-first by construction.

- **`gpu_session.sh`** — the on-box runner. In one command it runs the CPU
  pre-flight, builds `core/` with `nvcc`, runs every differential gate, builds
  and smoke-tests the Python extension, and runs the throughput and accuracy
  harnesses. It aborts before any timing if the build or a gate fails, so no
  recorded number ever comes from a failed gate.
- **`launch_session.sh`** — drives a session from a workstation: it captures the
  commit and container digest, copies the working tree to the host (excluding
  local-only files), runs `gpu_session.sh` there, and copies `results/` back.
- **`session.py`** — a provider-agnostic orchestrator with an explicit safety
  contract: `--dry-run` prints the plan and spends nothing; provisioning requires
  `--confirm`; the CPU verification suite must be green before any GPU work; and
  the run stops before any timing cell if a gate fails.

```bash
bash scripts/gpu_session.sh                         # on the CUDA host
bash scripts/launch_session.sh -p <port> <user>@<host> 89   # from a workstation
```
