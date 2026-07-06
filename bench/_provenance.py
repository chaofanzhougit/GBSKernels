"""Shared provenance for every benchmark artifact (the frozen-experiment record).

Anchor §9 / the reproducibility contract: *every* artifact -- throughput, end-to-end,
accuracy, sampler -- carries the same provenance block so a result can be reproduced
exactly: the code **commit**, the pinned **container digest**, the GPU, and when it ran.

The rented box has no ``.git`` (rsync excludes it) and its image is pinned by digest,
so both are passed in by the session script as environment variables captured on the
host / from the container before the run:

* ``GBS_COMMIT``            -- ``git rev-parse --short HEAD`` of the uploaded tree.
* ``GBS_CONTAINER_DIGEST``  -- the image digest the instance was launched from
  (e.g. ``nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:...``); pins the toolchain.

Both fall back to a local probe (git; ``/etc/gbs_container_digest`` if present) and to
``None`` -- recorded honestly rather than faked.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from typing import Any


def _gpu_info() -> dict[str, Any] | None:
    """GPU model / compute capability / driver / memory via nvidia-smi (None off-device)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            name, cc, drv, mem = (x.strip() for x in out.splitlines()[0].split(","))
            return {"name": name, "compute_cap": cc, "driver": drv, "memory_total": mem}
    except Exception:
        pass
    return None


def _cuda_version() -> str | None:
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"release (\d+\.\d+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def environment() -> dict[str, Any]:
    """The machine the artifact was produced on, so a throughput/accuracy number is
    self-describing: the GPU (model/driver/CUDA), the CPU + logical cores, and the BLAS/
    OpenMP thread caps (which materially affect the CPU-side timings; see the OpenBLAS
    many-core-host segfault mitigation in gpu_session.sh)."""
    return {
        "gpu": _gpu_info(),
        "cuda": _cuda_version(),
        "cpu": {"processor": platform.processor() or None,
                "machine": platform.machine(), "logical_cores": os.cpu_count()},
        "blas_threads": {v: os.environ.get(v) for v in
                         ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "GOTO_NUM_THREADS")},
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def commit() -> str | None:
    env = os.environ.get("GBS_COMMIT", "").strip()
    if env:
        return env
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def container_digest() -> str | None:
    """The pinned container image digest the run was launched from (or None)."""
    env = os.environ.get("GBS_CONTAINER_DIGEST", "").strip()
    if env:
        return env
    try:  # a session may drop the digest here on the box
        p = "/etc/gbs_container_digest"
        if os.path.exists(p):
            with open(p) as f:
                return f.read().strip() or None
    except Exception:
        pass
    return None


def provenance() -> dict[str, Any]:
    """The common provenance block embedded in every artifact (commit + container digest +
    machine environment), so a result reproduces from the file alone (docs/DESIGN.md §9)."""
    return {
        "commit": commit(),
        "container_digest": container_digest(),
        "hostname": platform.node() or None,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
    }
