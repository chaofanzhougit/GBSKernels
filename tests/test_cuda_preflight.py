"""Layer 5 -- CPU pre-flight of the CUDA kernels.

Compiles each CUDA kernel + its differential gate as plain C++ against the
cuda_shim headers and runs it on the host. A green pre-flight means the kernel
*logic* is validated without a GPU, so a paid rented-GPU session only has to
confirm device compilation + execution -- it should not be debugging syntax or
algorithm errors on a paid clock.

Skips cleanly where no C++ host compiler is available (so CI on a minimal runner
still passes); the real GPU differential gates run under nvcc in the session.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.layer5

SCRIPT = Path(__file__).resolve().parent.parent / "core" / "preflight" / "run_preflight.sh"


def _have_compiler() -> bool:
    return shutil.which("clang++") is not None or shutil.which("g++") is not None


@pytest.mark.skipif(not _have_compiler(), reason="no C++ host compiler (clang++/g++)")
def test_all_cuda_kernels_pass_on_host():
    assert SCRIPT.exists(), f"missing pre-flight script: {SCRIPT}"
    proc = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        f"CUDA host pre-flight failed:\n{proc.stdout}\n{proc.stderr}"
    )
    # all four kernels must report a host PASS
    for kernel in ("permanent", "hafnian", "loop_hafnian", "torontonian"):
        assert f"[{kernel}]" in proc.stdout and "PASS" in proc.stdout
    assert "ALL FOUR CUDA KERNELS PASS ON HOST" in proc.stdout
