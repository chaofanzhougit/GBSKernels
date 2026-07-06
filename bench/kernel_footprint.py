"""Static per-thread local-memory footprint of the one-eval-per-thread kernels.

This is the *static* half of "profile register spills / local-memory traffic"
(perf research item 5): it accounts, from the kernel sources, the per-thread local
arrays each kernel declares -- the quantity that drives register spilling and the
local-memory (off-chip) traffic that bounds the hard kernels. The *dynamic* half
(ncu/nvprof spill counts, achieved occupancy, DRAM throughput) needs a real device
and is run in a GPU session.

Why this matters, and why it is NOT optimization-by-analogy: the measured cooperative
results map exactly onto the footprint below. The permanent's per-thread state is a
single length-n vector (sub-KB -> lives in registers, no spill), so its bottleneck is
the *serial 2^(n-1) Glynn chain* and splitting that across a group wins ~5x. The
hafnian / loop hafnian / torontonian instead carry several n x n matrices per thread
(tens of KB -> spills to local/off-chip memory), so they are memory-bound on the
*per-subset* work; splitting their (short, 2^(N/2)) subset sum cannot help and adds
launch + global-partials overhead -- which is exactly what was measured (haf 0.6x,
lhaf 0.5x, tor 1.2x). The lever for the hard kernels is the per-thread footprint.

    uv run python -m bench.kernel_footprint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CPLX = 16  # bytes per cuDoubleComplex (2 x double)
DBL = 8

# Per-thread local arrays each one-eval-per-thread kernel declares, as
# (name, element_bytes, size_expr(maxdim)). maxdim is the kernel's compile-time cap
# (HAF_MAX_N etc.); the arrays are sized for it REGARDLESS of the actual matrix size,
# which is the core inefficiency for small inputs. Transcribed from core/*.cu.
_KERNELS: dict[str, dict[str, Any]] = {
    "permanent": {
        "cap": 28, "cap_name": "PERM_MAX_N",
        "buffers": [("rowsum", CPLX, lambda M: M)],
    },
    "hafnian": {
        "cap": 20, "cap_name": "HAF_MAX_N",
        "buffers": [("BX", CPLX, lambda M: M * M), ("P", CPLX, lambda M: M * M),
                    ("T", CPLX, lambda M: M * M), ("p", CPLX, lambda M: M + 1),
                    ("e", CPLX, lambda M: M + 1), ("pidx", 4, lambda M: M // 2)],
    },
    "loop_hafnian": {
        "cap": 20, "cap_name": "LHAF_MAX_N",
        "buffers": [("C", CPLX, lambda M: M * M), ("P", CPLX, lambda M: M * M),
                    ("Q", CPLX, lambda M: M * M), ("T", CPLX, lambda M: M * M),
                    ("d", CPLX, lambda M: M), ("w", CPLX, lambda M: M),
                    ("kg", CPLX, lambda M: M + 1), ("e", CPLX, lambda M: M + 1),
                    ("pidx", 4, lambda M: M // 2)],
    },
    "torontonian": {
        "cap": 24, "cap_name": "TOR_MAX_DIM",
        "buffers": [("sub", CPLX, lambda M: M * M), ("idx", 4, lambda M: M)],
    },
}

# A 64 KB/thread guideline: above this, an SM (with e.g. 64-99 KB usable as registers
# + L1) cannot hold even a handful of threads' working sets, so the per-thread arrays
# spill to off-chip local memory. The practical spill cliff for these kernels is well
# below: the register file is 256 KB/SM = 1 KB / thread at full occupancy, so anything
# past ~1 KB already starts spilling. We report both the footprint and the implied
# occupancy ceiling so the GPU session knows what to confirm with ncu.
_REGFILE_PER_SM = 256 * 1024  # bytes (Ampere/Ada: 64K 32-bit registers)


def footprint(kernel: str, dim: int) -> int:
    """Per-thread local-array bytes for `kernel` if its buffers were sized to `dim`."""
    return sum(eb * sz(dim) for _, eb, sz in _KERNELS[kernel]["buffers"])


def report() -> dict[str, Any]:
    rows = []
    for name, spec in _KERNELS.items():
        cap = spec["cap"]
        at_cap = footprint(name, cap)
        # threads/SM the register file can hold before spilling (rough occupancy ceiling)
        threads = _REGFILE_PER_SM // max(at_cap, 1)
        # for the hard kernels the matrices are usually small; show a small/typical size
        small = 8 if name != "torontonian" else 8  # dim 8 (haf/lhaf N=8; tor 2n=8)
        rows.append({
            "kernel": name, "cap_name": spec["cap_name"], "cap": cap,
            "bytes_at_cap": at_cap, "kb_at_cap": round(at_cap / 1024, 1),
            "bytes_at_dim8": footprint(name, small),
            "kb_at_dim8": round(footprint(name, small) / 1024, 2),
            "occupancy_threads_per_sm_at_cap": threads,
            "n_nxn_buffers": sum(1 for nm, _, sz in spec["buffers"] if sz(10) == 100),
        })
    return {"kind": "kernel_footprint", "regfile_per_sm_bytes": _REGFILE_PER_SM, "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = p.parse_args()
    rep = report()
    print("# per-thread local-memory footprint (static; the spill source)")
    print(f"#   register file/SM = {_REGFILE_PER_SM // 1024} KB "
          f"(~1 KB/thread at full occupancy -> anything past that spills)")
    print(f"  {'kernel':>13} {'cap':>14} {'KB@cap':>7} {'KB@dim8':>8} {'nxn bufs':>9} "
          f"{'occ thr/SM@cap':>15}")
    for r in rep["rows"]:
        print(f"  {r['kernel']:>13} {r['cap_name']+'='+str(r['cap']):>14} "
              f"{r['kb_at_cap']:>7} {r['kb_at_dim8']:>8} {r['n_nxn_buffers']:>9} "
              f"{r['occupancy_threads_per_sm_at_cap']:>15}")
    print("\n#  Reading: the permanent's <1 KB lives in registers (no spill); the hard")
    print("#  kernels carry several n x n matrices (tens of KB) -> spill -> memory-bound")
    print("#  on the per-subset work. Sizing buffers to the ACTUAL dim (size specialization)")
    print("#  is the lever -- e.g. hafnian at dim 8 is ~6x smaller than at the cap.")
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2))
        print(f"\n# wrote {args.json}")


if __name__ == "__main__":
    main()
