#!/usr/bin/env python3
"""Rented-GPU session orchestrator — provision, gate, run, terminate.

CPU-first is a hard rule: this script **refuses to spend** until the CPU dry-run
is green and the user passes ``--confirm``. It is idempotent/resumable per
manifest cell, prints a cost estimate up front, and streams artifacts back to
``results/`` (append-only). It deliberately does not embed any cloud provider's
credentials or API; the provisioning step is a pluggable command so the same
flow works for any rented instance.

Flow:
    cpu dry-run gate  ->  provision  ->  pull pinned container  ->  run manifest
    ->  stream artifacts to results/  ->  terminate (always, even on error)

Tiers: ``smoke`` (cheapest CUDA GPU; "does it run") and ``measurement`` (one
fixed consumer card, e.g. 4090/L40S; optionally H100 for high-end batch curves).

Examples:
    # See exactly what would happen and the cost estimate -- spends nothing:
    python scripts/session.py --tier smoke --dry-run

    # Actually provision and run (requires green CPU dry-run + explicit consent):
    python scripts/session.py --tier measurement --confirm \
        --provision "vastai-create.sh" --terminate "vastai-destroy.sh"

It prints a cost estimate up front and refuses to provision without
``--confirm`` and a green CPU dry-run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

# Rough USD/hour by tier -- for the up-front estimate only; the real charge comes
# from the provider. Conservative so the estimate never under-warns.
TIER_RATE_USD_PER_HOUR = {"smoke": 0.40, "measurement": 1.20, "h100": 3.50}
DEFAULT_COST_CAP_USD = 250.0    # default safety cap; override with --max-usd
HARD_STOP_USD = 400.0           # absolute abort threshold


@dataclass
class Manifest:
    """An ordered list of GPU work cells. Each cell is resumable: if its sentinel
    artifact already exists in results/, it is skipped (idempotent re-runs)."""

    tier: str
    est_hours: float
    cells: list[dict] = field(default_factory=list)

    @staticmethod
    def default(tier: str) -> "Manifest":
        # The canonical first session: build core/, run every differential gate
        # (must PASS before any timing), then the GPU throughput sweep.
        return Manifest(
            tier=tier,
            est_hours=0.5 if tier == "smoke" else 1.5,
            cells=[
                # Cheapest assurance first: the CPU pre-flight (all four kernels
                # already pass on host). If this is red, do not even build on GPU.
                {"name": "cpu_preflight", "cmd": "bash core/preflight/run_preflight.sh",
                 "artifact": None, "must_pass": True},
                {"name": "build_core", "cmd": "cmake -S core -B core/build "
                 "-DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build core/build -j",
                 "artifact": None},
                {"name": "gate_permanent", "cmd": "./core/build/check_permanent",
                 "artifact": "results/gpu_gates/permanent_PASS.txt", "must_pass": True},
                {"name": "gate_hafnian", "cmd": "./core/build/check_hafnian",
                 "artifact": "results/gpu_gates/hafnian_PASS.txt", "must_pass": True},
                {"name": "gate_loop_hafnian", "cmd": "./core/build/check_loop_hafnian",
                 "artifact": "results/gpu_gates/loop_hafnian_PASS.txt", "must_pass": True},
                {"name": "gate_torontonian", "cmd": "./core/build/check_torontonian",
                 "artifact": "results/gpu_gates/torontonian_PASS.txt", "must_pass": True},
                {"name": "gate_permanent_dd", "cmd": "./core/build/check_permanent_dd",
                 "artifact": "results/gpu_gates/permanent_dd_PASS.txt", "must_pass": True},
                {"name": "gate_hafnian_dd", "cmd": "./core/build/check_hafnian_dd",
                 "artifact": "results/gpu_gates/hafnian_dd_PASS.txt", "must_pass": True},
                {"name": "gate_loop_hafnian_dd", "cmd": "./core/build/check_loop_hafnian_dd",
                 "artifact": "results/gpu_gates/loop_hafnian_dd_PASS.txt", "must_pass": True},
                {"name": "gate_torontonian_dd", "cmd": "./core/build/check_torontonian_dd",
                 "artifact": "results/gpu_gates/torontonian_dd_PASS.txt", "must_pass": True},
                {"name": "gate_host_api", "cmd": "./core/build/check_host_api",
                 "artifact": "results/gpu_gates/host_api_PASS.txt", "must_pass": True},
                # device-resident session + the cooperative/warp kernels (must gate the
                # SAME build that the throughput run below benchmarks).
                {"name": "gate_session", "cmd": "./core/build/check_session",
                 "artifact": "results/gpu_gates/session_PASS.txt", "must_pass": True},
                {"name": "gate_permanent_coop", "cmd": "./core/build/check_permanent_coop",
                 "artifact": "results/gpu_gates/permanent_coop_PASS.txt", "must_pass": True},
                {"name": "gate_haf_coop", "cmd": "./core/build/check_haf_coop",
                 "artifact": "results/gpu_gates/haf_coop_PASS.txt", "must_pass": True},
                {"name": "gate_lhaf_coop", "cmd": "./core/build/check_lhaf_coop",
                 "artifact": "results/gpu_gates/lhaf_coop_PASS.txt", "must_pass": True},
                {"name": "gate_tor_coop", "cmd": "./core/build/check_tor_coop",
                 "artifact": "results/gpu_gates/tor_coop_PASS.txt", "must_pass": True},
                {"name": "gate_permanent_warp", "cmd": "./core/build/check_permanent_warp",
                 "artifact": "results/gpu_gates/permanent_warp_PASS.txt", "must_pass": True},
                # All kernels (incl. DD) timed in one run; driver writes results/throughput/.
                {"name": "throughput_gpu",
                 "cmd": "python -m bench.throughput_gpu --batch 4096 --repeats 7",
                 "artifact": None},
            ],
        )


def _run(cmd: str, check: bool = True) -> int:
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, cwd=REPO, check=check).returncode


def cpu_dry_run_green() -> bool:
    """The gate: the full CPU verification suite must pass before any GPU spend."""
    print("[gate] running CPU verification suite (must be green before GPU spend)...")
    rc = subprocess.run("uv run pytest -q", shell=True, cwd=REPO).returncode
    return rc == 0


def cost_estimate(tier: str, est_hours: float) -> float:
    return TIER_RATE_USD_PER_HOUR.get(tier, TIER_RATE_USD_PER_HOUR["measurement"]) * est_hours


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=["smoke", "measurement", "h100"], default="smoke")
    p.add_argument("--dry-run", action="store_true", help="plan only; spend nothing")
    p.add_argument("--confirm", action="store_true", help="required to actually provision/spend")
    p.add_argument("--skip-cpu-gate", action="store_true",
                   help="(discouraged) skip the CPU dry-run gate")
    p.add_argument("--provision", default=None, help="command to provision the GPU instance")
    p.add_argument("--terminate", default=None, help="command to destroy the instance (always run)")
    p.add_argument("--max-usd", type=float, default=DEFAULT_COST_CAP_USD)
    args = p.parse_args()

    manifest = Manifest.default(args.tier)
    est = cost_estimate(args.tier, manifest.est_hours)

    print(f"=== GBSKernels rented-GPU session :: tier={args.tier} ===")
    print(f"  estimated runtime : {manifest.est_hours:.1f} h")
    print(f"  estimated cost    : ${est:.2f}  (abort above ${HARD_STOP_USD:.0f})")
    print(f"  manifest cells    : {[c['name'] for c in manifest.cells]}")

    if est > HARD_STOP_USD:
        print(f"[abort] estimate ${est:.2f} exceeds hard stop ${HARD_STOP_USD:.0f}")
        return 2
    if est > args.max_usd:
        print(f"[abort] estimate ${est:.2f} exceeds --max-usd ${args.max_usd:.2f}")
        return 2

    if args.dry_run:
        print("\n[dry-run] plan above. No CPU gate run, nothing provisioned, $0 spent.")
        print("[dry-run] re-run with --confirm (and --provision/--terminate) to execute.")
        return 0

    if not args.confirm:
        print("\n[refused] this would spend money. Re-run with --confirm to proceed "
              "(or --dry-run to just plan).")
        return 1

    if not args.skip_cpu_gate:
        if not cpu_dry_run_green():
            print("[abort] CPU dry-run is RED -- nothing runs on a rented GPU until it is "
                  "green (docs/DESIGN.md §8). Fix the suite first.")
            return 3
        print("[gate] CPU suite green. Proceeding to provision.")

    if not args.provision or not args.terminate:
        print("[abort] --provision and --terminate commands are required to actually run "
              "(kept provider-agnostic on purpose).")
        return 1

    (RESULTS / "gpu_gates").mkdir(parents=True, exist_ok=True)
    provisioned = False
    try:
        _run(args.provision)
        provisioned = True
        for cell in manifest.cells:
            art = cell.get("artifact")
            if art and (REPO / art).exists():
                print(f"[skip] {cell['name']} (artifact exists: {art})")
                continue
            print(f"[run] {cell['name']}")
            rc = _run(cell["cmd"], check=False)
            if cell.get("must_pass") and rc != 0:
                print(f"[abort] gate {cell['name']} FAILED (rc={rc}); stopping before any "
                      "timing/throughput cell. No published number from a failed gate.")
                return 4
    finally:
        if provisioned:
            print("[teardown] terminating instance (always, even on error)...")
            _run(args.terminate, check=False)
        else:
            print("[teardown] nothing provisioned; nothing to terminate.")

    print("[done] session complete; artifacts in results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
