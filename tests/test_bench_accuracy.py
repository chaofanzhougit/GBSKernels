"""Layer 5 -- the accuracy harness produces well-formed, sensible artifacts.

Runs the harness on a tiny config (into a tmp dir, never the real append-only
results/) and asserts the structure and the qualitative shape of the measured
boundary: FP64 is near-exact when well-conditioned, and degrades as the Glynn
condition number is driven up.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.accuracy_permanent import run

pytestmark = pytest.mark.layer5


def test_all_four_functions_show_the_measured_boundary(tmp_path):
    """The measured FP64<->DD boundary covers ALL FOUR functions, not just the
    permanent: FP64 degrades as each result cancels; DD (when the kernels are
    built) holds near machine precision throughout."""
    from bench.accuracy import run as run_all

    artifact, path = run_all(dps=50, out_dir=tmp_path)
    assert path.exists()
    assert set(artifact["sweeps"]) == {"perm", "haf", "lhaf", "tor"}
    assert "INTERNAL precision tier" in artifact["note"]  # DD-is-internal stated
    assert artifact["dd_backend"] in {"none", "host-shim", "gpu"}  # honest provenance
    for func, sec in artifact["sweeps"].items():
        # physical (well-conditioned) inputs: FP64 is accurate, DD agrees
        for r in sec["physical"]:
            assert r["rel_err_fp64"] < 1e-10, f"{func}: FP64 accurate on physical inputs"
            if "rel_err_dd" in r:
                assert r["rel_err_dd"] < 1e-12, f"{func}: DD accurate on physical inputs"
        # adversarial (cancellation): FP64 degrades, DD holds
        adv = sec["adversarial"]
        fp = [r["rel_err_fp64"] for r in adv]
        assert fp[-1] > 1e4 * fp[0], f"{func}: FP64 should degrade as it cancels"
        assert fp[-1] > (1e-8 if func == "tor" else 1e-6), f"{func}: FP64 must visibly break"
        if "rel_err_dd" in adv[0]:
            dd = [r["rel_err_dd"] for r in adv]
            assert max(dd) < 1e-13, f"{func}: DD holds near machine precision"
            assert dd[-1] < fp[-1] / 1e4, f"{func}: DD vastly better where FP64 broke"


def test_loss_mixed_state_regime_is_measured(tmp_path):
    """The accuracy study includes the LOSS / mixed-state regime for the Gaussian
    functions (the third realistic regime). FP64 is accurate on these well-conditioned
    mixed-state matrices and DD agrees -- they are realistic, not adversarial."""
    from bench.accuracy import run as run_all

    artifact, _ = run_all(dps=50, out_dir=tmp_path)
    assert "loss (mixed-state" in artifact["params"]["sections"]
    for func in ("haf", "lhaf", "tor"):
        loss = artifact["sweeps"][func].get("loss")
        assert loss, f"{func}: loss/mixed-state section present"
        for r in loss:
            assert r["rel_err_fp64"] < 1e-9, f"{func}: FP64 accurate on lossy/mixed inputs"
            if "rel_err_dd" in r:
                assert r["rel_err_dd"] < 1e-12, f"{func}: DD accurate on lossy/mixed inputs"
    assert "loss" not in artifact["sweeps"]["perm"]  # no Gaussian-loss analog for the permanent


def test_harness_writes_artifact_and_is_well_conditioned_at_small_n(tmp_path):
    artifact, path = run(sizes=[2, 4, 6], seeds_per_size=3, dps=50, out_dir=tmp_path)

    assert path.exists() and path.parent == tmp_path
    assert artifact["kind"] == "accuracy_permanent"
    assert {"env", "params", "size_sweep", "cancellation_sweep"} <= artifact.keys()
    assert artifact["env"]["numpy"] and artifact["env"]["mpmath"]

    # Well-conditioned small sizes: FP64 tracks the reference to near machine eps.
    for row in artifact["size_sweep"]:
        assert row["rel_err_median"] < 1e-12


def test_harness_cancellation_axis_shows_the_boundary(tmp_path):
    artifact, _ = run(sizes=[4], seeds_per_size=2, dps=50, out_dir=tmp_path)
    sweep = artifact["cancellation_sweep"]
    kappas = [row["kappa"] for row in sweep]
    errs = [row["rel_err_fp64"] for row in sweep]

    # deltas descend, so kappa ascends monotonically...
    assert all(a < b for a, b in zip(kappas, kappas[1:]))
    # ...and the smallest-delta (worst-conditioned) case has lost real accuracy
    # while the best-conditioned one is essentially exact.
    assert errs[0] < 1e-12
    assert errs[-1] > 1e-4

    # The DD tier stays at ~machine precision across the whole sweep -- the
    # measured FP64<->DD boundary (docs/DESIGN.md §6).
    dd_errs = [row["rel_err_dd"] for row in sweep]
    assert all(e < 1e-13 for e in dd_errs)
    assert dd_errs[-1] < errs[-1] / 1e6  # DD vastly better where FP64 broke
