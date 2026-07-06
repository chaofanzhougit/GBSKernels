"""The static kernel-footprint analysis (perf research item 5, the static half).

Guards the diagnosis that drives the hard-kernel perf work: the permanent's
per-thread state is sub-KB (no spill), while haf/lhaf/tor carry several n x n matrices
(tens of KB -> spill -> memory-bound), and sizing buffers to the actual dim (size
specialization) is a multiple-x footprint cut. If a kernel's local buffers change,
this test makes the footprint change visible.
"""

from __future__ import annotations

from bench.kernel_footprint import footprint, report


def test_permanent_fits_in_registers_hard_kernels_spill():
    # the permanent's single length-n vector stays sub-KB at its cap; the hard kernels
    # are an order of magnitude larger (the measured-coop-failure explanation).
    assert footprint("permanent", 28) < 1024            # < 1 KB -> registers
    for k in ("hafnian", "loop_hafnian", "torontonian"):
        assert footprint(k, 20 if k != "torontonian" else 24) > 8 * 1024  # > 8 KB -> spill


def test_loop_hafnian_is_the_worst_footprint():
    fp = {k: footprint(k, 20 if k != "torontonian" else 24)
          for k in ("hafnian", "loop_hafnian", "torontonian")}
    assert fp["loop_hafnian"] == max(fp.values())       # 4 n x n buffers -> worst


def test_size_specialization_is_a_multiple_x_cut():
    # the lever: hafnian buffers at dim 8 vs the cap (20) -- a >=3x footprint cut.
    assert footprint("hafnian", 20) / footprint("hafnian", 8) >= 3.0
    assert footprint("loop_hafnian", 20) / footprint("loop_hafnian", 8) >= 3.0


def test_report_is_well_formed():
    r = report()
    assert r["kind"] == "kernel_footprint"
    kernels = {row["kernel"] for row in r["rows"]}
    assert kernels == {"permanent", "hafnian", "loop_hafnian", "torontonian"}
    perm = next(row for row in r["rows"] if row["kernel"] == "permanent")
    haf = next(row for row in r["rows"] if row["kernel"] == "hafnian")
    # occupancy ceiling: the permanent holds vastly more threads/SM than the hafnian
    assert perm["occupancy_threads_per_sm_at_cap"] > 10 * haf["occupancy_threads_per_sm_at_cap"]
