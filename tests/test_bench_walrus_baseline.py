"""The same-instance The Walrus baseline harness + the loss/mixed input families.

Guards the §9 frozen-experiment additions that don't need a GPU: the lossy/mixed-state
input families are valid and well-conditioned, and the Walrus baseline harness produces
a well-formed, provenance-stamped artifact (so the GPU session can drop it next to the
kernel throughput for an apples-to-apples, same-instance comparison).
"""

from __future__ import annotations

import numpy as np
import pytest

from bench import _inputs


def test_loss_families_are_valid_mixed_state_inputs():
    A = _inputs.loss_hafnian(6, seed=1)
    L = _inputs.loss_loop_hafnian(6, seed=1)
    O = _inputs.loss_torontonian(3, seed=1)
    assert A.shape == (6, 6) and np.allclose(A, A.T)          # complex symmetric
    assert L.shape == (6, 6) and np.allclose(L - np.diag(np.diag(L)), (L - np.diag(np.diag(L))).T)
    assert O.shape == (6, 6) and np.allclose(O.imag, 0)        # real threshold matrix
    # a lossy/mixed state is genuinely mixed: det(Q) > 1 (a pure state would be closer
    # to the lower bound), and distinct from the pure construction.
    Q = _inputs._qmat(_inputs._lossy_cov(3, seed=1, eta=0.6))
    assert np.linalg.det(Q).real > 1.0
    assert not np.allclose(_inputs._lossy_cov(3, 1, eta=0.6), _inputs._lossy_cov(3, 1, eta=1.0))


def test_loss_kernels_match_mpmath():
    import mpmath
    import cpu_ref
    import highprec_ref as hp
    for fn, mk, mpf in [(cpu_ref.haf, lambda: _inputs.loss_hafnian(6, 2), hp.hafnian_mp),
                        (cpu_ref.lhaf, lambda: _inputs.loss_loop_hafnian(6, 2), hp.loop_hafnian_mp),
                        (cpu_ref.tor, lambda: _inputs.loss_torontonian(3, 2), hp.torontonian_mp)]:
        M = mk()
        with mpmath.workdps(60):
            ex = mpf(M, dps=60)
        rel = float(abs(mpmath.mpc(complex(fn(M))) - ex) / abs(ex))
        assert rel < 1e-10, "FP64 accurate on the well-conditioned lossy/mixed inputs"


def test_walrus_baseline_artifact_is_well_formed(tmp_path):
    pytest.importorskip("thewalrus")
    from bench.walrus_baseline import run

    artifact, path = run(batch=8, repeats=2, warmup=1, out_dir=tmp_path)
    assert path.exists()
    assert artifact["kind"] == "walrus_baseline" and artifact["library"] == "thewalrus"
    # provenance block present (the reproducibility contract)
    assert {"commit", "container_digest", "hostname", "captured_utc"} <= set(artifact)
    funcs = {r["func"] for r in artifact["rows"]}
    assert funcs == {"perm", "haf", "lhaf", "tor"}
    for r in artifact["rows"]:
        assert r["evals_per_sec_median"] > 0 and "evals_per_sec_iqr" in r
