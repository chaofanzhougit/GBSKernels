from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import gbskernels


@pytest.mark.parametrize("dd", [False, True])
@pytest.mark.parametrize(
    ("value", "bound"),
    [
        (np.inf, 0.0),
        (-np.inf, 0.0),
        (1.0, np.nan),
        (1.0, np.inf),
        (1.0, -np.inf),
        (1.0, -1.0),
    ],
)
def test_tor_single_certified_rejects_invalid_native_certificate(
    monkeypatch, dd, value, bound
):
    ext = SimpleNamespace(
        tor_single=lambda matrix, groups: 0.0,
        tor_single_certified=lambda matrix, groups: (value, bound),
        tor_single_ddcertified=lambda matrix, groups: (value, bound),
    )
    monkeypatch.setattr(gbskernels, "_load_gpu_ext", lambda: ext)

    with pytest.raises(ValueError, match="uncertifiable"):
        gbskernels.tor_single(np.zeros((2, 2)), certified=not dd, dd=dd)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_tor_single_plain_rejects_nonfinite_native_value(monkeypatch, value):
    ext = SimpleNamespace(tor_single=lambda matrix, groups: value)
    monkeypatch.setattr(gbskernels, "_load_gpu_ext", lambda: ext)

    with pytest.raises(ValueError, match="physical domain"):
        gbskernels.tor_single(np.zeros((2, 2)))


@pytest.mark.parametrize("target", ["matrix", "gamma"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_lhaf_repeated_certified_rejects_nonfinite_input(target, bad):
    matrix = np.array([[0.25 + 0.0j]])
    gamma = np.array([0.0 + 0.0j])
    if target == "matrix":
        matrix[0, 0] = bad
    else:
        gamma[0] = bad

    with pytest.raises(ValueError, match="finite input"):
        gbskernels.lhaf_repeated(matrix, gamma, np.array([2]), certified=True)


@pytest.mark.parametrize(
    ("value", "bound"),
    [
        (complex(np.nan, 0.0), 0.0),
        (complex(np.inf, 0.0), 0.0),
        (0.0j, np.nan),
        (0.0j, np.inf),
        (0.0j, -1.0),
    ],
)
def test_lhaf_repeated_certified_rejects_invalid_native_certificate(
    monkeypatch, value, bound
):
    ext = SimpleNamespace(
        lhaf_repeated=lambda matrix, gamma, reps: np.array([0.0j]),
        lhaf_repeated_certified=lambda matrix, gamma, reps: (
            np.array([value], dtype=np.complex128),
            np.array([bound], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(gbskernels, "_load_gpu_ext", lambda: ext)

    with pytest.raises(FloatingPointError, match="invalid bound|non-finite value"):
        gbskernels.lhaf_repeated(
            np.array([[0.25 + 0.0j]]),
            np.array([0.0 + 0.0j]),
            np.array([2]),
            backend="gpu",
            certified=True,
        )
