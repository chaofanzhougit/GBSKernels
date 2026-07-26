"""Rigorous real torontonian intervals using python-flint/Arb.

This module is deliberately independent of the production prefix-Cholesky
implementation and of the existing mpmath reference.  The dense evaluator
uses the defining subset sum

    tor(O) = sum_S (-1)^(n-|S|) / sqrt(det(I - O_S)),

where ``O_S`` selects ``[S, S+n]`` from a ``2n x 2n`` matrix in xxpp order.
Every input is first rounded to binary64, then converted to an exact rational
with ``float.as_integer_ratio``.  Arb supplies outward-rounded determinants,
square roots, and sums.

The returned endpoints are exact dyadic numbers.  Their mantissas are emitted
as strings by :meth:`TorontonianInterval.to_dict`, avoiding a second rounding
step in JSON consumers.
"""

from __future__ import annotations

import itertools
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, ClassVar, Iterator

import numpy as np
from flint import arb, arb_mat, ctx, fmpq

__all__ = [
    "ArbDomainError",
    "ArbPrecisionError",
    "DyadicEndpoint",
    "TorontonianInterval",
    "torontonian_arb",
    "torontonian_block_diagonal_arb",
]


class ArbDomainError(ValueError):
    """The real positive-square-root branch is proved to be unavailable."""


class ArbPrecisionError(RuntimeError):
    """The requested enclosure could not be certified at the precision cap."""


@dataclass(frozen=True)
class DyadicEndpoint:
    """An exact value ``mantissa * 2**exponent``."""

    mantissa: int
    exponent: int

    @classmethod
    def from_float(cls, value: float) -> "DyadicEndpoint":
        """Capture a finite binary64 value exactly."""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("a dyadic endpoint must be finite")
        numerator, denominator = value.as_integer_ratio()
        # A binary floating-point denominator is a power of two.
        exponent = -(denominator.bit_length() - 1)
        return cls(numerator, exponent)

    def to_fraction(self) -> Fraction:
        """Return the endpoint as an exact :class:`fractions.Fraction`."""
        if self.exponent >= 0:
            return Fraction(self.mantissa << self.exponent, 1)
        return Fraction(self.mantissa, 1 << -self.exponent)

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-safe exact representation."""
        return {"mantissa": str(self.mantissa), "exponent": self.exponent}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DyadicEndpoint":
        """Reconstruct an endpoint produced by :meth:`to_dict`."""
        return cls(int(payload["mantissa"]), int(payload["exponent"]))


@dataclass(frozen=True)
class TorontonianInterval:
    """A rigorous interval and the metadata needed to audit its construction.

    ``minimum_determinant_lower`` bounds every paired principal-subset
    determinant from below, including the empty-subset determinant one.
    """

    SCHEMA: ClassVar[str] = "gbskernels.torontonian-arb-interval.v1"

    lower: DyadicEndpoint
    upper: DyadicEndpoint
    minimum_determinant_lower: DyadicEndpoint
    precision_bits: int
    target_bits: int
    n_modes: int
    subset_count: int
    method: str

    def __post_init__(self) -> None:
        if self.lower.to_fraction() > self.upper.to_fraction():
            raise ValueError("interval lower endpoint exceeds upper endpoint")

    def contains(self, value: Fraction | int | float) -> bool:
        """Test containment, treating a float as its exact binary64 value."""
        if isinstance(value, float):
            if not math.isfinite(value):
                return False
            exact = Fraction.from_float(value)
        else:
            exact = Fraction(value)
        return self.lower.to_fraction() <= exact <= self.upper.to_fraction()

    def to_dict(self) -> dict[str, Any]:
        """Return a strict-JSON-compatible representation of the certificate."""
        return {
            "schema": self.SCHEMA,
            "method": self.method,
            "n_modes": self.n_modes,
            "subset_count": self.subset_count,
            "precision_bits": self.precision_bits,
            "target_bits": self.target_bits,
            "lower": self.lower.to_dict(),
            "upper": self.upper.to_dict(),
            "minimum_determinant_lower": self.minimum_determinant_lower.to_dict(),
        }


class _RetryAtHigherPrecision(Exception):
    pass


# python-flint's precision is process-global.  Serialising changes makes calls
# safe in threaded test runners; campaigns should parallelise across processes.
_PRECISION_LOCK = threading.RLock()


@contextmanager
def _workprec(bits: int) -> Iterator[None]:
    with _PRECISION_LOCK:
        previous = ctx.prec
        ctx.prec = bits
        try:
            yield
        finally:
            ctx.prec = previous


def _endpoint(value: arb) -> DyadicEndpoint:
    if not value.is_finite() or not value.is_exact():
        raise ArbPrecisionError("an Arb endpoint was not exact and finite")
    mantissa, exponent = value.man_exp()
    return DyadicEndpoint(int(mantissa), int(exponent))


def _as_real_binary64_matrix(value: Any, *, kind: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{kind} requires a rectangular real array") from exc
    if np.iscomplexobj(raw):
        raise ValueError(f"{kind} requires real binary64 input")
    try:
        matrix = np.ascontiguousarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{kind} requires real binary64 input") from exc
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{kind} requires finite binary64 input")
    return matrix


def _as_exact_rationals(matrix: np.ndarray) -> tuple[tuple[fmpq, ...], ...]:
    rows: list[tuple[fmpq, ...]] = []
    for row in matrix:
        exact_row = []
        for value in row:
            numerator, denominator = float(value).as_integer_ratio()
            exact_row.append(fmpq(numerator, denominator))
        rows.append(tuple(exact_row))
    return tuple(rows)


def _positive_inverse_sqrt(determinant: arb, *, label: str) -> tuple[arb, arb]:
    lower = determinant.lower()
    upper = determinant.upper()
    if upper <= 0:
        raise ArbDomainError(
            f"det(I - O_S) is proved nonpositive for {label}; "
            "the real torontonian branch is undefined"
        )
    if lower <= 0:
        raise _RetryAtHigherPrecision(
            f"det(I - O_S) still contains zero for {label}"
        )
    return arb(1) / determinant.sqrt(), lower


def _accuracy_is_sufficient(total: arb, scale: arb, target_bits: int) -> bool:
    """Require radius <= 2^-target_bits times a proved-positive scale."""
    radius = _endpoint(total.rad()).to_fraction()
    scale_lower = _endpoint(scale.lower()).to_fraction()
    if scale_lower <= 0:
        return False
    return radius <= scale_lower / (1 << target_bits)


def _make_interval(
    total: arb,
    minimum_determinant_lower: arb,
    *,
    precision_bits: int,
    target_bits: int,
    n_modes: int,
    subset_count: int,
    method: str,
) -> TorontonianInterval:
    return TorontonianInterval(
        lower=_endpoint(total.lower()),
        upper=_endpoint(total.upper()),
        minimum_determinant_lower=_endpoint(minimum_determinant_lower),
        precision_bits=precision_bits,
        target_bits=target_bits,
        n_modes=n_modes,
        subset_count=subset_count,
        method=method,
    )


def _validate_precision(
    initial_precision_bits: int, max_precision_bits: int, target_bits: int
) -> None:
    if initial_precision_bits < 64:
        raise ValueError("initial_precision_bits must be at least 64")
    if max_precision_bits < initial_precision_bits:
        raise ValueError("max_precision_bits must be at least initial_precision_bits")
    if target_bits < 0:
        raise ValueError("target_bits must be nonnegative")


_Evaluator = Callable[[], tuple[arb, arb, arb]]


def _adaptive_interval(
    evaluator: _Evaluator,
    *,
    initial_precision_bits: int,
    max_precision_bits: int,
    target_bits: int,
    n_modes: int,
    subset_count: int,
    method: str,
) -> TorontonianInterval:
    _validate_precision(initial_precision_bits, max_precision_bits, target_bits)
    precision = initial_precision_bits
    last_reason = "the requested target width was not reached"

    while True:
        with _workprec(precision):
            try:
                total, scale, minimum_determinant_lower = evaluator()
            except _RetryAtHigherPrecision as exc:
                last_reason = str(exc)
            else:
                if not total.is_finite() or not scale.is_finite():
                    last_reason = "the Arb evaluation was not finite"
                elif _accuracy_is_sufficient(total, scale, target_bits):
                    return _make_interval(
                        total,
                        minimum_determinant_lower,
                        precision_bits=precision,
                        target_bits=target_bits,
                        n_modes=n_modes,
                        subset_count=subset_count,
                        method=method,
                    )
                else:
                    last_reason = (
                        f"the enclosure did not reach {target_bits} "
                        "cancellation-normalized bits"
                    )

        if precision == max_precision_bits:
            raise ArbPrecisionError(
                f"could not certify the torontonian at {max_precision_bits} bits: "
                f"{last_reason}"
            )
        precision = min(2 * precision, max_precision_bits)


def torontonian_arb(
    O: Any,
    *,
    initial_precision_bits: int = 128,
    max_precision_bits: int = 2048,
    target_bits: int = 80,
) -> TorontonianInterval:
    """Return a rigorous interval for a real binary64 torontonian.

    This is the independent dense oracle.  It performs one Arb determinant per
    subset and therefore costs ``O(n^3 2^n)``; it is intended for modest mode
    counts.  If a determinant is not proved positive, precision is doubled up
    to ``max_precision_bits``.  A determinant proved nonpositive raises
    :class:`ArbDomainError` rather than choosing a complex square-root branch.
    """
    matrix = _as_real_binary64_matrix(O, kind="torontonian_arb")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"torontonian requires a square 2-D matrix, got {matrix.shape}")
    dimension = matrix.shape[0]
    if dimension % 2:
        raise ValueError(f"torontonian requires even dimension, got {dimension}")
    n_modes = dimension // 2
    exact_rows = _as_exact_rationals(matrix)

    def evaluate() -> tuple[arb, arb, arb]:
        rows = [[arb(value) for value in row] for row in exact_rows]
        total = arb(0)
        absolute_term_sum = arb(0)
        minimum_lower = arb(1)

        for subset_size in range(n_modes + 1):
            for subset in itertools.combinations(range(n_modes), subset_size):
                if subset_size == 0:
                    inverse_sqrt = arb(1)
                else:
                    indices = list(subset) + [mode + n_modes for mode in subset]
                    submatrix = arb_mat(
                        [
                            [
                                (arb(1) if row == column else arb(0))
                                - rows[indices[row]][indices[column]]
                                for column in range(2 * subset_size)
                            ]
                            for row in range(2 * subset_size)
                        ]
                    )
                    inverse_sqrt, determinant_lower = _positive_inverse_sqrt(
                        submatrix.det(), label=f"subset {subset}"
                    )
                    if determinant_lower < minimum_lower:
                        minimum_lower = determinant_lower

                if (n_modes - subset_size) % 2:
                    total -= inverse_sqrt
                else:
                    total += inverse_sqrt
                absolute_term_sum += inverse_sqrt

        return total, absolute_term_sum, minimum_lower

    return _adaptive_interval(
        evaluate,
        initial_precision_bits=initial_precision_bits,
        max_precision_bits=max_precision_bits,
        target_bits=target_bits,
        n_modes=n_modes,
        subset_count=1 << n_modes,
        method="dense-subset-determinants",
    )


def torontonian_block_diagonal_arb(
    blocks: Any,
    *,
    initial_precision_bits: int = 128,
    max_precision_bits: int = 2048,
    target_bits: int = 80,
) -> TorontonianInterval:
    """Rigorous ``O(n)`` reference for independent 2x2 mode blocks.

    ``blocks[i]`` is the matrix on coordinates ``(x_i, p_i)``.  In a dense
    xxpp matrix its entries occupy rows and columns ``(i, i+n)``.  With no
    coupling between distinct modes, the defining subset sum factorises as

    ``prod_i (1/sqrt(det(I_2 - blocks[i])) - 1)``.

    This helper evaluates that identity independently with Arb and is practical
    through at least 32 modes.  ``subset_count`` records the number of terms in
    the equivalent dense definition, not the amount of work performed here.
    """
    matrix = _as_real_binary64_matrix(blocks, kind="block-diagonal torontonian")
    if matrix.ndim != 3 or matrix.shape[1:] != (2, 2):
        raise ValueError(f"blocks must have shape (n, 2, 2), got {matrix.shape}")
    n_modes = matrix.shape[0]
    exact_blocks = tuple(_as_exact_rationals(block) for block in matrix)

    def evaluate() -> tuple[arb, arb, arb]:
        total = arb(1)
        # Expanding the product gives the absolute subset-term scale
        # prod_i(1 + 1/sqrt(det_i)), without enumerating its 2^n terms.
        absolute_term_sum = arb(1)
        # A subset determinant is a product of the selected block
        # determinants.  Multiplying every proved lower endpoint below one
        # therefore gives a lower bound for *all* subset determinants (the
        # empty subset included).  Reporting merely min(det_i) would be false
        # metadata when several det_i < 1.
        all_subset_determinant_lower = arb(1)

        for mode, block in enumerate(exact_blocks):
            entries = [[arb(value) for value in row] for row in block]
            identity_minus_block = arb_mat(
                [
                    [
                        (arb(1) if row == column else arb(0))
                        - entries[row][column]
                        for column in range(2)
                    ]
                    for row in range(2)
                ]
            )
            inverse_sqrt, determinant_lower = _positive_inverse_sqrt(
                identity_minus_block.det(), label=f"mode block {mode}"
            )
            if determinant_lower < 1:
                all_subset_determinant_lower = (
                    all_subset_determinant_lower * determinant_lower
                ).lower()
            total *= inverse_sqrt - 1
            absolute_term_sum *= inverse_sqrt + 1

        return total, absolute_term_sum, all_subset_determinant_lower

    return _adaptive_interval(
        evaluate,
        initial_precision_bits=initial_precision_bits,
        max_precision_bits=max_precision_bits,
        target_bits=target_bits,
        n_modes=n_modes,
        subset_count=1 << n_modes,
        method="factorized-mode-blocks",
    )
