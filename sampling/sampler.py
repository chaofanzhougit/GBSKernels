"""Conditional Gaussian-boson-sampling sampler -- the real photonics workload.

Draws photon-number samples from a zero-displacement GBS state by the chain rule
(sample mode 1, then mode 2 conditioned on it, ...). Each mode's conditional is a
*batch* of hafnians of growing submatrices -- exactly the ragged, sampler-shaped
workload :class:`gbskernels.Workspace` exists for. It is **hybrid host-orchestrated**:
``backend="gpu"`` evaluates each mode's hafnian *batch* on the GPU through one Workspace
reused across the chain (ragged bucketing + device-buffer residency), while the chain itself
-- prefix bookkeeping, the inverse-CDF draw, the submatrix gather -- runs on the host. So we
report **samples/sec**, not only kernel evals/sec. The *fully* on-device chain exists as
``resident=True`` (early: fp64-only, no diagnostics, cap 2·M·cutoff ≤ 20, no sieve).

Method (reduced-covariance chain rule). For a Gaussian state with covariance
``cov``, the marginal over the first ``k`` modes is the Gaussian state with the
reduced covariance, whose A-matrix is ``A_k = X (I - Q_k^{-1})`` (``Q_k`` the Husimi
matrix of the reduced covariance). The probability of a partial pattern ``n`` on
those modes is ``haf(A_k reduced by n) / (n! sqrt(det Q_k))``, so the conditional
for mode ``k``'s photon count ``j`` (given the already-sampled prefix) is

    P(j | prefix) ∝ haf(A_k[prefix, j]) / j!

over ``j = 0..cutoff``. Those reduced matrices are always **even**-sized (each mode
contributes a conjugate pair), so the GPU hafnian -- which is even-N -- handles them
directly. Validated distributionally against The Walrus and the exact distribution
(TV distance + chi-square) in tests/test_gbs_sampler.py.
"""

from __future__ import annotations

from math import factorial
from typing import Any

import numpy as np

import gbskernels

from .gbs import _qmat

__all__ = ["sample", "samples_per_second"]

_EMPTY = np.empty((0, 0), dtype=np.complex128)
# Largest submatrix the GPU hafnian kernels take; bigger ones (deep-photon prefixes)
# fall back to the CPU, so the GPU sampler is never capped by it. The DD kernel has a
# smaller buffer cap than FP64; 'auto' runs the FP64 pass (FP64 cap) and reruns risky
# elements itself (handling the DD cap internally), so its split is the FP64 cap too.
_GPU_CAPS = getattr(gbskernels, "_GPU_MAX_DIM", {})
_HAF_CAP = _GPU_CAPS.get("haf", 20)         # FP64 (and 'auto') GPU/CPU split
_HAF_DD_CAP = _GPU_CAPS.get("haf_dd", 16)   # DD kernel buffer cap

# Over-cap matrices run on the CPU; the fallback must NOT silently downgrade a requested
# high-precision tier to plain CPU FP64. The CPU has no double-double, so 'dd' maps to the
# (strictly more accurate) mpmath 'ref'; 'fp64'/'auto'/'ref' map to themselves.
_CPU_FALLBACK = {"fp64": "fp64", "auto": "auto", "dd": "ref", "ref": "ref"}


def _gpu_haf_cap(precision: str) -> int:
    return _HAF_DD_CAP if precision == "dd" else _HAF_CAP


def _eval_haf_batch(mats: list[np.ndarray], ws: Any, precision: str) -> np.ndarray:
    """Hafnians of a ragged batch: the Workspace (GPU) for matrices within the GPU cap
    for ``precision``, the CPU for any that exceed it (deep prefixes). The CPU fallback
    preserves the requested precision -- a requested 'dd'/'auto' is never silently served
    as CPU FP64 (the over-cap elements use a CPU tier at least as accurate; see
    ``_CPU_FALLBACK``). With no Workspace (CPU backend), everything goes to the CPU."""
    if ws is None:
        return gbskernels.haf_batched(mats, precision=precision, backend="cpu")
    cap = _gpu_haf_cap(precision)
    big = [i for i, M in enumerate(mats) if M.shape[0] > cap]
    if not big:
        return ws.haf_batched(mats, precision=precision)
    out = np.empty(len(mats), dtype=np.complex128)
    big_set = set(big)
    small_idx = [i for i in range(len(mats)) if i not in big_set]
    if small_idx:
        res = ws.haf_batched([mats[i] for i in small_idx], precision=precision)
        for k, i in enumerate(small_idx):
            out[i] = res[k]
    cpu_prec = _CPU_FALLBACK[precision]
    res_big = gbskernels.haf_batched([mats[i] for i in big], precision=cpu_prec, backend="cpu")
    for k, i in enumerate(big):
        out[i] = res_big[k]
    return out


def _reduced_A(cov: np.ndarray, k: int, hbar: float) -> np.ndarray:
    """A-matrix ``X (I - Q_k^{-1})`` of the state reduced to the first ``k`` modes."""
    M = cov.shape[0] // 2
    ix = list(range(k)) + list(range(M, M + k))   # x_1..x_k, p_1..p_k (xxpp)
    Q = _qmat(cov[np.ix_(ix, ix)], hbar)
    X = np.block([[np.zeros((k, k)), np.eye(k)], [np.eye(k), np.zeros((k, k))]])
    return X @ (np.eye(2 * k) - np.linalg.inv(Q))


def _conditional_weights(A: np.ndarray, uniq: np.ndarray, k: int, n_terms: int,  # noqa: PLR0913
                         inv_fac: np.ndarray, ws: Any, precision: str,
                         repeated_sieve: bool = False, certified: bool = False):
    """Unnormalized chain-rule conditional weights ``clip(haf(A_sub), 0) / j!`` for the
    photon count ``j = 0 .. n_terms-1`` of mode ``k``, for every unique prefix in ``uniq``
    -- a ``(len(uniq), n_terms)`` array. The (always even) reduced submatrix's hafnian is
    the unnormalized conditional probability; the caller renormalizes over the kept range.
    The whole ragged set of growing submatrices is one batched evaluation (bucketed on the
    GPU through the Workspace; the rest of the sampler is vectorized over draws)."""
    if repeated_sieve:
        # R4: the same conditionals in their native repeated-row shape -- the
        # pattern over A's 2k indices is (pre..., j, pre..., j), so each weight
        # is one sieve evaluation at cost prod(n_i + 1) instead of a 2^(N/2)
        # power-trace on the expansion (gamma = 0: zero displacement).
        gam = np.zeros(2 * k, dtype=np.complex128)
        reps = np.empty((len(uniq) * n_terms, 2 * k), dtype=np.int32)
        r = 0
        for pre in uniq:
            base = np.concatenate([pre[: k - 1], [0], pre[: k - 1], [0]]).astype(np.int32)
            for j in range(n_terms):
                base[k - 1] = base[2 * k - 1] = j
                reps[r] = base
                r += 1
        backend_ = "gpu" if ws is not None else "cpu"
        if certified:
            # certified: values are BIT-IDENTICAL to the plain sieve (so draws are
            # unchanged); each weight additionally carries a rigorous bound.
            vals, d = gbskernels.lhaf_repeated(np.ascontiguousarray(A), gam, reps,
                                               backend=backend_, certified=True)
            b = np.asarray(d["abs_error_bound"], dtype=np.float64)
            hafs = np.real(vals).reshape(len(uniq), n_terms)
            # |Im| is certified junk on a real weight: fold it into the width.
            widths = (b + np.abs(np.imag(vals))).reshape(len(uniq), n_terms)
            w = np.clip(hafs, 0.0, None) * inv_fac[:n_terms]
            # inv_fac is a double (rel err <= u) -> widen; clip is monotone-safe
            u_ = 2.0 ** -53
            wid = widths * inv_fac[:n_terms] * (1.0 + 4.0 * u_) + u_ * w
            return w, wid
        if ws is not None:
            # GPU: ONE kernel launch per mode-step -- the whole (prefix, j)
            # grid as a reps table on the shared reduced A (core/repeated.cu's
            # native shape). Off-GPU sizes cannot occur (the sieve has no
            # expansion-dim cap; guards are photons/terms, checked below).
            vals = gbskernels.lhaf_repeated(np.ascontiguousarray(A), gam, reps,
                                            backend="gpu")
            hafs = np.real(vals).reshape(len(uniq), n_terms)
            return np.clip(hafs, 0.0, None) * inv_fac[:n_terms]
        import cpu_ref
        hafs = np.empty((len(uniq), n_terms))
        for u, pre in enumerate(uniq):
            base = np.concatenate([pre[: k - 1], [0], pre[: k - 1], [0]]).astype(np.int64)
            for j in range(n_terms):
                base[k - 1] = base[2 * k - 1] = j
                hafs[u, j] = float(np.real(cpu_ref.lhaf_repeated(A, gam, base)))
        return np.clip(hafs, 0.0, None) * inv_fac[:n_terms]
    mats: list[np.ndarray] = []
    for pre in uniq:                                       # only loop: over unique prefixes
        a = [i for i in range(k - 1) for _ in range(int(pre[i]))]       # a-block (fixed)
        ad = [i + k for i in range(k - 1) for _ in range(int(pre[i]))]  # a-dagger block
        for j in range(n_terms):
            idx = a + [k - 1] * j + ad + [2 * k - 1] * j
            mats.append(A[np.ix_(idx, idx)] if idx else _EMPTY)
    hafs = np.real(_eval_haf_batch(mats, ws, precision)).reshape(len(uniq), n_terms)
    return np.clip(hafs, 0.0, None) * inv_fac[:n_terms]


def _tail_estimate(W: np.ndarray, cutoff: int, inv: np.ndarray, num_samples: int):
    """Per-mode ESTIMATE of the discarded conditional tail mass, aggregated over draws.

    NOT a rigorous upper bound: it is a tail **estimate under a geometric-continuation
    assumption** (that the per-parity decay observed at the end of the kept range continues
    past the cutoff). For a general Gaussian conditional that continuation is not guaranteed
    without a theorem, so this is a calibrated heuristic, validated empirically against the
    measured TV in tests -- not a proof.

    GBS conditionals decay geometrically *per photon pair* (each extra pair costs another
    ~``tanh^2 r`` factor) with an even/odd parity structure, so *consecutive* terms are not
    monotone but each same-parity subsequence is. Estimate each parity's decay ratio from
    the last same-parity pair INSIDE the kept range (free -- no extra hafnians), then continue
    that parity's discarded tail as a geometric series past the cutoff:

        sigma = w_top / w_{top-2}                       (last kept same-parity ratio)
        tail  = sigma * w_top / (1 - sigma)            (if sigma < 1; else not decaying -> inf)

    for ``top = cutoff`` (its parity) and ``top = cutoff-1`` (the other), then the discarded
    fraction is ``T/(K+T)``. ``tail_not_decaying`` flags any occurring prefix with a ratio
    >= 1 (cutoff too small to estimate -> that prefix's estimate is 1). Needs ``cutoff >= 3``
    (four kept terms); below that the estimate is reported vacuous (1)."""
    K = W.sum(axis=1)
    counts = np.bincount(inv, minlength=W.shape[0]).astype(float)
    occ = counts > 0
    if cutoff < 3:                                 # too few terms to estimate a pair ratio
        return (1.0 if np.any(occ) else 0.0), (1.0 if np.any(occ) else 0.0), True

    # A boundary term this far below the kept mass is float noise (the cutoff sits past the
    # photon support); its parity tail is then negligible, NOT vacuous -- guard the 0/0 ratio
    # so a generously large cutoff is not spuriously flagged as "not decaying".
    floor = 1e-9 * np.maximum(K, 1e-300)

    def parity_tail(top: np.ndarray, prev: np.ndarray):
        neg = top <= floor
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.where(prev > 0, top / prev, np.where(top > 0, np.inf, 0.0))
            geom = np.where(sigma < 1.0, sigma * top / np.maximum(1.0 - sigma, 1e-300), np.inf)
        t = np.where(neg, top, np.where(top > 0, geom, 0.0))    # inf only for non-negligible non-decay
        flag = (sigma >= 1.0) & (top > 0) & (~neg)
        return t, flag

    t_a, f_a = parity_tail(W[:, cutoff], W[:, cutoff - 2])       # parity of the cutoff
    t_b, f_b = parity_tail(W[:, cutoff - 1], W[:, cutoff - 3])   # the other parity
    growing = bool(np.any(f_a) or np.any(f_b))
    T = t_a + t_b                                                # inf if either parity tail vacuous
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.isinf(T), 1.0, T / np.where(K + T > 0, K + T, 1.0))
    mean = float((counts * frac).sum() / num_samples) if num_samples else 0.0
    mx = float(frac[occ].max()) if np.any(occ) else 0.0
    return mean, mx, growing


def _sample_resident_gpu(cov: np.ndarray, num_samples: int, cutoff: int, hbar: float,
                         seed: int | None) -> np.ndarray:
    """v3 FULLY on-device sampler: the whole chain (prefix state, submatrix gather, conditional
    hafnians, inverse-CDF + cuRAND draw) runs on the GPU via ``gbskernels_ext.sample_resident`` --
    no per-mode host round trip. The host only computes the reduced A-matrices {A_k} once and reads
    the samples back. cuRAND != numpy RNG, so this is distributionally equivalent to the hybrid /
    CPU sampler, NOT seed-identical. Restricted to submatrices within the GPU hafnian cap (worst
    case 2*M*cutoff <= cap); larger configs use the hybrid path (resident=False)."""
    ext = gbskernels._load_gpu_ext()
    if ext is None or not hasattr(ext, "sample_resident"):
        raise RuntimeError("resident=True needs the GPU extension with sample_resident "
                           "(rebuild bindings/, or use resident=False).")
    cov = np.asarray(cov, dtype=np.float64)
    M = cov.shape[0] // 2
    cap = _HAF_CAP
    if 2 * M * cutoff > cap:
        raise ValueError(
            f"resident sampler: worst-case submatrix 2*M*cutoff={2 * M * cutoff} exceeds the "
            f"hafnian cap {cap}. Use resident=False (the hybrid path falls back to the CPU for "
            "deep-photon prefixes), or reduce modes/cutoff.")
    A_pref = [_reduced_A(cov, k, hbar) for k in range(1, M + 1)]
    flat, off, tot = [], [], 0
    for A in A_pref:
        a = np.ascontiguousarray(A, dtype=np.complex128).reshape(-1)
        off.append(tot); flat.append(a); tot += a.size
    ak = np.ascontiguousarray(np.concatenate(flat)) if flat else np.zeros(0, np.complex128)
    offs = np.ascontiguousarray(np.array(off, dtype=np.int32))
    invfac = np.ascontiguousarray(np.array([1.0 / factorial(j) for j in range(cutoff + 1)], dtype=np.float64))
    out = ext.sample_resident(ak, offs, int(M), int(num_samples), int(cutoff), int(cap),
                              invfac, 0 if seed is None else int(seed))
    return np.asarray(out, dtype=np.int64)


def sample(
    cov: np.ndarray,
    num_samples: int,
    cutoff: int = 6,
    hbar: float = 2.0,
    backend: str = "cpu",
    precision: str = "fp64",
    seed: int | None = None,
    return_diagnostics: bool = False,
    resident: bool = False,
    repeated_sieve: bool | None = None,
    certified_weights: bool = False,
):
    """Draw ``num_samples`` photon-number samples (shape ``(num_samples, M)``).

    **Target distribution (cutoff semantics, defined).** This draws from the *sequentially
    per-mode-truncated* chain-rule GBS distribution: mode ``k``'s photon count is drawn
    from the exact conditional ``P(n_k | prefix)`` **restricted to ``{0, 1, ..., cutoff}``
    and renormalized**, given the already-sampled prefix ``n_1..n_{k-1}``. As
    ``cutoff -> inf`` this is the exact GBS distribution; at finite ``cutoff`` it discards,
    at each mode, the conditional tail mass above ``cutoff``. It is **not** identical to
    sampling the *globally* cutoff-truncated joint (which renormalizes once over the joint
    support); the two differ by that discarded tail mass.

    **Bias estimate (quantified).** The total-variation distance to the exact distribution is
    bounded by the sum over modes of the expected discarded conditional tail mass. With
    ``return_diagnostics=True`` the sampler returns ``(samples, diag)`` where ``diag`` carries
    a per-pair geometric-tail **estimate** of that discarded mass -- per mode and summed
    (``tv_bias_estimate_mean`` / ``tv_bias_estimate_max``), from the kept conditional terms at
    no extra cost (see :func:`_tail_estimate`). It is an estimate **under a geometric-
    continuation assumption**, NOT a rigorous bound (it is validated empirically against the
    measured TV in tests, not proven), so the cutoff is chosen against a measured quantity
    rather than blindly assumed (docs/DESIGN.md §8). Diagnostics never change the drawn samples.

    ``backend="gpu"`` runs every mode's conditional-probability hafnian batch through one
    :class:`gbskernels.Workspace` (ragged bucketing + buffer residency across the chain);
    over-cap submatrices fall back to the CPU at a precision at least as accurate as
    ``precision`` (a requested 'dd'/'auto' is never silently downgraded to CPU fp64).
    """
    if precision not in ("fp64", "dd", "auto", "ref"):
        raise ValueError(f"unknown precision {precision!r}; expected fp64/dd/auto/ref")
    if certified_weights:
        # Resolve the certificate's sieve requirement FIRST -- before the generic
        # None->auto default below, which would otherwise turn None into False on
        # the CPU/non-eligible configs and then wrongly reject certified_weights.
        if resident:
            raise NotImplementedError("certified_weights=True needs the sieve chain; "
                                      "the resident sampler has no certified path.")
        if precision != "fp64":
            raise NotImplementedError("certified_weights=True is the certified-fp64 sieve; "
                                      "use precision='fp64'.")
        if repeated_sieve is False:        # explicit opt-out conflicts with the certificate
            raise ValueError("certified_weights=True requires the sieve "
                             "(repeated_sieve True or None, not False).")
        repeated_sieve = True              # None or True -> the certificate rides the sieve
    if repeated_sieve is None:
        # Data-backed default (4090 + A100 sweeps at HEAD, results/sampling/
        # 2026-07-04/05): the GPU sieve chain wins 2.8x-189x at cutoff >= 4 for
        # modes <= 8 on BOTH cards, and LOSES at the widest shallow cell on both
        # (m=10/c=4: -12% on the 4090, -59% on the A100) -- hence the modes gate.
        # The CPU sieve is python-overhead-bound and stays opt-in. Explicit
        # True/False always overrides.
        repeated_sieve = (backend == "gpu" and not resident
                          and precision == "fp64" and cutoff >= 4
                          and len(cov) // 2 <= 8)
    if repeated_sieve and precision != "fp64":
        raise NotImplementedError(
            "repeated_sieve=True is an fp64 path (the sieve has no dd/auto/ref "
            "variant yet).")
    if backend == "cpu" and precision == "dd":
        raise ValueError(
            "precision='dd' is a GPU tier (no CPU double-double); use backend='gpu', "
            "or precision='auto'/'fp64'/'ref' on the CPU backend."
        )
    if resident:   # v3: the fully on-device chain (different RNG -> distributionally equivalent)
        if backend != "gpu":
            raise ValueError("resident=True requires backend='gpu' (the fully on-device sampler).")
        if repeated_sieve:
            raise NotImplementedError(
                "resident=True evaluates its conditionals with the on-device variable-N "
                "hafnian chain; the repeated-row sieve is not wired into that chain yet. "
                "Use resident=False with repeated_sieve=True, or resident=True without it."
            )
        if precision != "fp64":
            raise ValueError("the resident sampler is fp64-only; use precision='fp64' (or resident=False).")
        if return_diagnostics:
            raise ValueError("the resident sampler does not produce diagnostics yet; use resident=False.")
        return _sample_resident_gpu(cov, num_samples, cutoff, hbar, seed)
    cov = np.asarray(cov, dtype=np.float64)
    M = cov.shape[0] // 2
    rng = np.random.default_rng(seed)
    out = np.zeros((num_samples, M), dtype=np.int64)
    inv_fac = np.array([1.0 / factorial(j) for j in range(cutoff + 1)])
    A_pref = [_reduced_A(cov, k, hbar) for k in range(1, M + 1)]
    disc_mean = np.zeros(M)            # per-mode discarded-mass estimate (geometric continuation)
    disc_max = np.zeros(M)
    tv_kept_max = np.zeros(M)          # certified kept-part TV bounds (certified_weights)
    tv_kept_mean = np.zeros(M)
    tail_growing = False

    ws = gbskernels.Workspace(backend="gpu") if backend == "gpu" else None
    try:
        for k in range(1, M + 1):            # sample mode k (1-indexed) for all draws
            A = A_pref[k - 1]
            # The conditional for a draw depends only on its prefix (modes 1..k-1), and many
            # draws share a prefix -- so compute each DISTINCT prefix's conditional once
            # (np.unique groups in C); the gather + inverse-CDF is vectorized over draws.
            uniq, inv = np.unique(out[:, : k - 1], axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            cw = _conditional_weights(A, uniq, k, cutoff + 1, inv_fac, ws, precision,
                                      repeated_sieve=repeated_sieve,
                                      certified=certified_weights)
            if certified_weights:
                Wk, Wwidth = cw
                # per-prefix kept-mass certificate: Z in [Z - sum(width), Z + ...];
                # TV of the drawn (kept, renormalized) conditional from the exact
                # kept conditional <= sum(width_j) / Z_lo  (interval arithmetic,
                # slack absorbed; inf if the interval cannot exclude Z = 0).
                Zs = Wk.sum(axis=1)
                Wd = Wwidth.sum(axis=1)
                Zlo = Zs - Wd
                with np.errstate(divide="ignore", invalid="ignore"):
                    tv_u = np.where(Zlo > 0.0, Wd / np.maximum(Zlo, 1e-300), np.inf)
                # weight per prefix by how many draws use it (inv not yet gathered
                # for this k: compute after uniq/inv exist -- inv is in scope)
                counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
                tv_kept_max[k - 1] = float(np.max(tv_u))
                tv_kept_mean[k - 1] = float(np.sum(tv_u * counts) / max(counts.sum(), 1.0))
            else:
                Wk = cw
            wuniq = Wk.copy()
            wuniq /= wuniq.sum(axis=1, keepdims=True)         # (U, cutoff+1)
            W = wuniq[inv]                                    # (num_samples, cutoff+1) gather
            u = rng.random(num_samples)                       # inverse-CDF sampling per draw
            out[:, k - 1] = (np.cumsum(W, axis=1) < u[:, None]).sum(axis=1)
            if return_diagnostics:
                disc_mean[k - 1], disc_max[k - 1], grew = _tail_estimate(Wk, cutoff, inv, num_samples)
                tail_growing = tail_growing or grew
    finally:
        if ws is not None:
            ws.close()
    if not return_diagnostics:
        return out
    diag = {
        "definition": "sequential per-mode truncation: mode k ~ P(n_k|prefix) restricted "
                      "to {0..cutoff} and renormalized",
        "cutoff": cutoff,
        "tail_estimate_method": "per-pair geometric-tail ESTIMATE (not a rigorous bound) of "
                                "each conditional's discarded mass, from the kept terms, under "
                                "the assumption that the observed per-parity decay continues; "
                                "tail_not_decaying flags a same-parity ratio >= 1",
        "per_mode_discarded_mean": disc_mean.tolist(),
        "per_mode_discarded_max": disc_max.tolist(),
        **({"kept_mass_certified": {
                "meaning": "RIGOROUS per-mode bounds (certified-fp64 sieve) on the TV "
                           "distance between the drawn kept-and-renormalized conditional "
                           "and the exact one -- the KEPT part is proven; the tail above "
                           "the cutoff remains the geometric ESTIMATE above",
                "per_mode_tv_bound_max": tv_kept_max.tolist(),
                "per_mode_tv_bound_mean": tv_kept_mean.tolist(),
                "total_tv_kept_bound": float(tv_kept_max.sum()),
            }} if certified_weights else {}),
        "tv_bias_estimate_mean": float(disc_mean.sum()),
        "tv_bias_estimate_max": float(disc_max.sum()),
        "tail_not_decaying": bool(tail_growing),
    }
    return out, diag


def samples_per_second(
    cov: np.ndarray, num_samples: int, cutoff: int = 6, backend: str = "cpu",
    hbar: float = 2.0, seed: int = 0, resident: bool = False,
    repeated_sieve: bool | None = None,
) -> dict[str, Any]:
    """Time :func:`sample` and report end-to-end samples/sec (not kernel evals/sec).

    ``repeated_sieve=None`` (default) times exactly what ``sample()`` does by
    default -- including its auto-sieve resolution -- so the reported number IS
    the default path's; the row records ``repeated_sieve_effective``. Explicit
    ``True``/``False`` pin the path for A/B rows (suffixed ``+sieve`` /
    ``+nosieve``). ``resident=True`` times the v3 on-device chain
    (``gpu-resident``)."""
    import time
    t0 = time.perf_counter()
    s = sample(cov, num_samples, cutoff=cutoff, hbar=hbar, backend=backend, seed=seed,
               resident=resident, repeated_sieve=repeated_sieve)
    dt = time.perf_counter() - t0
    label = "gpu-resident" if resident else backend
    if repeated_sieve is True:     # explicitly pinned A/B rows are suffixed;
        label += "+sieve"          # the None (default-path) row is unsuffixed
    elif repeated_sieve is False:
        label += "+nosieve"
    effective = repeated_sieve
    if effective is None:          # mirror sample()'s auto resolution for the record
        effective = (backend == "gpu" and not resident and cutoff >= 4
                     and len(cov) // 2 <= 8)
    return {"backend": label, "repeated_sieve_effective": bool(effective),
            "num_samples": num_samples, "modes": cov.shape[0] // 2,
            "cutoff": cutoff, "seconds": dt, "samples_per_sec": num_samples / dt if dt else float("inf"),
            "mean_photons": float(s.sum(axis=1).mean())}
