"""GBSKernels end-to-end demo — kernels, precision tiers, and a GBS sampling run.

Runs on the CPU backend with the wheel's only dependencies (numpy + mpmath); no
GPU or extra packages needed. On a machine with the CUDA extension built, every
call shown here also accepts ``backend="gpu"``. Takes ~10 seconds:

    uv run python examples/gbs_demo.py         # or: python examples/gbs_demo.py

What it shows, in order:
 1. the four functions against exact closed forms (Layer-1 style checks);
 2. the precision tiers — and ``precision="auto"`` catching a deliberately
    cancellation-heavy input that silently breaks plain FP64 (docs/DESIGN.md §6);
 3. a Gaussian-boson-sampling distribution computed with one batched hafnian
    call, summing to ~1 (docs/DESIGN.md §8, Layer 4);
 4. the conditional sampler drawing photon-number samples, cross-checked
    against the exact distribution (total-variation distance ~ sampling noise)
    with its truncation-bias diagnostic.
"""

from __future__ import annotations

from math import factorial

import numpy as np

import gbskernels
from sampling import gbs, sampler

HBAR = 2.0


def pure_gbs_state(m: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A pure GBS state: Haar interferometer U + modest squeezing r.

    Returns ``(B, r, cov)`` — the hafnian kernel ``B = U diag(tanh r) U^T`` and the
    xxpp covariance ``cov = (hbar/2) S S^T`` with ``S`` the symplectic of the
    squeezers followed by the interferometer (numpy-only; agrees with The Walrus's
    ``symplectic`` construction to machine precision).
    """
    g = np.random.default_rng(seed)
    r = g.uniform(0.15, 0.35, m)
    z = (g.standard_normal((m, m)) + 1j * g.standard_normal((m, m))) / np.sqrt(2.0)
    U, rr = np.linalg.qr(z)
    ph = np.diagonal(rr).copy()
    ph /= np.abs(ph)
    U = U * ph
    B = U @ np.diag(np.tanh(r)) @ U.T
    X, Y = U.real, U.imag
    S = np.block([[X, -Y], [Y, X]]) @ np.diag(np.concatenate([np.exp(-r), np.exp(r)]))
    cov = (HBAR / 2.0) * S @ S.T
    return B, r, cov


def main() -> None:
    print("== 1. the four functions vs exact closed forms ==")
    J4 = np.ones((4, 4))
    print(f"perm(J_4)  = {gbskernels.perm(J4).real:.1f}   (exact: 4! = 24)")
    print(f"haf(J_4)   = {gbskernels.haf(J4).real:.1f}    (exact: 3!! = 3 perfect matchings of K_4)")
    print(f"lhaf(J_4)  = {gbskernels.lhaf(J4).real:.1f}   (exact: 10 = involution number a.k.a. telephone number T(4))")
    a = 0.2
    O = a * np.eye(4)                       # 4x4 xxpp -> 2 modes
    tor_exact = (a / (1 - a)) ** 2          # tor(a*I) = (a/(1-a))^n_modes, exact
    print(f"tor(0.2*I) = {gbskernels.tor(O).real:.6f}   (exact: (0.2/0.8)^2 = {tor_exact:.6f})")

    print("\n== 2. precision tiers, and 'auto' catching FP64 cancellation ==")
    from bench._inputs import cancellation_hafnian

    A_bad = cancellation_hafnian(1e-10, seed=0)  # tuned so the subset terms nearly cancel
    fp64 = gbskernels.haf(A_bad)                       # fast, silently inaccurate here
    exact = gbskernels.haf(A_bad, precision="ref")     # mpmath ground truth
    val, diag = gbskernels.haf(A_bad, precision="auto", return_diagnostics=True)
    rel = abs(fp64 - exact) / abs(exact)
    rel_auto = abs(val - exact) / abs(exact)
    print(f"fp64:  rel err = {rel:.1e}   <- catastrophic cancellation, no warning")
    print(f"auto:  rel err = {rel_auto:.1e}   tier={diag['tier']!r}, cancellation kappa = {diag['cancellation']:.1e}")
    print("(kappa flagged the FP64 value as risky, so 'auto' reran it in the high-precision tier)")
    _, cert = gbskernels.haf(A_bad, precision="certified", return_diagnostics=True)
    print(f"certified: rel err <= {cert['rel_error_bound']:.1e}  -- a rigorous bound on the "
          f"same FP64 value (true rel err {rel:.1e}), not a heuristic")

    print("\n== 3. a GBS distribution from one batched hafnian call ==")
    m, cutoff = 3, 4
    B, r, cov = pure_gbs_state(m, seed=11)
    patterns, probs = gbs.probabilities(B, r, cutoff=cutoff)
    top = np.argsort(probs)[::-1][:4]
    print(f"{len(patterns)} patterns (m={m} modes, cutoff={cutoff}); total probability = {probs.sum():.6f}")
    for i in top:
        print(f"  P{patterns[i]} = {probs[i]:.4f}")

    print("\n== 4. drawing samples with the conditional sampler ==")
    N = 1500
    samples, sdiag = sampler.sample(cov, N, cutoff=cutoff, seed=1, return_diagnostics=True)
    emp: dict[tuple[int, ...], float] = {}
    for s in samples:
        t = tuple(int(v) for v in s)
        emp[t] = emp.get(t, 0.0) + 1.0 / N
    exact_p = {tuple(p): float(q) for p, q in zip(patterns, probs)}
    tv = 0.5 * sum(abs(emp.get(k, 0.0) - exact_p.get(k, 0.0)) for k in set(emp) | set(exact_p))
    print(f"{N} samples; TV(empirical, exact) = {tv:.3f}  (sampling noise ~ {1/np.sqrt(N):.3f})")
    print(f"truncation-bias estimate (sequential per-mode cutoff): {sdiag['tv_bias_estimate_mean']:.2e}")

    kind = gbskernels.gpu_backend_kind()  # "gpu" | "host-shim" | "none"
    print(f"\nGPU extension: {kind}"
          + ("" if kind == "gpu"
             else "  (every call above also takes backend='gpu' on a CUDA build)"))


if __name__ == "__main__":
    main()
