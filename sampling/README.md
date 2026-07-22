# sampling/ — boson-sampling orchestration

A thin layer over the kernels that exercises them on the distributions photonic
sampling produces, and where the end-to-end statistical validation attaches
([`docs/DESIGN.md`](../docs/DESIGN.md) §8).

- **`boson_sampling.py`** — standard (Aaronson–Arkhipov) boson sampling via the
  **permanent**: `P(T) = |perm(U[T, S])|² / (∏ tᵢ! ∏ sᵢ!)`, validated to sum to
  one by unitarity.
- **`gbs.py`** — Gaussian boson sampling across the three detector regimes, one
  batched call per distribution:
  - `probabilities` (number-resolving, zero displacement) via the **hafnian**,
    `P(n̄) = |haf(B_n̄)|² / (n̄! ∏ cosh r)`;
  - `displaced_probabilities` (displaced GBS) via the **loop hafnian**, with the
    displacement carried as diagonal loop weights;
  - `torontonian_threshold_probabilities` (threshold detectors) via the
    **torontonian**, `tor(O_c) / √det Q`.
- **`gbs.threshold_O_xxpp`** — constructs the exact real quadrature-basis
  threshold matrix `Ox = I - Sigma_x^{-1}` and `log(sqrt(det Q))`. It is the
  real-domain input used by `gbskernels.tor_single`; taking the entrywise real
  part of the complex-basis matrix is not equivalent for complex correlations.
- **`sampler.py`** — the conditional sampler. `sample(cov, num_samples, cutoff,
  backend=..., precision=..., ...)` draws photon-number samples by the
  reduced-covariance chain rule. It is hybrid host-orchestrated: the host drives
  the chain (prefix bookkeeping, inverse-CDF draw, submatrix gather) while each
  mode's conditional — a ragged batch of hafnians of growing submatrices — is
  evaluated on the GPU through a reused `gbskernels.Workspace`. Over-cap
  submatrices fall back to the CPU at a precision at least as accurate as
  requested. The cutoff is a sequential per-mode truncation, and
  `return_diagnostics=True` returns an estimate of the discarded tail mass. A
  fully on-device variant is available as `resident=True` (double precision,
  within a submatrix-size cap). The `repeated_sieve` path evaluates collision
  patterns without materializing expanded matrices; `certified_weights=True`
  adds rigorous kept-mass weight bounds while leaving the cutoff-tail estimate
  explicitly heuristic.

**Validation.** Sum-to-one, vacuum and odd-parity behavior, marginals, and
batched-equals-looped identities; per-pattern agreement with The Walrus in all
three regimes; the displaced distribution reduces to the hafnian case as the
displacement vanishes; and the torontonian threshold distribution matches the
marginalized hafnian distribution. The conditional sampler is validated
distributionally: its empirical distribution matches the exact distribution and
The Walrus's independent sampler to within sampling noise (total-variation
distance and a chi-square test), and the GPU-routed chain reproduces the CPU
samples exactly. The Walrus high-level oracle is used through a small NumPy-2.0
compatibility shim in the relevant tests.

**Scope.** The sampler is for pure states; loss and mixed-state matrices are
characterized at the kernel level. The throughput advantage is on the GPU in the
deep-photon regime (large hafnians); at modest squeezing the chain is inexpensive
on the CPU.
