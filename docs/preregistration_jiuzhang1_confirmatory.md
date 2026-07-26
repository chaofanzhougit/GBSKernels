# Historical private plan — Jiuzhang 1.0 squashed-vs-squeezed campaign

> **Historical status (2026-07-24): privately selected fixed sample, not public
> preregistration or an authenticated held-out study.** The selected events have
> been evaluated. This record cannot
> acquire public-preregistration status retroactively, and its legacy IID/
> diagonal-normalizer interval is not the frozen analysis. New confirmatory work
> uses the fail-closed public-registration workflow in `docs/confirmatory_v2.md`.

> The imperative and “FROZEN” language below is retained as a record of the
> authors' contemporaneous intention. It does not establish public chronology,
> held-out status, or the validity of the originally specified IID/normalizer
> analysis. The corrected descriptive reconstruction is explicitly labeled a
> private fixed-sample exploratory audit.

## Realized-design reconciliation

The text after this section is the archived plan and is retained to show what
was intended before the run. It is not the analysis now reported. The public
audit reconstructs the realized design as follows:

| Archived intention | Realized and released treatment |
|---|---|
| Publicly timestamped preregistration / held-out confirmation | No public pre-run timestamp was established. The result is historical exploratory evidence only. |
| Proportional allocation across ten time strata | The code used equal quotas within each band. The corrected estimator weights stratum means by their eligible finite-population counts. |
| Rounded band weights | Exact counts `(9342, 13898, 19981, 27671) / 70892` determine the four band weights. |
| End-to-end certified event log-ratios | The saved halfwidth is a kernel-derived arithmetic proxy. State construction, determinant normalization, and logarithms were not end-to-end certified, and the historical GPU/container/extension identity was not recorded. |
| Replacement after a refusal | The archived checkpoints contain 2,000 selected events and zero refusals. The audit refuses to form the stratified estimate if any selected event is refused or missing; it does not substitute another event. |
| Block bootstrap and full normalizer covariance | Neither artifact exists. The audit reports a design-based event sensitivity from the finite-population formula and the published per-band across-group standard deviations as a separate diagonal sensitivity scale; it reports no confidence interval or significance statistic. |
| Reconstruction-marginal physical conclusion | No reconstruction layer was run. The result is conditional on the two frozen point models and is not an absolute-fit or quantum-advantage verdict. |

The exact selection, checkpoint, normalizer, histogram, source-data, and
aggregation-script hashes are recorded in the released legacy artifact. The
unknown historical event-execution identity remains an explicit unresolved
provenance item.

## Archived contemporaneous plan (not the released analysis)

**This document must be committed and publicly timestamped (repo tag + Zenodo
or OSF) BEFORE any confirmatory likelihood is evaluated.** A commit pushed
after the run is not a preregistration (see the timing and exposure rules in
`docs/confirmatory_v2.md`). Everything below is frozen at timestamp; deviations
are reported as deviations, not folded in.

**FROZEN 2026-07-15** (after Item 1: DD revalidation + validated-kernel cost
curve). `N_C = 800/500/400/300`, `SEED = 20260715`, `n_strata = 10`, band-mass
weights per Section 2. Committed before any confirmatory likelihood is evaluated
and not changed thereafter. (For a publication-grade preregistration, promote
this record to a public timestamp — OSF/Zenodo — since the origin repo is
private; a private commit fixes the plan in advance but is not public.)

---

## 1. Object and hypotheses

Two frozen point models from Martínez-Cifuentes, Fonseca-Romero & Quesada,
*Quantum* 7, 1076 (2023), built verbatim by
`examples/jiuzhang/q7_construction.build_state`:

- **SQUE** — the targeted 25-TMSS squeezed ground truth (alternating
  `{−r_k,+r_k}` single-mode squeezers through balanced beamsplitters, propagated
  through the reconstructed lossy transfer matrix `T`).
- **SQUA** — the classical squashed alternative (per-source covariance
  `diag(1, 1+4 sinh² r)` in the same basis; photon-number preserving).

Both are evaluated on the **exact real quadrature-basis** torontonian input
`O_x = I − Σ_x⁻¹`, `Σ_x = (σ+I)/2` (`sampling.gbs.threshold_O_xxpp`), which is
torontonian-equivalent to the complex `I − Q⁻¹` (supplement §"exact real
input"). The real-cast `Re(I − Q⁻¹)` is the wrong matrix and is refused by the
regression guard.

Construction parity (covariance to ≤2e-15 vs their published arrays; exact
`C̄`, `σ(C)` inside Table-2 MC error) is a **precondition gate** that must pass
before any evaluation; it is already established (`q7_parity_*` artifacts) and
re-run at the release commit.

## 2. Primary estimand

For the beyond-ceiling band set `B = {27, 28, 29, 30}`,

    Δ_B  =  Σ_{C∈B} w_C · ΔH(C),

    ΔH(C) = (1/L_C) Σ_k ln[ Pr_SQUA(s_k) / Pr_SQUE(s_k) ]
                    + ln[ Pr_SQUE(C) / Pr_SQUA(C) ].

`ΔH(C) > 0` means the classical squashed model assigns the click-`C` events
higher conditional likelihood. Per-event terms are certified DD torontonian
log-ratios (Item 1 must be green); grouped-click normalizers `Pr(C)` are the
one non-certified term (Section 5.3).

**Band weights are the frozen empirical click-count mass** from the first
3,000,000-record histogram (`click_count_dist.npy`), NOT computational sample
counts, achieved stopping times, inverse standard errors, or observed effects:

    w_27 = 0.13178
    w_28 = 0.19604
    w_29 = 0.28185
    w_30 = 0.39033        (sum = 1)

The estimand is: among events in click window `B`, sampled with the empirical
click mixture, which frozen point model has lower conditional empirical cross
entropy. It does **not** establish absolute fit, classical simulability of the
full experiment, or absence of quantum features.

## 3. Design — fixed sample, no optional stopping

**No repeated-look stopping is permitted for the primary analysis.** Per-band
sample sizes `N_C` are frozen here; evaluation proceeds to exactly `N_C` usable
events (refusals replaced by the next eligible event under the same key, with
refusals logged — Section 6).

Per-band `N_C`, **FROZEN 2026-07-15** from the Item-1 re-measured
(validated-kernel) cost curve and the exploratory per-event spread
(`σ ≈ 0.16` conditioned):

| C  | N_C (FROZEN) | s/DD-eval (4090, validated) | s/event (both hyp, 2×) | GPU-hr |
|----|------|------|------|------|
| 27 | 800  | 12.5 | 25   | 5.6  |
| 28 | 500  | 26   | 52   | 7.2  |
| 29 | 400  | 55   | 110  | 12.2 |
| 30 | 300  | 115  | 229  | 19.1 |

Total 2000 events, ≈ 44 GPU-hours ≈ \$13 on one RTX 4090 (before reruns/refusals).

At this allocation the band-mass-weighted stratified standard error on `Δ_B` is
≈ 0.0049 (from the exploratory spreads), so against the exploratory
`Δ_B ≈ +0.032` the nominal power is ≈ 6.5σ — deliberate margin so the conclusion
survives the additional normalizer-covariance, block-dependence, and
reconstruction uncertainty of Section 5. `N_C` is frozen and is not revised
after any likelihood is evaluated.

## 4. Event selection — disjoint, time-stratified, seed-published

1. **Pool.** The full 51,463,365-record decoded file (`data.bin`, proven
   decoder), all bands `C ∈ B`, normal (non-abnormal) events only.
2. **Exclusions (disjoint holdout).** Exclude every stage-1 exploratory event
   (recorded record indices in `results/jiuzhang/campaign_C*.jsonl`) and every
   archived published pattern (their Zenodo `patterns_exp/samples_0_clicks_C`)
   from the confirmatory pool. The confirmatory sample overlaps neither.
3. **Selection key.** For each eligible record, a public deterministic hash
   `key = SHA256(record_index ‖ SEED)`; events are ordered by `key` within
   time strata.
4. **Time strata.** The acquisition is partitioned into `n_strata = 10` (FROZEN)
   equal record-index blocks; `N_C` is drawn proportionally across strata so
   drift/nonstationarity can be tested (secondary outcome).
5. **Seed.** `SEED = 20260715` (FROZEN in this document, committed before any
   likelihood is evaluated; the deterministic selection manifest it produces is
   committed alongside, so the exact confirmatory sample is fixed in advance).
6. **Retained per event:** record index, timestamp bits, abnormal flag,
   detector pattern, input hash, and the deterministic selection key.

## 5. Uncertainty — four separate layers, never merged

1. **Numerical enclosure (deterministic).** Worst-case interval from the
   DD-certified log-ratios; applied as a **worst-case displacement** of `Δ_B`,
   not a Gaussian variance. (Requires Item 1 green.)
2. **Event sampling.** Time-block / cluster bootstrap over strata (not an
   IID-only SEM), to account for block dependence.
3. **Grouped-click normalizers.** Independent replicated simulation of the
   log-normalizer differences with the **full covariance** across hypotheses
   and `C` (`thewalrus.grouped_click_probabilities`, the authors' own
   estimator; independent seed from theirs). The published across-group-std
   convention (no `1/√G`) is ~10× conservative as an SE; carried unchanged.
4. **Physical reconstruction.** Bootstrap/posterior draws of `T`, squeezing,
   losses; for each draw both hypotheses recomputed coherently. **Report the
   point-model result and the reconstruction-marginal result separately.** If
   the reconstruction layer is not run, the conclusion is stated strictly
   conditional on the frozen point models, in the title and abstract.

## 6. Reporting (frozen analysis)

1. Report a **confidence interval for `Δ_B`**, not a "proven statistical sign."
2. Arithmetic width enters as a worst-case displacement (Section 5.1), and the
   stopping/inclusion decisions do **not** depend on it beyond refusal.
3. **All refusals reported**, with a test of whether refusal depends on the
   event score (score-dependent refusal would bias `Δ_B`).
4. Secondary outcomes: each band `ΔH(C)` and a heterogeneity statistic across
   bands; the drift test across time strata; absolute held-out checks (click
   marginals, selected correlation functions, PIT-style calibration under both
   models).
5. Benchmark against an independent MPFR/Arb torontonian and a current
   Piquasso/Julia baseline on identical matrices, hardware-normalized where
   possible.
6. The paper build consumes a hash-pinned artifact manifest and **fails closed**
   when an input is absent; no hard-coded plotting fallback.

## 7. Deviations

Any departure from the frozen text above (N, seed, weights, estimand,
selection, analysis) is reported in a "Deviations from preregistration"
section with its reason and its effect, and does not silently replace the
frozen plan.
