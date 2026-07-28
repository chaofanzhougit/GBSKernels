# Changelog

## [0.2.2] - 2026-07-28

### Correctness

- Charge the recursive-torontonian double-double leaf accumulation against the
  operand magnitudes instead of the post-addition result. The signed `dd_add`
  is the sloppy double-word addition, whose represented-value error scales with
  the operand magnitudes; the previous `u_DD * md_hi(total)` charge could
  under-bound it whenever a single accumulation step cancels the running sum by
  more than ~16x (`quick_two_sum`'s `|s| >= |e|` precondition can fail there).
  The returned value is unchanged (centers stay bit-identical); only the
  returned enclosure radius grows, and only at deeply-cancelling steps. The
  Cholesky dot chain already charged its operand sum, so this makes the leaf
  sum consistent with it.

## [0.2.1] - 2026-07-24

### Correctness

- Add explicit absolute underflow terms to the certified FP64 and
  double-double permanent, hafnian, loop-hafnian, and recursive-torontonian
  recurrences.
  Per-operation propagation now encloses mixed-scale products where an
  intermediate underflows and a later factor amplifies the lost term.
- Replace assumed forward constants for double-double division and square root
  with outward FMA residual bounds on the values actually returned. Complex
  division refuses denominator, numerator-product, and dot-product ranges that
  cannot be enclosed safely, and preserves tiny numerator uncertainty through
  large quotient amplification.
- Preserve `+inf` refusal through host subtree reductions. Certified Python
  APIs now reject non-finite inputs and refuse non-finite values, negative or
  non-finite bounds, and invalid tolerances; an explicit `rtol` may escalate a
  refused certificate to an independently computed finite reference value.
- Make `tor_single` fail before GPU dispatch unless its real binary64 input is
  finite and exactly symmetric. The recursive Cholesky walk consumes one
  triangle, so silently accepting a nonsymmetric matrix made the evaluated
  problem depend on storage convention.
- Explicitly symmetrize the xxpp Husimi covariance and returned
  `threshold_O_xxpp` matrix after binary64 linear algebra, satisfying the new
  single-large input contract without retaining inversion roundoff skew.
- Replace the historical Jiuzhang fixed-sample summary with its actual
  finite-population stratified estimand. Remove pseudo-confirmatory intervals
  and normalizer significance calculations that could not be justified from
  the retained inputs; report the design-based event sensitivity, normalizer
  diagonal sensitivity, and arithmetic proxy separately.

### Reproducibility

- Extend artifact provenance with the full Git commit and a tracked-dirty
  indicator. Local Git checkouts record whether tracked bytes differ; rsynced
  sessions using `GBS_COMMIT` have no Git metadata and record dirty state as
  unknown rather than claiming a clean tree.
- Package the deterministic historical selector, aggregator, decoder audit,
  five retained checkpoint JSONL files, and a hash-bound regenerated
  selection/result pair under `results/jiuzhang/legacy_fixed_sample/`.
- The retained checkpoint rows predate the common provenance schema and do not
  contain recoverable original GPU, container, driver, or commit metadata. The
  release preserves their exact hashes and records current provenance for the
  audit artifacts; it does not invent missing historical environment data.

### Scope

- The historical fixed sample was selected and evaluated during exploratory
  development. It is not held out, preregistered, or confirmatory, and this
  correction does not turn it into a confirmatory scientific result.
- The v0.2.1 release carries a validation manifest covering all 24 device gates,
  the Python binding smoke, adversarial and physical enclosure coverage, and the
  Jiuzhang Gate C probe. The session is release-commit-attributed through
  `GBS_COMMIT`; it does not hash-bind the uploaded source tree or loaded CUDA
  extension. Retained v0.2.0 RTX 4090 evidence remains historical rather than
  validation of v0.2.1.

## [0.2.0] - 2026-07-22

### Important Correction

- Replace the legacy Jiuzhang state reconstruction with the published
  paired-source Q7 construction and the exact real xxpp threshold-torontonian
  matrix. The two v0.1 frontier JSON files remain historical artifacts and are
  rejected by the current figure tooling.

### Added

- A fail-closed Jiuzhang confirmatory-v2 workflow covering exposure-ledger
  audit, outcome-blind design, immutable public registration and beacon-derived
  selection, content-addressed evaluation, refusal recovery,
  reconstruction/calibration uncertainty, joint normalizer replicates,
  absolute predictive checks, simultaneous coherence-grid inference, and
  hash-verified release bundles.
- Canonical registration, design, and exposure templates plus focused contract
  and regression tests for the workflow and exact xxpp construction.

### Fixed

- Harden certified recursive-torontonian bounds with directed downward
  subtraction, sound double-double magnitude bounds, and explicit
  double-double-to-FP64 collapse residual accounting, plus a dedicated
  enclosure gate.

### Scope

- This release provides code, templates, and validation contracts. It contains
  no completed v2 registration or new scientific outcome; external calibration,
  public timestamp/beacon evidence, and acquisition inputs remain required.

[0.2.1]: https://github.com/chaofanzhougit/GBSKernels/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/chaofanzhougit/GBSKernels/compare/v0.1.1...v0.2.0
