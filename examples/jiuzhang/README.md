# Jiuzhang Reproduction And Confirmatory V2 Workflow

This directory contains two related public surfaces: the corrected
certified-torontonian reproduction on the Jiuzhang 1.0 dataset, and the
fail-closed confirmatory-v2 registration, selection, evaluation, inference,
and release tooling.

The v2 workflow contains no completed registration and no new scientific
outcome. Its templates intentionally remain unresolved until external
calibration, public timestamp/beacon evidence, and acquisition inputs exist.
See the [full protocol](../../docs/confirmatory_v2.md).

## Data source

- Landing page: <https://quantum.ustc.edu.cn/web/en/node/915>
- Archive: `web_0.zip` (~744 MB), SHA-256
  `50ee65bef10934b4c2df9fb1f2d88d57a25f894c7100381097255578565809bc`
- Contents: `data.bin` (~5×10⁷ threshold click samples; 128-bit records —
  16 timestamp bits, 4 ignored bit positions, 100 detector bits ordered
  detector-100→1, 8 flag bits with the last marking an abnormal sample),
  `matrix re.xlsx` / `matrix im.xlsx` (the 50×100 transfer matrix, input phases
  and system efficiencies folded in), `squeezing parameters.txt` (25 squeezers).

The archive is not redistributed here; download it from the source above.

For the public v0.2.1 GPU `validate` session, the repository ships five small,
hash-bound runtime inputs plus one decoder-audit reference. Stage them into the
ignored `data/` layout before launching a session:

```bash
python scripts/prepare_validation_data.py
```

This is sufficient for the 24 device gates, binding smoke, adversarial
enclosure, and Gate C probe. The full raw archive is still required only when
rerunning the complete decoder or regenerating the event supply; the Zenodo
bundle is still required for the broader parity and reconstruction scripts.
The staged files and their provenance are listed in
[`validation_data/README.md`](validation_data/README.md); third-party licensing
scope is recorded in [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).

## Publication Arb enclosure campaign (prospective)

`arb_enclosure_campaign.py` defines a new, independent validation of the
certified double-double torontonian interval. It is distinct from the mpmath
adversarial-enclosure gate included in the public v0.2.1 validation, and no GPU
result from the Arb campaign is claimed here yet.

The registered small-case matrix is exactly 312 cases: `k = 2,...,14`, six
families, and four deterministic instances per `(k, family)` cell. The families
are random SPD, near-refusal at scales 0.9, 0.99, and 0.999, cancellation, and
the physical Jiuzhang squeezed/squashed construction. The physical family uses
the small hash-bound validation inputs described above. Each exact binary64
matrix is retained in a compressed corpus and SHA-256-bound to its result.

For these small dense cases, python-flint/Arb independently evaluates every
subset determinant after converting each binary64 entry exactly. The campaign
then compares exact rational endpoints: `proved` means the complete Arb interval
is contained in the candidate's returned `[value - bound, value + bound]`;
partial overlap is `inconclusive`, disjoint intervals are `violation`, and an
invalid or non-finite certificate is `refused`.

The publication profile adds eight depth tests, one at each `k = 25,...,32`.
Each is the structured `2k x 2k` quarter-identity matrix, assembled from
independent `0.25 I_2` mode blocks. The production GPU candidate still executes
the recursive torontonian call, while the Arb reference uses the exact block
factorization (the torontonian is `(1/3)^k`). These cases test recursion depth and
the returned enclosure at large `k`; they are explicitly not validation over
general dense, correlated, or physical large-`k` matrices.

The archive-bound session is configured to accept the campaign only if all 320
cases are `proved`, with no inconclusive, violation, or refusal. This is a future
acceptance criterion, not a statement that the criterion has already been met.
The JSON artifact records the Arb/FLINT versions, oracle and campaign source
hashes, exact endpoints, timings, source-archive/build-manifest provenance, and
the hash and inventory of the create-only matrix corpus.

## State construction

`q7_construction.py` implements the paired-source construction of
Martinez-Cifuentes et al., Quantum 7, 1076. Each source pair starts from the
published `{-r_k, +r_k}` ordering, passes through a 50:50 beamsplitter, and
then through the measured lossy transfer matrix.

The evaluator uses the exact real quadrature-basis torontonian matrix
`Ox = I - Sigma_x^-1`, where `Sigma_x = (cov + I) / 2`; see
`sampling.gbs.threshold_O_xxpp`. The helper explicitly symmetrizes the
binary64 result before it reaches the one-triangle recursive kernel.
Restricting a click pattern to its modes in both quadrature blocks preserves
the torontonian under the fixed basis change.

The two timestamped frontier JSONs committed in the v0.1 series predate this
correction. They are retained as historical public artifacts and must not be
used for new figures or claims. Regenerate both inputs with the current scripts.

## Historical fixed-sample audit

Version 0.2.1 also publishes the corrected audit of the deterministic sample
selected and evaluated during earlier exploratory development. It is a private
fixed sample in the statistical sense: it was not held out from prior analysis,
preregistered, or selected from a future public beacon, so it is not a
confirmatory result.

- `decode_events.py` contains an executable detector-order and dead-slot audit.
- `select_confirmatory.py` reconstructs the historical deterministic selection
  and can bind it byte-for-byte to the retained original manifest.
- `campaign_confirmatory.py` verifies every selected row and reports the exact
  finite-population stratified estimand. Event sampling error, normalizer
  diagonal sensitivity, and the arithmetic proxy remain separate quantities.

The preserved inputs and regenerated audit outputs live in
`results/jiuzhang/legacy_fixed_sample/`. The five historical checkpoint JSONL
files predate the common provenance schema and lack recoverable original GPU,
container, driver, and commit metadata. Their hashes preserve what is known;
the regenerated manifest/result record current provenance without inventing
the missing historical environment.

## Scripts

- `build_exclusion_ledger.py`, `audit_selection_population.py`,
  `confirmatory_design.py`, `registration_readiness_v2.py`, and
  `prepare_registration_v2.py` implement the fail-closed v2 registration path
  described in the protocol.
- `select_confirmatory_v2.py`, `campaign_confirmatory_v2.py`,
  `analyze_refusals.py`, `confirmatory_inference.py`, and
  `confirmatory_release.py` implement selection, immutable evaluation, refusal
  recovery, inference, and release validation.
- `coherence_family.py`, `joint_normalizer_replicates.py`,
  `calibration_normalizer_replicates.py`, `reconstruction_replicates.py`, and
  `absolute_predictive_checks.py` implement the registered model family and
  nuisance/predictive propagation required before a claim can be released.
- `jiuzhang_frontier.py` and `dd_validate.py` regenerate corrected fp64 and
  double-double frontier artifacts under `results/jiuzhang/`.
- `arb_enclosure_campaign.py` defines the prospective 312-case dense small-matrix
  plus eight-case structured large-`k` Arb enclosure campaign described above.
- `make_precision_wall.py` requires explicit corrected `--fp64-artifact` and
  `--dd-artifact` inputs and refuses the superseded v0.1 artifacts.
- `q7_construction.py` provides the fixed point models used by the v2 workflow.
- `decode_events.py`, `select_confirmatory.py`, and
  `campaign_confirmatory.py` reproduce the historical fixed-sample audit; their
  names are retained for hash/history continuity and do not make it
  confirmatory.
- `click_count_dist.npy` — click-count histogram (101 int64 bins) over the
  2,995,852 decoded normal samples; used for the grey event histogram in the
  figure so it can be rebuilt without the 744 MB dataset.

Run from the repository root with the CUDA extension built
(`GBSKERNELS_EXT_DIR`) and the data files in place. For the final command,
replace the uppercase timestamp markers with the filenames printed by the first
two commands:

```bash
uv run python examples/jiuzhang/dd_validate.py --events 30 --kmax 26
uv run python examples/jiuzhang/jiuzhang_frontier.py --kmax 32 --events 120
uv run python examples/jiuzhang/make_precision_wall.py \
  --fp64-artifact results/jiuzhang/jiuzhang1_frontier_CORRECTED_TIMESTAMP.json \
  --dd-artifact results/jiuzhang/dd_frontier_CORRECTED_TIMESTAMP.json
```
