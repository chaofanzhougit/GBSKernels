# Confirmatory v2 workflow

The 2026-07-15 Jiuzhang 1.0 sample is a privately frozen held-out comparison.
It is not retroactively promoted to a public preregistration. The legacy runner
and artifacts are outside this public workflow; new confirmatory claims use the
v2 process below.

## Scientific question

The physical model is a continuous source family, not only two endpoints. For
source pair `k`, the per-mode population is `n_k` and the anomalous coherence is
`m_k = <a_1 a_2>`. The Gaussian physical region and classical positive-P region
are, respectively,

```text
|m_k| <= sqrt(n_k (n_k + 1)),    |m_k| <= n_k.
```

`examples/jiuzhang/coherence_family.py` constructs this family through the
same lossy transfer-matrix channel as the published Q7 point models. A
registered campaign must state whether its primary result is predictive
comparison of fixed coherence points, an interval/posterior for coherence and
its distance from the classical boundary, or a reconstruction-marginal
predictive comparison.

Click-count fit `log p(C)` and within-band fit `log p(s | C)` are separate
registered estimands. A conditional score alone is never described as absolute
fit or evidence against the whole classical region.

## Registration contract

`examples/jiuzhang/confirmatory_contract.py` accepts only an immutable HTTPS
registration whose timestamp strictly precedes a named, frozen randomness-
beacon source and round. The resolved record contains an independently
retrievable beacon proof URL/content hash, the beacon value, and deterministic
SHA-256 seed derivation. The public digest must equal the canonical plan hash;
an unrelated archive digest is rejected. The public plan freezes:

- exact raw-data, complete exposure-ledger, population-audit, design-report,
  calibration, state, code, and numerical-contract hashes;
- common acquisition strata, targets, ranked reserve counts, and estimands;
- the model grid, simultaneous decision rule, and minimum relevant coherence;
- bootstrap seed, replicate count, alpha, and band weights;
- joint normalizer-replicate and reconstruction-draw requirements;
- absolute predictive checks for every statistically plausible best model; and
- prior aggregate use and the finite-acquisition scope of the intended claim.

The repository intentionally contains no fake completed registration. Obtain a
real public timestamp and a future beacon result, then create the resolved JSON.
The validator refuses a private URL, mutable record, incorrect plan hash, stale
beacon timing, or seed chosen outside the registered derivation.
Sample sizes and absolute-fit thresholds are deliberately placeholders. They
must come from a frozen design specification and independently generated
calibration-forward simulations, never from the evaluated legacy holdout.

```bash
# Mechanically pin the known repository evidence. Then resolve the catalog's
# provenance-risk and author-attestation fields through an actual human review.
uv run python examples/jiuzhang/build_exclusion_ledger.py pin-catalog \
  --template docs/exploratory_exposure_manifest.template.json \
  --out exploratory-exposure-catalog.json
uv run python examples/jiuzhang/build_exclusion_ledger.py build \
  --catalog exploratory-exposure-catalog.json --require-complete \
  --out exploratory-exclusion-ledger.json

# Audit the complete finite population against that canonical ledger.
uv run python examples/jiuzhang/audit_selection_population.py \
  --exclude-records exploratory-exclusion-ledger.json --n-strata 20 \
  --out selection-population-audit.json

# After resolving every external placeholder in a working design-spec JSON,
# materialize the create-only canonical artifact used by all later commands.
uv run python examples/jiuzhang/confirmatory_design.py --canonicalize-spec \
  --spec resolved-design-spec.json --out confirmatory-design-spec.json

# Generate design-simulation.json only with the frozen simulator and independent
# calibration posterior named in that canonical spec. Evaluate every frozen candidate.
uv run python examples/jiuzhang/confirmatory_design.py \
  --spec confirmatory-design-spec.json \
  --population-audit selection-population-audit.json \
  --simulation design-simulation.json --out confirmatory-design-report.json

# Inspect all blockers before attempting to prepare a public plan.
uv run python examples/jiuzhang/registration_readiness_v2.py \
  --template docs/confirmatory_v2_plan.template.json \
  --exposure-catalog exploratory-exposure-catalog.json \
  --exclusion-ledger exploratory-exclusion-ledger.json \
  --population-audit selection-population-audit.json \
  --design-spec confirmatory-design-spec.json \
  --design-report confirmatory-design-report.json \
  --design-simulation design-simulation.json \
  --simulator-source frozen-design-simulator.py \
  --simulation-bank design-simulation-bank.npz \
  --refusal-recovery-source independent_recovery.py \
  --calibration calibration-posterior.npz \
  --out registration-readiness.json

# This refuses incomplete evidence, failed design assurance, unresolved
# placeholders, dirty analysis source, or a non-future beacon.
uv run python examples/jiuzhang/prepare_registration_v2.py prepare-plan \
  --template docs/confirmatory_v2_plan.template.json \
  --exposure-catalog exploratory-exposure-catalog.json \
  --exclusion-ledger exploratory-exclusion-ledger.json \
  --population-audit selection-population-audit.json \
  --design-spec confirmatory-design-spec.json \
  --design-report confirmatory-design-report.json \
  --design-simulation design-simulation.json \
  --simulator-source frozen-design-simulator.py \
  --simulation-bank design-simulation-bank.npz \
  --refusal-recovery-source independent_recovery.py \
  --calibration calibration-posterior.npz \
  --out registration-plan.json
# Publicly archive that exact file, then wait for the registered beacon round.
uv run python examples/jiuzhang/prepare_registration_v2.py resolve \
  --plan registration-plan.json --public-url https://... \
  --public-sha256 <sha256-of-the-canonical-plan> \
  --published-at <public-record-utc> \
  --timestamp-proof-url https://... \
  --timestamp-proof-sha256 <archive-record-sha256> \
  --beacon-source <source> \
  --beacon-round <frozen-round> --beacon-value <value> \
  --beacon-proof-url https://... \
  --beacon-proof-sha256 <provider-record-sha256> \
  --out registration-resolved.json
```

## Selection

`select_confirmatory_v2.py` makes two streaming passes over the full raw file.
It uses common strata `floor(record_index * H / N)`, removes explicit audited
record indices, allocates samples by exact largest remainder from eligible
stratum populations, and ranks primary plus reserve events with the beacon
seed. Its canonical JSON manifest retains raw record and timestamp information,
detector patterns and hashes, selection keys, strata and reserve ranks, eligible
populations, quotas, and exact inclusion probabilities.

```bash
uv run python examples/jiuzhang/select_confirmatory_v2.py \
  --registration registration-resolved.json \
  --exclude-records exploratory-exclusion-ledger.json \
  --out confirmatory-manifest-v2.json
```

## Immutable evaluation

`campaign_confirmatory_v2.py` creates a content-addressed run ID from the public
plan, manifest, commit, container digest, loaded CUDA-extension binary hash,
numerical scope, and state matrices.
Each event is a create-only JSON object named by its manifest-bound event ID.
Existing unequal content is a hard failure. The reducer rejects wrong bindings,
uses reserves only in registered order, requires every registered usable cell
count, reports all refusals, and emits no confidence interval itself. If a
primary evaluation is refused, the required independent refusal artifact
restores that primary event with its recovered score and removes the
operational reserve replacement; a reserve is never silently treated as the
same sampled unit. The refusal diagnostic is itself frozen in the plan: it uses
the registered repetitions, seed, and alpha for a maximum absolute mean-score
difference permutation test within each fixed band-by-stratum cell. Reserve
replacements are excluded from that diagnostic once their refused primaries
have been recovered. The plan explicitly freezes `inferential_gate: false`:
because every selected primary is restored, the permutation p-value is
descriptive and is never interpreted as evidence that missingness is ignorable.
If no fixed cell contains both a recovered primary and an unreplaced accepted
primary, the artifact records that the diagnostic is unavailable rather than
turning that lack of contrast into a failure or a p-value of one.
Distributed workers must therefore use the same compiled extension bytes (for
example, one image-baked binary); independently rebuilt binaries intentionally
produce different run IDs even when their source commit is identical.

The current evaluator claims only a DD enclosure of the registered binary64
torontonian matrices. It does not call this an end-to-end probability
certificate. A registration demanding certified probabilities must provide and
consume outward-rounded state, inverse, determinant, and logarithm intervals.

## Inference

`confirmatory_inference.py` accepts only a complete verified run. The registered
acquisition strata remain fixed and receive their full eligible-population
weights. Within each band-by-stratum cell, centered event vectors are resampled
under the registered SRSWOR design with a finite-population correction; strata
are never resampled as if they were sampled clusters. Paired normalizer
probability replicates are pooled before taking the log ratio, and uncertainty
resamples all paired replicate rows and recomputes that pooled estimator.
Deterministic arithmetic widths enter by Minkowski expansion.

The registered primary is a simultaneous paired max-t confidence set for the
best anomalous-coherence grid point (classical boundary `eta <= 0`), not a
percentile interval for a nonregular bootstrap argmax. A nonclassical claim is
supported only when that entire confidence set lies above the classical
boundary and every model in the set passes the absolute predictive gate. A
failed condition remains a complete, releasable negative analysis. The endpoint
decomposition is secondary. Point-model and reconstruction-marginal results are
reported separately; the latter propagates every registered grid point through
squeezing, transfer, loss, detector response, and block-drift draws.

`joint_normalizer_replicates.py` writes the joint-normalizer NPZ. It contains
JSON `meta` with schema
`gbskernels.joint-normalizer-replicates.v1` and `bands`, plus paired arrays
`p_reference` and `p_alternative` of shape `(replicate, band)`. The
reconstruction NPZ contains schema `gbskernels.reconstruction-replicates.v3`,
run/registration/input hashes, absolute band summaries
`joint_log_score_band_draws`, `normalizer_log_ratio_band_draws`, and
`model_log_score_band_draws`, plus draw-by-event model score and arithmetic
enclosure tensors. Stable event identities bind those tensors to the verified
sample. Inference selects one posterior draw per replicate and resamples events
within each fixed cell using that same draw, preserving calibration-by-event
interaction and pairing the corresponding click normalizer. The normalizer
artifact also carries physical calibration-draw fingerprints and payload
fingerprints, so draw-axis permutation, changed effort, or a changed seed is
rejected. Its calibration NPZ
must explicitly name squeezing, transfer, loss, and block-drift nuisance
families, bind to the registered posterior hash, state the detector-response
model, and declare dark clicks explicitly zero (a nonzero dark-click
convolution is refused until implemented).

Generate the required stratum-aware normalizers before propagation:

```bash
uv run python examples/jiuzhang/calibration_normalizer_replicates.py \
  --registration registration-resolved.json \
  --calibration calibration-posterior.npz \
  --out calibration-normalizer-draws.npz
```

The draw-level normalizer input must carry the same calibration-posterior hash
and be generated by `calibration_normalizer_replicates.py` with paired
`(draw, common_stratum, model, band)` probabilities. A single normalizer per
draw is insufficient when the registered block-drift nuisance is nonzero;
reconstruction propagation stops on that shape or pairing mismatch.

```bash
uv run python examples/jiuzhang/reconstruction_replicates.py \
  --registration registration-resolved.json --manifest confirmatory-manifest-v2.json \
  --verified-run verified-run.json --calibration calibration-posterior.npz \
  --normalizer-draws calibration-normalizer-draws.npz \
  --nominal-normalizers joint-normalizers.npz \
  --out reconstruction-replicates.npz
```

If the run contains refusals, independently recover every refused score before
reconstruction. Every recovered selected primary must include an explicit
nonnegative score half-width and finite intervals for the complete registered
model grid. `analyze_refusals.py` takes no free analysis settings: it consumes
the frozen refusal contract and binds its v2 artifact to the exact registration,
verified-run bytes, recovery-input bytes, analysis source, commit, and container.
Artifact `pass` means that recovery is complete and all bindings and enclosures
validate; it does not mean that the diagnostic p-value exceeds alpha.
The recovery input is a self-hashed
`gbskernels.independent-refusal-recovery.v1` object, not a bare score list. The
plan freezes its independent interval-reevaluation method, minimum precision,
source hash, and container digest; the recovery object also binds the exact
verified-run hash. Its source bytes and raw recovery object are release roles.
Readiness verifies the source bytes and validates the pinned container-digest
syntax; availability of that external image cannot be established offline.

```bash
uv run python examples/jiuzhang/analyze_refusals.py \
  --registration registration-resolved.json --verified-run verified-run.json \
  --recovered independently-recovered-refusals.json \
  --recovery-source independent_recovery.py \
  --out refusal-analysis.json
```

`absolute_predictive_checks.py` independently scans the complete eligible
acquisition (the canonical manifest's explicit exclusions are applied) and
compares every registered coherence point against click-count mass, detector
marginals, and registered detector-pair correlations. Its
`gbskernels.absolute-predictive-checks.v1` artifact is required by the inference
command and is bound to the verified run and raw-data hash. The registered
`model_pass_policy` then gates the inferred/plausible best coherence models,
not merely an unrelated model that happens to fit. These are frozen
nominal-state diagnostics; calibration uncertainty is not silently folded into
them and is handled by the reconstruction-marginal layer.

```bash
uv run python examples/jiuzhang/absolute_predictive_checks.py \
  --registration registration-resolved.json --manifest confirmatory-manifest-v2.json \
  --verified-run verified-run.json --normalizer-replicates joint-normalizers.npz \
  --data data/jiuzhang1/data.bin --out absolute-predictive-checks.json
uv run python examples/jiuzhang/confirmatory_inference.py \
  --verified-run verified-run.json --normalizer-replicates joint-normalizers.npz \
  --reconstruction-replicates reconstruction-replicates.npz \
  --predictive-checks absolute-predictive-checks.json --out confirmatory-analysis.json
```

After inference, create the paper-build input with
`examples/jiuzhang/confirmatory_release.py assemble ...`. It hashes and binds
the exposure catalog, exclusion ledger, population audit, design spec,
simulation bank, reproduced design report, registration, canonical selection
manifest, complete run, normalizers, calibration posterior, stratum-aware
calibration normalizers, reconstruction draws, absolute predictive checks,
refusal analysis (when needed), and final analysis. It recomputes the registered
decision but does not reject a scientifically negative result.
`confirmatory_release.py verify` must pass before any figure/table build; it
fails on missing or changed bytes.

```bash
uv run python examples/jiuzhang/confirmatory_release.py assemble \
  --registration registration-resolved.json \
  --exposure-catalog exploratory-exposure-catalog.json \
  --exclusion-ledger exploratory-exclusion-ledger.json \
  --population-audit selection-population-audit.json \
  --design-spec confirmatory-design-spec.json \
  --design-report confirmatory-design-report.json \
  --design-simulation design-simulation.json \
  --simulator-source frozen-design-simulator.py \
  --simulation-bank design-simulation-bank.npz \
  --manifest confirmatory-manifest-v2.json --verified-run verified-run.json \
  --normalizers joint-normalizers.npz --calibration calibration-posterior.npz \
  --calibration-normalizers calibration-normalizer-draws.npz \
  --reconstruction reconstruction-replicates.npz \
  --predictive-checks absolute-predictive-checks.json \
  --refusal-analysis refusal-analysis.json \
  --refusal-recovery independently-recovered-refusals.json \
  --refusal-recovery-source independent_recovery.py \
  --analysis confirmatory-analysis.json --out confirmatory-release.json
uv run python examples/jiuzhang/confirmatory_release.py verify \
  --release confirmatory-release.json
```

When any primary evaluation is refused, pass the same validated
`--refusal-analysis` artifact to reconstruction, inference, and release. Each
consumer recomputes the registered permutation result, verifies the recovered
score payload, and checks the exact run hash and provenance. Release additionally
hashes the raw recovery object and the preregistered recovery source bytes;
reconstruction records the refusal-artifact hash it consumed. No consumer gates
inference on failure to reject the diagnostic permutation null.

Independent acquisitions, public registration, beacon publication, calibration
draws, and the missing Jiuzhang 2.0 `nu=0.975` archive cannot be generated by
software. The v2 commands fail closed until those external inputs exist. A run
on the already released Jiuzhang 1.0 file remains a finite registered-
acquisition result, not a process-level claim about future devices.
