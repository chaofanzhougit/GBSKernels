# Jiuzhang 1.0 certified-torontonian reproduction

Reproduces the precision-wall figure and the fp64/double-double certified
frontier on the public Jiuzhang 1.0 threshold Gaussian-boson-sampling dataset.

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

## State construction

`q7_construction.py` implements the paired-source construction of
Martinez-Cifuentes et al., Quantum 7, 1076. Each source pair starts from the
published `{-r_k, +r_k}` ordering, passes through a 50:50 beamsplitter, and
then through the measured lossy transfer matrix.

The evaluator uses the exact real quadrature-basis torontonian matrix
`Ox = I - Sigma_x^-1`, where `Sigma_x = (cov + I) / 2`; see
`sampling.gbs.threshold_O_xxpp`. Restricting a click pattern to its modes in
both quadrature blocks preserves the torontonian under the fixed basis change.

The two timestamped frontier JSONs committed in the v0.1 series predate this
correction. They are retained as historical public artifacts and must not be
used for new figures or claims. Regenerate both inputs with the current scripts.

## Scripts

- `build_exclusion_ledger.py`, `audit_selection_population.py`,
  `confirmatory_design.py`, `registration_readiness_v2.py`, and
  `prepare_registration_v2.py` implement the fail-closed v2 registration path
  described in `docs/confirmatory_v2.md`.
- `select_confirmatory_v2.py`, `campaign_confirmatory_v2.py`,
  `analyze_refusals.py`, `confirmatory_inference.py`, and
  `confirmatory_release.py` implement selection, immutable evaluation, refusal
  recovery, inference, and release validation.
- `jiuzhang_frontier.py` and `dd_validate.py` regenerate corrected fp64 and
  double-double frontier artifacts under `results/jiuzhang/`.
- `make_precision_wall.py` requires explicit corrected `--fp64-artifact` and
  `--dd-artifact` inputs and refuses the superseded v0.1 artifacts.
- `q7_construction.py` provides the fixed point models used by the v2 workflow.
- `click_count_dist.npy` — click-count histogram (101 int64 bins) over the
  2,995,852 decoded normal samples; used for the grey event histogram in the
  figure so it can be rebuilt without the 744 MB dataset.

Run from a checkout with the CUDA extension built (`GBSKERNELS_EXT_DIR`) and
the data files in place; for example,
`python dd_validate.py --events 30 --kmax 26`.
