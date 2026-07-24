# Historical Jiuzhang fixed-sample audit

This directory preserves the inputs and regenerated outputs for the corrected
audit of the deterministic Jiuzhang 1.0 sample selected on 2026-07-15. The
sample was exposed during exploratory development. It was not held out,
preregistered, or selected by a future public beacon, so no file here is a
confirmatory result.

The package contains:

- a v2 selection NPZ reconstructed from the raw acquisition and verified
  byte-for-byte against the archived historical selection;
- a v2 aggregate JSON reporting the finite-population stratified point
  estimate, event sampling standard error, normalizer diagonal sensitivity,
  and arithmetic proxy as separate quantities; and
- five original per-event checkpoint logs covering all 2,000 selected rows.

The historical checkpoint files are preserved byte-for-byte:

| File | Rows | SHA-256 |
|---|---:|---|
| `confirmatory_C27_b1.jsonl` | 800 | `950da7a02a72ab72c4e70433b574b522578f729cec2cf68a5a5c2e34121c5439` |
| `confirmatory_C28_b1.jsonl` | 500 | `202402d6160137d9a0ff10a608daf45a9255150076309893c2abd7b515c302b3` |
| `confirmatory_C29_s1.jsonl` | 400 | `6e5bf218f848840481b3f50bb57c48ec8449404e2da9deddcd9c717824c82e20` |
| `confirmatory_C30_b4.jsonl` | 150 | `8b701fea46f059e101da14b95f76011d580b9a5d16ca94208dbaf296f93a8d48` |
| `confirmatory_C30_s1.jsonl` | 150 | `d48261ab38f958e5900d7765a870b7897941d2566fd33f568cad34d6083fab6f` |

These rows predate the repository's common provenance schema. Their original
GPU, container, driver, extension, and commit identities were not recorded and
cannot be recovered. In particular, the `sec` field is historical runtime
diagnostic data, not publishable benchmark evidence. The regenerated selection
and aggregate record current source hashes and Git state without filling in the
missing historical environment.

The per-row `x_half` field is a kernel-derived arithmetic sensitivity proxy for
the frozen binary64 point-model calculation. It is not an end-to-end
probability certificate. The aggregate deliberately does not combine it with
event sampling error or the published normalizer uncertainty into a confidence
interval or significance statistic.

Reproduction uses `examples/jiuzhang/decode_events.py`,
`select_confirmatory.py`, and `campaign_confirmatory.py`. The 744 MB raw USTC
archive and the Q7-1076 normalizer arrays are not redistributed; their hashes
and source locations are recorded in the generated artifacts.

With those external normalizer arrays installed at the recorded paths, rerun
the released aggregate against exactly the checkpoint files in this directory:

```bash
GBS_ALLOW_LEGACY_CONFIRMATORY=1 uv run python \
  examples/jiuzhang/campaign_confirmatory.py \
  --manifest results/jiuzhang/legacy_fixed_sample/private_fixed_sample_selection_v2.npz \
  --aggregate \
  --checkpoint-dir results/jiuzhang/legacy_fixed_sample \
  --out /tmp/private_fixed_sample_result_v2.reproduced.json
```

The explicit `--checkpoint-dir` is part of the audit contract: it prevents an
unrelated root-level checkpoint from being discovered and records portable
paths plus hashes for the released row inputs.
