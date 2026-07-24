# Small validation inputs

These six hash-bound files are the small inputs staged for the public Jiuzhang
workflows. Five are consumed by the `validate` GPU session;
`empirical_click_rates.npy` is retained for the decoder audit. They are copied
into the ignored `data/` layout by:

```bash
python scripts/prepare_validation_data.py
```

The raw USTC `web_0.zip` archive and the complete Quantum 7 Zenodo bundle
remain external. `T_full.npy`, the squeezing parameters, and the event-band
array are derived from the USTC release; `empirical_click_rates.npy` is the
decoder-audit reference for the USTC event stream. The two click-probability
arrays are retained from Zenodo record
[7194775](https://doi.org/10.5281/zenodo.7194775) under CC BY 4.0. See the
repository's [`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md) for
attribution and licensing scope. Every file is verified by SHA-256 before
staging.
