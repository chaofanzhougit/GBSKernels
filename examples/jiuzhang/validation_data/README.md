# Small validation inputs

These six derived files are the complete data payload consumed by the public
`validate` GPU session. They are copied into the ignored `data/` layout by:

```bash
python scripts/prepare_validation_data.py
```

The raw USTC `web_0.zip` archive and the Quantum 7 Zenodo bundle remain
external inputs and are not redistributed here. `T_full.npy`, the squeezing
parameters, and the event-band array are derived from the USTC release; the
two click-probability arrays are the Jiuzhang-1.0 files from Zenodo record
7194775. `empirical_click_rates.npy` is the decoder-audit reference for the
USTC event stream. Every file is verified by SHA-256 before staging.
