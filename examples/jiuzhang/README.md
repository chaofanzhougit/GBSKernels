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

## Reconstruction (validated to 1.5 % RMS)

With squeezer `k` feeding input modes `2k, 2k+1`, `nbar = sinh²r`,
`m = sinh r cosh r`:

    <a†a> = conj(T)ᵀ diag(nbar) T,   <aa> = Tᵀ diag(m) T

then the xxpp Wigner covariance, the Husimi `Q`, and `O = I − Q⁻¹`. A click
pattern with `k` clicks has probability ∝ `tor(O_c)`, `O_c` the clicking modes
in both blocks (dimension `2k`). The reconstruction is validated by comparing the
theoretical per-detector click probability `1 − det(Q_{d,d+100})^{-1/2}` against
the empirical rate over 10⁶ decoded samples.

## Scripts

- `jiuzhang_frontier.py` — the fp64 certified κ-vs-clicks curve (Fig. 3, blue),
  `jiuzhang1_frontier_20260707T164417Z.json`.
- `dd_validate.py` — the on-device double-double certified frontier (Fig. 3,
  purple) + the enclosure gate, `dd_frontier_20260708T105821Z.json`
  (RTX 4090, `gpu_backend=gpu`, 30 events, 0 enclosure failures, tight to 26
  clicks).
- `make_precision_wall.py` — regenerates the precision-wall figure
  (`--out` to choose the path; default alongside the script).
- `click_count_dist.npy` — click-count histogram (101 int64 bins) over the
  2,995,852 decoded normal samples; used for the grey event histogram in the
  figure so it can be rebuilt without the 744 MB dataset.

Committed result JSONs carry their commit and container digest. Run from a
checkout with the CUDA extension built (`GBSKERNELS_EXT_DIR`) and the data files
in place; e.g. `python dd_validate.py --events 30 --kmax 26`.
