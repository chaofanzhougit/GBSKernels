"""Build the certified precision-wall figure from corrected run artifacts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import argparse
HERE = Path(__file__).resolve().parent
_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--fp64-artifact", type=Path, required=True,
                help="corrected jiuzhang_frontier.py JSON artifact")
_p.add_argument("--dd-artifact", type=Path, required=True,
                help="corrected dd_validate.py JSON artifact")
_p.add_argument("--out", type=Path, default=HERE / "precision_wall.pdf",
                help="output figure path (default: alongside this script)")
_args = _p.parse_args()
OUT = _args.out


def _load_corrected(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    expected = "q7_construction.build_state('squeezed')"
    if artifact.get("state_construction") != expected:
        raise SystemExit(
            f"{path} predates the corrected Q7 state construction; regenerate it"
        )
    return artifact


a = _load_corrected(_args.fp64_artifact)
c = a["subpattern_curve"]
k = np.array([r["clicks"] for r in c])
med = np.array([r["rel_bound_median"] for r in c])
lo = np.array([r["rel_bound_q05"] for r in c])
hi = np.array([r["rel_bound_q95"] for r in c])
dist = np.load(HERE / "click_count_dist.npy").astype(float)
kk = np.arange(len(dist))
frac_ge26 = dist[26:].sum() / dist.sum()
mean_clicks = (kk * dist).sum() / dist.sum()

plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.linewidth": 0.8,
                     "mathtext.fontset": "cm"})
fig, ax = plt.subplots(figsize=(6.6, 4.1))

# fp64-meaningless region (bound >= value)
ax.axhspan(1.0, 1e9, color="#d62728", alpha=0.06, zorder=0)
ax.axhline(1.0, color="#d62728", lw=1.1, ls="--", zorder=2)
ax.text(4.3, 1.6, r"fp64 meaningless: bound $\geq$ value (0 correct digits)",
        color="#d62728", fontsize=8.5, va="bottom")

# the Jiuzhang certified precision wall (fp64)
ax.fill_between(k, lo, hi, color="#1f77b4", alpha=0.18, zorder=3,
                label="fp64 certified bound (5–95%)")
ax.plot(k, med, "-o", color="#1f77b4", ms=4.5, lw=1.8, zorder=4,
        label="fp64 certified bound (median)")

# the double-double certificate tunnels under the wall (on-device, real GPU)
_dd = _load_corrected(_args.dd_artifact)
dk = np.array([r["clicks"] for r in _dd["rows"]])
dv = np.array([r["dd_rel_bound_median"] for r in _dd["rows"]])
ax.plot(dk, dv, "-s", color="#9467bd", ms=4.5, lw=1.8, zorder=5,
        label="double-double certified bound")

# where fp64 loses its last digit (curve crosses rel_bound = 1)
kx = np.interp(0.0, np.log10(med), k)   # click count where median bound = 1
ax.axvline(kx, color="#555", lw=0.9, ls=":", zorder=2)
ax.annotate("fp64 loses its\nlast digit ($k\\approx" + f"{int(np.ceil(kx))}$)", xy=(kx, 1e-3),
            xytext=(9.0, 3e-6), fontsize=8.5, color="#333",
            arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))

# published exact-validation ceiling
ax.axvline(26, color="#2ca02c", lw=1.1, ls="-.", zorder=2)
ax.text(26.2, 1e-10, "published exact-\nvalidation ceiling\n(26 clicks)",
        color="#2ca02c", fontsize=8.5, va="bottom")

# Borealis: distinct lower-photon dataset, observables in the safe regime
ax.plot([6], [1e-10], marker="*", ms=15, color="#ff7f0e", zorder=5,
        markeredgecolor="k", markeredgewidth=0.4)
ax.annotate("Borealis observables\n(distinct dataset; fp64 safe)", xy=(6, 1e-10),
            xytext=(4.2, 2e-9), fontsize=8.5, color="#cc6600")

# real Jiuzhang events live far past the wall (twin histogram)
ax2 = ax.twinx()
ax2.bar(kk, dist / dist.sum(), width=1.0, color="#7f7f7f", alpha=0.25, zorder=1)
ax2.set_ylim(0, (dist / dist.sum()).max() * 4.5)
ax2.set_yticks([])

ax.set_yscale("log")
ax.set_xlim(4, 50)
ax.set_ylim(1e-11, 1e8)
ax.set_xlabel("detector clicks $k$  (torontonian dimension $2k$)")
ax.set_ylabel("certified relative error bound  $E/|\\hat y|$")
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)

# annotate the real-event bulk on the main axis (no clipping)
ax.annotate(f"real Jiuzhang events live here:\nmean {mean_clicks:.0f} clicks, "
            f"{frac_ge26*100:.1f}% at/past the ceiling",
            xy=(mean_clicks, 3e-11), xytext=(28.5, 8e2), fontsize=8.5, color="#444",
            ha="left", arrowprops=dict(arrowstyle="->", color="#7f7f7f", lw=0.8))
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}  (fp64 last digit at k={kx:.1f}, {frac_ge26*100:.1f}% at/past 26)")
