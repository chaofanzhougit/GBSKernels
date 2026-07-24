#!/usr/bin/env python3
"""Stage the small, hash-bound inputs used by the public GPU validation.

The raw USTC archive and complete Zenodo bundle remain external.  The source
release ships five attributed runtime inputs plus one decoder-audit reference;
see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "jiuzhang" / "validation_data"

ASSETS = {
    "T_full.npy": ("jiuzhang1/T_full.npy", "57bbe04f127a6981a1f777ed93d54c9d8a3617bb8499d40c85e467d73890310d"),
    "events_band13_32.npy": ("jiuzhang1/events_band13_32.npy", "527767f265c2a5e8c9a14c3b61a4ee19011bf41642205fbc864af673db35c493"),
    "squeezing parameters.txt": ("jiuzhang1/squeezing parameters.txt", "3f553d39a086a89b255c377a335aa194382a3f37ff19c0b3c1551a3ee01d706f"),
    "empirical_click_rates.npy": ("jiuzhang1/empirical_click_rates.npy", "12cf5221f38a6894ef611d12674ccf3dc1cf2a3ec33d9a2dced3e8791d5a3f7d"),
    "click_probs_squeezed_0.npy": ("q7_1076_zenodo/click_probs/click_probs_squeezed_0.npy", "a17d938cc1eb0276829d3968edbd13d5ec524b81bb295dccf81120c574df5d63"),
    "click_probs_squashed_0.npy": ("q7_1076_zenodo/click_probs/click_probs_squashed_0.npy", "85b7fef71e4b56c1d35ff647bb2a05a0107d0bdc1ad496d2b6a30315c6ee6edf"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--force", action="store_true", help="replace mismatched destination files")
    args = ap.parse_args()

    for name, (relative, expected) in ASSETS.items():
        source = SOURCE / name
        if sha256(source) != expected:
            raise SystemExit(f"source hash mismatch: {source}")
        destination = args.data_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != expected and not args.force:
            raise SystemExit(f"destination exists with a different hash: {destination} (use --force)")
        shutil.copyfile(source, destination)
        print(f"{destination}  {expected}")


if __name__ == "__main__":
    main()
