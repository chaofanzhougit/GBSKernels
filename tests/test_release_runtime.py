"""Release-session entry points must be present in the source tree."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPU_SESSION = ROOT / "scripts" / "gpu_session.sh"


def test_gpu_session_referenced_entry_points_exist() -> None:
    referenced = sorted(
        set(
            re.findall(
                r"\b((?:examples|scripts)/[A-Za-z0-9_./-]+\.(?:py|sh))\b",
                GPU_SESSION.read_text(),
            )
        )
    )
    assert referenced
    missing = [path for path in referenced if not (ROOT / path).is_file()]
    assert not missing, f"gpu_session.sh references missing files: {missing}"
