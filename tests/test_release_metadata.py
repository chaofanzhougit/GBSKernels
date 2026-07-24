"""Release metadata and source-reference consistency checks."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import gbskernels


ROOT = Path(__file__).resolve().parents[1]


def test_version_metadata_is_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    version = project["project"]["version"]
    locked = next(pkg for pkg in lock["package"] if pkg["name"] == "gbskernels")
    citation = (ROOT / "CITATION.cff").read_text()
    changelog = (ROOT / "CHANGELOG.md").read_text()
    readme = (ROOT / "README.md").read_text()

    assert gbskernels.__version__ == version
    assert locked["version"] == version
    assert re.search(rf"(?m)^version: {re.escape(version)}$", citation)
    assert f"## [{version}]" in changelog
    assert f"gbskernels=={version}" in readme


def test_referenced_public_docs_exist() -> None:
    reference = re.compile(r"(?<![A-Za-z0-9_.-])(docs/[A-Za-z0-9_.-]+\.md)")
    suffixes = {".cu", ".cuh", ".cpp", ".md", ".py", ".sh", ".toml", ".yml", ".yaml"}
    missing: list[tuple[Path, str]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part.startswith("build") or part in {".git", ".venv"} for part in path.parts):
            continue
        for target in reference.findall(path.read_text(errors="ignore")):
            if not (ROOT / target).is_file():
                missing.append((path.relative_to(ROOT), target))
    assert not missing
