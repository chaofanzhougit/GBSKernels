from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "publication_gpu_session.sh"

EXPECTED_GATES = {
    "check_permanent",
    "check_hafnian",
    "check_loop_hafnian",
    "check_torontonian",
    "check_torontonian_real_chol",
    "check_permanent_dd",
    "check_hafnian_dd",
    "check_loop_hafnian_dd",
    "check_torontonian_dd",
    "check_permanent_coop",
    "check_haf_coop",
    "check_lhaf_coop",
    "check_tor_coop",
    "check_haf_small",
    "check_permanent_warp",
    "check_certified",
    "check_repeated",
    "check_tor_recursive",
    "check_host_api",
    "check_session",
    "check_sampler_draw",
    "check_sampler_gather",
    "check_sampler_haf_varn",
    "check_sampler_session",
}


def test_publication_session_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_publication_session_embedded_python_has_valid_syntax():
    source = SCRIPT.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(?P<body>.*?)\nPY(?:\n|\Z)", source, re.DOTALL)
    assert len(blocks) >= 8
    for index, block in enumerate(blocks):
        compile(block, f"{SCRIPT}:heredoc-{index}", "exec")


def test_publication_session_derives_exact_constraints_from_lock(tmp_path):
    source = SCRIPT.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(?P<body>.*?)\nPY(?:\n|\Z)", source, re.DOTALL)
    exporter = next(block for block in blocks if "uv.lock contains no registry" in block)
    output = tmp_path / "constraints.txt"
    sdists = tmp_path / "sdists.txt"
    subprocess.run(
        [sys.executable, "-", str(ROOT / "uv.lock"), str(output), str(sdists)],
        input=exporter,
        text=True,
        check=True,
    )
    pins = {
        line for line in output.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert "numpy==2.4.6" in pins
    assert "python-flint==0.9.0" in pins
    assert "thewalrus==0.22.0" in pins
    assert "piquasso==8.0.1" in pins
    assert [
        line for line in sdists.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ] == ["antlr4-python3-runtime==4.9.2"]


def test_publication_session_requires_explicit_contract_and_spends_nothing():
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--source-root" in completed.stderr
    assert "is required" in completed.stderr


def test_publication_session_gate_set_and_scientific_workload_are_frozen():
    source = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"PUBLICATION_GATES=\(\n(?P<body>.*?)\n\)", source, re.DOTALL)
    assert match is not None
    gates = set(shlex.split(match.group("body")))
    assert gates == EXPECTED_GATES
    assert len(gates) == 24

    assert "--small-kmax 14" in source
    assert "--per-cell 4" in source
    assert "--structured-modes 25,26,27,28,29,30,31,32" in source
    assert "--include-gbskernels-dd" in source
    assert source.count("--require-provenance") == 2
    assert "--source-tree-sha256" in source
    assert '"case_count": 320' in source
    assert 'engines != {"gbskernels_dd", "walrus", "piquasso"}' in source


def test_publication_session_is_out_of_source_and_hashes_final_evidence():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "PYTHONPYCACHEPREFIX" not in source
    assert "uv.lock" in source
    assert "import tomllib" in source
    assert "--no-binary=:all: --no-build-isolation" in source
    assert '-c "$LOCK_CONSTRAINTS" -r "$REQUIREMENTS"' in source
    assert '-S "$SOURCE_ROOT/core" -B "$CORE_BUILD"' in source
    assert '-S "$SOURCE_ROOT/bindings" -B "$BINDINGS_BUILD"' in source
    assert "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" in source
    assert "--extract-ptx all" in source
    assert "--extract-elf all" in source
    assert "capture_build_provenance.py" in source
    assert "source_tree_inventory(root)" in source
    assert 'nvidia-smi -q >"$EVIDENCE_ROOT/device/nvidia-smi.txt"' in source
    assert 'dpkg-query -W' in source
    assert '"environment/dpkg-query.txt"' in source
    assert "ACTUAL_ARCHIVE_SHA256_FINAL" in source
    assert '"git_tree": git_tree' in source
    assert "build-provenance canonical digest is invalid" in source
    assert '"science/arb_enclosure_matrices.npz"' in source
    assert '"environment/uv-lock-constraints.txt"' in source
    assert '"environment/uv-lock-sdists.txt"' in source
    assert "Arb campaign corpus hash mismatch" in source
    assert "evidence_manifest.json" in source
    assert "allow_nan=False" in source
    assert "vastai" not in source.lower()
    assert "rsync" not in source.lower()
    assert "ssh " not in source.lower()
