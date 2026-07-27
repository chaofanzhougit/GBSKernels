#!/usr/bin/env bash
# Build and run the publication evidence campaign on an already-provisioned GPU.
# This script never provisions, terminates, pushes to, or pulls from a cloud host.

set -Eeuo pipefail
IFS=$'\n\t'
umask 022

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf 'ABORT: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
usage: publication_gpu_session.sh \
  --source-root <immutable-extracted-release> \
  --source-archive <release.tar[.gz]> \
  --archive-sha256 <64-hex> \
  --source-tree-sha256 <64-hex> \
  --git-commit <40-or-64-hex> \
  --git-tree <40-or-64-hex> \
  --container-digest <image@sha256:64-hex> \
  --output-root <new-external-directory> \
  [--cuda-arch 89] [--python python3.12] [--jobs N]

The source archive, extracted source, and output root must be distinct.  The
output root must not exist.  All builds, environments, wheels, and evidence are
created below it.  Vast provisioning and teardown are intentionally out of scope.
EOF
}

PUBLICATION_GATES=(
  check_permanent
  check_hafnian
  check_loop_hafnian
  check_torontonian
  check_torontonian_real_chol
  check_permanent_dd
  check_hafnian_dd
  check_loop_hafnian_dd
  check_torontonian_dd
  check_permanent_coop
  check_haf_coop
  check_lhaf_coop
  check_tor_coop
  check_haf_small
  check_permanent_warp
  check_certified
  check_repeated
  check_tor_recursive
  check_host_api
  check_session
  check_sampler_draw
  check_sampler_gather
  check_sampler_haf_varn
  check_sampler_session
)

SOURCE_ROOT=""
SOURCE_ARCHIVE=""
EXPECTED_ARCHIVE_SHA256=""
EXPECTED_SOURCE_TREE_SHA256=""
GIT_COMMIT=""
GIT_TREE=""
CONTAINER_DIGEST=""
OUTPUT_ROOT=""
CUDA_ARCH="89"
BOOTSTRAP_PYTHON="python3.12"
JOBS=""

need_value() {
  [ "$#" -ge 2 ] && [ -n "$2" ] || die "missing value for $1"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source-root) need_value "$@"; SOURCE_ROOT="$2"; shift 2 ;;
    --source-archive) need_value "$@"; SOURCE_ARCHIVE="$2"; shift 2 ;;
    --archive-sha256) need_value "$@"; EXPECTED_ARCHIVE_SHA256="$2"; shift 2 ;;
    --source-tree-sha256) need_value "$@"; EXPECTED_SOURCE_TREE_SHA256="$2"; shift 2 ;;
    --git-commit) need_value "$@"; GIT_COMMIT="$2"; shift 2 ;;
    --git-tree) need_value "$@"; GIT_TREE="$2"; shift 2 ;;
    --container-digest) need_value "$@"; CONTAINER_DIGEST="$2"; shift 2 ;;
    --output-root) need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --cuda-arch) need_value "$@"; CUDA_ARCH="$2"; shift 2 ;;
    --python) need_value "$@"; BOOTSTRAP_PYTHON="$2"; shift 2 ;;
    --jobs) need_value "$@"; JOBS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

for value_name in SOURCE_ROOT SOURCE_ARCHIVE EXPECTED_ARCHIVE_SHA256 \
                  EXPECTED_SOURCE_TREE_SHA256 GIT_COMMIT \
                  GIT_TREE CONTAINER_DIGEST OUTPUT_ROOT; do
  [ -n "${!value_name}" ] || { usage >&2; die "--${value_name,,} is required"; }
done
[[ "$EXPECTED_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || die "archive SHA-256 must be 64 lowercase hexadecimal characters"
[[ "$EXPECTED_SOURCE_TREE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || die "source-tree SHA-256 must be 64 lowercase hexadecimal characters"
[[ "$GIT_COMMIT" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]] \
  || die "git commit must be a 40- or 64-character lowercase object ID"
[[ "$GIT_TREE" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]] \
  || die "git tree must be a 40- or 64-character lowercase object ID"
[[ "$CONTAINER_DIGEST" =~ ^[^[:space:]@]+(/[^[:space:]@]+)*@sha256:[0-9a-f]{64}$ ]] \
  || die "container digest must be image@sha256:<64 lowercase hex>"
[[ "$CUDA_ARCH" =~ ^[0-9]+$ ]] || die "CUDA architecture must be numeric"
if [ -n "$JOBS" ]; then
  [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
fi

command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1 \
  || die "Python 3.12 executable not found: $BOOTSTRAP_PYTHON"
"$BOOTSTRAP_PYTHON" -c \
  'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' \
  || die "the publication environment requires Python 3.12"

canonical_path() {
  "$BOOTSTRAP_PYTHON" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
    "$1"
}

SOURCE_ROOT="$(canonical_path "$SOURCE_ROOT")"
SOURCE_ARCHIVE="$(canonical_path "$SOURCE_ARCHIVE")"
OUTPUT_ROOT="$(canonical_path "$OUTPUT_ROOT")"
[ -d "$SOURCE_ROOT" ] || die "source root is not a directory: $SOURCE_ROOT"
[ ! -L "$SOURCE_ROOT" ] || die "source root must not be a symbolic link"
[ -f "$SOURCE_ARCHIVE" ] && [ ! -L "$SOURCE_ARCHIVE" ] \
  || die "source archive is not a regular file: $SOURCE_ARCHIVE"
[ ! -e "$OUTPUT_ROOT" ] || die "output root already exists: $OUTPUT_ROOT"

inside_source() {
  [[ "$1" == "$SOURCE_ROOT" || "$1" == "$SOURCE_ROOT/"* ]]
}
inside_source "$SOURCE_ARCHIVE" && die "source archive must be outside source root"
inside_source "$OUTPUT_ROOT" && die "output root must be outside source root"
[ ! -e "$SOURCE_ROOT/.git" ] \
  || die "source root must be an archive extraction without .git metadata"

REQUIRED_SOURCE_FILES=(
  pyproject.toml
  uv.lock
  envs/publication-requirements.txt
  scripts/capture_build_provenance.py
  scripts/publication_gpu_session.sh
  examples/jiuzhang/arb_enclosure_campaign.py
  bench/torontonian_baselines.py
  core/CMakeLists.txt
  bindings/CMakeLists.txt
)
for relative in "${REQUIRED_SOURCE_FILES[@]}"; do
  [ -f "$SOURCE_ROOT/$relative" ] || die "release source is missing $relative"
done

REQUIRED_COMMANDS=(sha256sum cmake ninja nvcc cuobjdump nvidia-smi c++ tar dpkg-query sort)
for tool in "${REQUIRED_COMMANDS[@]}"; do
  command -v "$tool" >/dev/null 2>&1 || die "required command is unavailable: $tool"
done
[ "${#PUBLICATION_GATES[@]}" -eq 24 ] || die "internal gate list is not exactly 24 entries"

ACTUAL_ARCHIVE_SHA256="$(sha256sum "$SOURCE_ARCHIVE" | awk '{print $1}')"
[ "$ACTUAL_ARCHIVE_SHA256" = "$EXPECTED_ARCHIVE_SHA256" ] \
  || die "source archive SHA-256 mismatch"

export PYTHONDONTWRITEBYTECODE=1
export GBS_COMMIT="$GIT_COMMIT"
export GBS_CONTAINER_DIGEST="$CONTAINER_DIGEST"
export GBS_SOURCE_ARCHIVE_SHA256="$EXPECTED_ARCHIVE_SHA256"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GOTO_NUM_THREADS=1
unset PYTHONPATH || true

SOURCE_TREE_SHA256="$(
  "$BOOTSTRAP_PYTHON" - "$SOURCE_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from scripts.capture_build_provenance import source_tree_inventory

print(source_tree_inventory(root)["tree_sha256"])
PY
)"
[[ "$SOURCE_TREE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to compute the immutable source-tree digest"
[ "$SOURCE_TREE_SHA256" = "$EXPECTED_SOURCE_TREE_SHA256" ] \
  || die "extracted source-tree SHA-256 does not match the publication contract"

mkdir -p "$OUTPUT_ROOT"
WORK_ROOT="$OUTPUT_ROOT/work"
EVIDENCE_ROOT="$OUTPUT_ROOT/evidence"
VENV_ROOT="$WORK_ROOT/venv"
CORE_BUILD="$WORK_ROOT/core-build"
BINDINGS_BUILD="$WORK_ROOT/bindings-build"
WHEEL_EXTRACTION="$WORK_ROOT/wheel-source"
RUNTIME_ROOT="$WORK_ROOT/runtime"
mkdir -p "$WORK_ROOT" "$EVIDENCE_ROOT" "$RUNTIME_ROOT" \
         "$EVIDENCE_ROOT/environment" "$EVIDENCE_ROOT/build" \
         "$EVIDENCE_ROOT/gates" "$EVIDENCE_ROOT/device" \
         "$EVIDENCE_ROOT/wheel" "$EVIDENCE_ROOT/binary" \
         "$EVIDENCE_ROOT/science"

LC_ALL=C nvidia-smi -q >"$EVIDENCE_ROOT/device/nvidia-smi.txt"
[ -s "$EVIDENCE_ROOT/device/nvidia-smi.txt" ] \
  || die "nvidia-smi produced no device evidence"
LC_ALL=C dpkg-query -W -f='${binary:Package}\t${Version}\n' \
  | LC_ALL=C sort >"$EVIDENCE_ROOT/environment/dpkg-query.txt"
[ -s "$EVIDENCE_ROOT/environment/dpkg-query.txt" ] \
  || die "dpkg-query produced no operating-system package evidence"

"$BOOTSTRAP_PYTHON" - "$EVIDENCE_ROOT/session_contract.json" \
  "$SOURCE_ROOT" "$SOURCE_ARCHIVE" "$EXPECTED_ARCHIVE_SHA256" \
  "$SOURCE_TREE_SHA256" "$GIT_COMMIT" "$GIT_TREE" \
  "$CONTAINER_DIGEST" "$CUDA_ARCH" <<'PY'
from pathlib import Path
import json
import sys

output = Path(sys.argv[1])
payload = {
    "schema": "gbskernels.publication-gpu-session-contract.v1",
    "source_root": sys.argv[2],
    "source_archive": sys.argv[3],
    "source_archive_sha256": sys.argv[4],
    "source_tree_sha256": sys.argv[5],
    "git_commit": sys.argv[6],
    "git_tree": sys.argv[7],
    "container_digest": sys.argv[8],
    "cuda_arch": "sm_" + sys.argv[9],
}
with output.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY

CURRENT_STAGE="Python environment"
say "$CURRENT_STAGE"
"$BOOTSTRAP_PYTHON" -m venv "$VENV_ROOT"
PYTHON="$VENV_ROOT/bin/python"
"$PYTHON" -c \
  'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' \
  || die "created environment is not Python 3.12"

LOCK_CONSTRAINTS="$EVIDENCE_ROOT/environment/uv-lock-constraints.txt"
LOCK_SDISTS="$EVIDENCE_ROOT/environment/uv-lock-sdists.txt"
"$PYTHON" - "$SOURCE_ROOT/uv.lock" "$LOCK_CONSTRAINTS" "$LOCK_SDISTS" <<'PY'
from pathlib import Path
import re
import sys
import tomllib

lock_path = Path(sys.argv[1])
output = Path(sys.argv[2])
sdist_output = Path(sys.argv[3])
lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
if lock.get("version") != 1:
    raise SystemExit(f"unsupported uv.lock schema version: {lock.get('version')!r}")

pins = {}
sdist_pins = {}
for package in lock.get("package", []):
    source = package.get("source", {})
    if "registry" not in source:
        continue
    name = package.get("name")
    version = package.get("version")
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
        or not isinstance(version, str)
        or not version
        or any(character.isspace() for character in version)
    ):
        raise SystemExit(f"invalid registry package in uv.lock: {package!r}")
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    previous = pins.setdefault(normalized, version)
    if previous != version:
        raise SystemExit(
            f"uv.lock contains multiple versions for {normalized}: "
            f"{previous!r} and {version!r}"
        )
    if not package.get("wheels"):
        sdist_pins[normalized] = version
if not pins:
    raise SystemExit("uv.lock contains no registry package pins")
if not sdist_pins:
    raise SystemExit("uv.lock contains no explicitly handled sdist-only packages")

with output.open("x", encoding="utf-8", newline="\n") as handle:
    handle.write("# Exact registry versions derived from the immutable uv.lock.\n")
    for name, version in sorted(pins.items()):
        handle.write(f"{name}=={version}\n")
with sdist_output.open("x", encoding="utf-8", newline="\n") as handle:
    handle.write("# Sdist-only packages built with the pinned bootstrap toolchain.\n")
    for name, version in sorted(sdist_pins.items()):
        handle.write(f"{name}=={version}\n")
PY
[ -s "$LOCK_CONSTRAINTS" ] || die "failed to derive constraints from uv.lock"
[ -s "$LOCK_SDISTS" ] || die "failed to identify sdist-only locked dependencies"

BOOTSTRAP_PINS=(
  pip==25.1.1
  setuptools==80.9.0
  wheel==0.45.1
)
"$PYTHON" -m pip install --disable-pip-version-check --no-cache-dir \
  --only-binary=:all: --upgrade \
  --report "$EVIDENCE_ROOT/environment/bootstrap-install-report.json" \
  "${BOOTSTRAP_PINS[@]}"
"$PYTHON" -m pip install --disable-pip-version-check --no-cache-dir \
  --no-binary=:all: --no-build-isolation \
  --report "$EVIDENCE_ROOT/environment/sdist-install-report.json" \
  -c "$LOCK_CONSTRAINTS" -r "$LOCK_SDISTS"

REQUIREMENTS="$SOURCE_ROOT/envs/publication-requirements.txt"
"$PYTHON" - "$REQUIREMENTS" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
pins = []
for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    line = raw.split("#", 1)[0].strip()
    if not line:
        continue
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_.,-]+\])?==[^\s;]+", line):
        raise SystemExit(f"unlocked publication requirement at {path}:{number}: {line}")
    pins.append(line)
if not pins:
    raise SystemExit("publication requirements are empty")
PY
"$PYTHON" -m pip install --disable-pip-version-check --no-cache-dir \
  --only-binary=:all: \
  --report "$EVIDENCE_ROOT/environment/publication-install-report.json" \
  -c "$LOCK_CONSTRAINTS" -r "$REQUIREMENTS"
"$PYTHON" -m pip check

if [ -z "$JOBS" ]; then
  JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "could not determine a positive build parallelism"
NVCC="$(command -v nvcc)"
CMAKE="$(command -v cmake)"
CXX="$(command -v c++)"
CUOBJDUMP="$(command -v cuobjdump)"
CUDA_AUDIT_FLAGS="--fmad=false --ftz=false --prec-div=true --prec-sqrt=true -lineinfo -Xcompiler=-fno-fast-math,-ffp-contract=off"
CXX_AUDIT_FLAGS="-fno-fast-math -ffp-contract=off"

CURRENT_STAGE="out-of-source CUDA core build"
say "$CURRENT_STAGE"
"$CMAKE" -S "$SOURCE_ROOT/core" -B "$CORE_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_FLAGS="$CUDA_AUDIT_FLAGS" \
  -DCMAKE_CXX_FLAGS="$CXX_AUDIT_FLAGS" \
  >"$EVIDENCE_ROOT/build/core-configure.log" 2>&1
"$CMAKE" --build "$CORE_BUILD" --parallel "$JOBS" \
  >"$EVIDENCE_ROOT/build/core-build.log" 2>&1
[ -s "$CORE_BUILD/compile_commands.json" ] \
  || die "core build did not emit compile_commands.json"

CURRENT_STAGE="24 on-device gates"
say "$CURRENT_STAGE"
GATE_BINARIES=()
for gate in "${PUBLICATION_GATES[@]}"; do
  executable="$CORE_BUILD/$gate"
  log="$EVIDENCE_ROOT/gates/${gate}.log"
  [ -x "$executable" ] || die "gate executable is missing: $executable"
  GATE_BINARIES+=("$executable")
  if ! "$executable" >"$log" 2>&1; then
    cat "$log" >&2
    die "GPU gate failed: $gate"
  fi
  last_line="$(awk 'NF {line=$0} END {print line}' "$log")"
  [ "$last_line" = "PASS" ] || { cat "$log" >&2; die "GPU gate lacks final PASS: $gate"; }
done
[ "$(find "$EVIDENCE_ROOT/gates" -type f -name '*.log' | wc -l | tr -d ' ')" = "24" ] \
  || die "gate evidence count is not 24"

CURRENT_STAGE="out-of-source nanobind extension build"
say "$CURRENT_STAGE"
"$CMAKE" -S "$SOURCE_ROOT/bindings" -B "$BINDINGS_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_FLAGS="$CUDA_AUDIT_FLAGS" \
  -DCMAKE_CXX_FLAGS="$CXX_AUDIT_FLAGS" \
  -DPython_EXECUTABLE="$PYTHON" \
  >"$EVIDENCE_ROOT/build/bindings-configure.log" 2>&1
"$CMAKE" --build "$BINDINGS_BUILD" --parallel "$JOBS" \
  >"$EVIDENCE_ROOT/build/bindings-build.log" 2>&1
[ -s "$BINDINGS_BUILD/compile_commands.json" ] \
  || die "bindings build did not emit compile_commands.json"
mapfile -d '' -t EXTENSIONS < <(
  find "$BINDINGS_BUILD" -type f -name 'gbskernels_ext*.so' -print0 | sort -z
)
[ "${#EXTENSIONS[@]}" -eq 1 ] \
  || die "expected exactly one compiled gbskernels extension, found ${#EXTENSIONS[@]}"
EXTENSION="${EXTENSIONS[0]}"
EVIDENCE_EXTENSION="$EVIDENCE_ROOT/binary/$(basename "$EXTENSION")"
cp "$EXTENSION" "$EVIDENCE_EXTENSION"
cmp "$EXTENSION" "$EVIDENCE_EXTENSION" \
  || die "retained extension differs from the compiled extension"
export GBSKERNELS_EXT_DIR="$BINDINGS_BUILD"

CURRENT_STAGE="embedded device-code audit"
say "$CURRENT_STAGE"
SASS_DUMP="$EVIDENCE_ROOT/device/embedded.sass"
(
  cd "$EVIDENCE_ROOT/device"
  "$CUOBJDUMP" --list-ptx "$EXTENSION" > ptx-list.txt
  "$CUOBJDUMP" --list-elf "$EXTENSION" > elf-list.txt
  "$CUOBJDUMP" --extract-ptx all "$EXTENSION" > extract-ptx.log 2>&1
  "$CUOBJDUMP" --dump-sass "$EXTENSION" > "$SASS_DUMP"
)
"$BOOTSTRAP_PYTHON" - "$EVIDENCE_ROOT/device/elf-list.txt" "$SASS_DUMP" <<'PY'
from pathlib import Path
import sys

elf_list = Path(sys.argv[1]).read_text(encoding="utf-8")
sass = Path(sys.argv[2]).read_text(encoding="utf-8")
if not any(
    line.lstrip().startswith("ELF file")
    and line.rstrip().endswith((".cubin", ".fatbin"))
    for line in elf_list.splitlines()
):
    raise SystemExit("cuobjdump listed no embedded cubin/fatbin")
if "Fatbin elf code:" not in sass or "Function :" not in sass:
    raise SystemExit("cuobjdump produced no recognizable embedded SASS")
PY
mapfile -d '' -t PTX_FILES < <(
  find "$EVIDENCE_ROOT/device" -maxdepth 1 -type f -name '*.ptx' -print0 | sort -z
)
DEVICE_CODE_FILES=("$SASS_DUMP")
for ptx_file in "${PTX_FILES[@]}"; do
  DEVICE_CODE_FILES+=("$ptx_file")
done
for device_file in "${DEVICE_CODE_FILES[@]}"; do
  [ -s "$device_file" ] || die "empty device-code evidence: $device_file"
done

CURRENT_STAGE="wheel build from separate archive extraction"
say "$CURRENT_STAGE"
mkdir -p "$WHEEL_EXTRACTION"
WHEEL_SOURCE="$(
  "$PYTHON" - "$SOURCE_ARCHIVE" "$WHEEL_EXTRACTION" <<'PY'
from pathlib import Path
import sys
import tarfile

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
with tarfile.open(archive, "r:*") as handle:
    handle.extractall(destination, filter="data")
entries = list(destination.iterdir())
root = entries[0] if len(entries) == 1 and entries[0].is_dir() else destination
if not (root / "pyproject.toml").is_file():
    raise SystemExit("release archive does not contain a buildable project root")
print(root.resolve())
PY
)"
WHEEL_SOURCE_SHA256="$(
  "$PYTHON" - "$SOURCE_ROOT" "$WHEEL_SOURCE" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
sys.path.insert(0, str(source))
from scripts.capture_build_provenance import source_tree_inventory

expected = source_tree_inventory(source)["tree_sha256"]
actual = source_tree_inventory(Path(sys.argv[2]))["tree_sha256"]
if actual != expected:
    raise SystemExit(f"wheel extraction tree mismatch: {actual} != {expected}")
print(actual)
PY
)"
[ "$WHEEL_SOURCE_SHA256" = "$SOURCE_TREE_SHA256" ] \
  || die "wheel source does not match immutable release source"
"$PYTHON" -m build --wheel --no-isolation \
  --outdir "$EVIDENCE_ROOT/wheel" "$WHEEL_SOURCE" \
  >"$EVIDENCE_ROOT/build/wheel-build.log" 2>&1
mapfile -d '' -t WHEELS < <(
  find "$EVIDENCE_ROOT/wheel" -maxdepth 1 -type f -name '*.whl' -print0 | sort -z
)
[ "${#WHEELS[@]}" -eq 1 ] || die "expected exactly one built wheel"
WHEEL="${WHEELS[0]}"
"$PYTHON" -m pip install --disable-pip-version-check --no-cache-dir \
  --force-reinstall --no-deps \
  --report "$EVIDENCE_ROOT/environment/wheel-install-report.json" "$WHEEL"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze --all > "$EVIDENCE_ROOT/environment/pip-freeze.txt"
[ -s "$EVIDENCE_ROOT/environment/pip-freeze.txt" ] || die "pip freeze evidence is empty"

(
  cd "$RUNTIME_ROOT"
  "$PYTHON" - "$VENV_ROOT" "$EXTENSION" <<'PY'
from pathlib import Path
import sys

venv = Path(sys.argv[1]).resolve()
expected_extension = Path(sys.argv[2]).resolve()
import gbskernels

package = Path(gbskernels.__file__).resolve()
if venv not in package.parents:
    raise SystemExit(f"gbskernels was not imported from the built wheel: {package}")
if gbskernels.gpu_backend_kind() != "gpu":
    raise SystemExit("compiled extension does not identify as a real GPU backend")
loaded = Path(gbskernels._load_gpu_ext().__file__).resolve()
if loaded != expected_extension:
    raise SystemExit(f"loaded extension mismatch: {loaded} != {expected_extension}")
print(f"wheel package: {package}")
print(f"GPU extension: {loaded}")
PY
) > "$EVIDENCE_ROOT/build/wheel-extension-smoke.log" 2>&1

CURRENT_STAGE="source/build provenance capture"
say "$CURRENT_STAGE"
CAPTURE_ARGS=(
  --source-archive "$SOURCE_ARCHIVE"
  --expected-source-archive-sha256 "$EXPECTED_ARCHIVE_SHA256"
  --source-tree "$SOURCE_ROOT"
  --expected-source-tree-sha256 "$SOURCE_TREE_SHA256"
  --git-commit "$GIT_COMMIT"
  --git-tree "$GIT_TREE"
  --container-digest "$CONTAINER_DIGEST"
  --compile-commands "$CORE_BUILD/compile_commands.json"
  --compile-commands "$BINDINGS_BUILD/compile_commands.json"
  --cmake-cache "$CORE_BUILD/CMakeCache.txt"
  --cmake-cache "$BINDINGS_BUILD/CMakeCache.txt"
  --build-product "$CORE_BUILD/bench_kernels"
  --build-product "$EVIDENCE_ROOT/environment/pip-freeze.txt"
  --build-product "$EVIDENCE_ROOT/device/elf-list.txt"
  --extension "$EVIDENCE_EXTENSION"
  --wheel "$WHEEL"
  --nvcc "$NVCC"
  --cmake "$CMAKE"
  --cxx "$CXX"
  --cuobjdump "$CUOBJDUMP"
  --output "$EVIDENCE_ROOT/build_provenance.json"
)
for executable in "${GATE_BINARIES[@]}"; do
  CAPTURE_ARGS+=(--build-product "$executable")
done
for device_file in "${DEVICE_CODE_FILES[@]}"; do
  CAPTURE_ARGS+=(--device-code "$device_file")
done
"$PYTHON" "$SOURCE_ROOT/scripts/capture_build_provenance.py" "${CAPTURE_ARGS[@]}"
GBS_BUILD_MANIFEST_SHA256="$(
  "$PYTHON" - "$EVIDENCE_ROOT/build_provenance.json" <<'PY'
from pathlib import Path
import json
import sys

payload = json.loads(
    Path(sys.argv[1]).read_text(encoding="utf-8"),
    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
)
value = payload.get("manifest_sha256")
if not isinstance(value, str) or len(value) != 64:
    raise SystemExit("build provenance has no valid manifest SHA-256")
print(value)
PY
)"
export GBS_BUILD_MANIFEST_SHA256

CURRENT_STAGE="Arb enclosure campaign (312 small plus k=25..32 structured)"
say "$CURRENT_STAGE"
(
  cd "$RUNTIME_ROOT"
  "$PYTHON" - "$SOURCE_ROOT/examples/jiuzhang/arb_enclosure_campaign.py" \
    --backend gpu \
    --small-kmax 14 \
    --per-cell 4 \
    --seed 20260714 \
    --structured-modes 25,26,27,28,29,30,31,32 \
    --jiuzhang-data "$SOURCE_ROOT/examples/jiuzhang/validation_data" \
    --target-bits 80 \
    --max-precision-bits 2048 \
    --require-provenance \
    --output "$EVIDENCE_ROOT/science/arb_enclosure_campaign.json" \
    --corpus "$EVIDENCE_ROOT/science/arb_enclosure_matrices.npz" <<'PY'
from pathlib import Path
import runpy
import sys

script = Path(sys.argv[1]).resolve()
arguments = sys.argv[2:]
# Preload the installed wheel before the campaign adds the immutable source to
# sys.path for its non-packaged driver and Jiuzhang construction modules.
import gbskernels

if "site-packages" not in str(Path(gbskernels.__file__).resolve()):
    raise SystemExit("Arb campaign did not preload gbskernels from the wheel")
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), *arguments]
runpy.run_path(str(script), run_name="__main__")
PY
) > "$EVIDENCE_ROOT/science/arb_enclosure_campaign.log" 2>&1

CURRENT_STAGE="matched GPU-DD and recursive CPU baselines"
say "$CURRENT_STAGE"
(
  cd "$RUNTIME_ROOT"
  "$PYTHON" -m bench.torontonian_baselines \
    --modes 4,8,12,16,20 \
    --matrices-per-size 3 \
    --warmups 2 \
    --repeats 7 \
    --regime loss \
    --seed 20260726 \
    --agreement-atol 0 \
    --agreement-rtol 1e-8 \
    --arb-oracle-max-modes 20 \
    --arb-target-bits 80 \
    --arb-max-precision-bits 2048 \
    --include-gbskernels-dd \
    --require-provenance \
    --out "$EVIDENCE_ROOT/science/torontonian_matched_baselines.json"
) > "$EVIDENCE_ROOT/science/torontonian_matched_baselines.log" 2>&1

CURRENT_STAGE="final evidence validation and hashing"
say "$CURRENT_STAGE"
ACTUAL_ARCHIVE_SHA256_FINAL="$(sha256sum "$SOURCE_ARCHIVE" | awk '{print $1}')"
[ "$ACTUAL_ARCHIVE_SHA256_FINAL" = "$EXPECTED_ARCHIVE_SHA256" ] \
  || die "source archive changed during the publication session"
SOURCE_TREE_FINAL="$(
  "$PYTHON" - "$SOURCE_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from scripts.capture_build_provenance import source_tree_inventory

print(source_tree_inventory(root)["tree_sha256"])
PY
)"
[ "$SOURCE_TREE_FINAL" = "$SOURCE_TREE_SHA256" ] \
  || die "immutable source tree changed during the publication session"

"$PYTHON" - "$EVIDENCE_ROOT" "$SOURCE_TREE_SHA256" \
  "$EXPECTED_ARCHIVE_SHA256" "$GBS_BUILD_MANIFEST_SHA256" \
  "$GIT_COMMIT" "$GIT_TREE" "$CONTAINER_DIGEST" <<'PY'
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import math
import sys

root = Path(sys.argv[1]).resolve()
tree_sha256, archive_sha256, build_sha256, commit, git_tree, container = sys.argv[2:]
output = root / "evidence_manifest.json"
if output.exists():
    raise SystemExit("refusing to overwrite final evidence manifest")

def reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value}")

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result

def load_strict(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )

def sha256(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"evidence changed while hashing: {path}")
    return digest.hexdigest()

build = load_strict(root / "build_provenance.json")
if build.get("manifest_sha256") != build_sha256:
    raise SystemExit("build-provenance manifest hash mismatch")
unsigned_build = dict(build)
unsigned_build.pop("manifest_sha256", None)
canonical_build = (json.dumps(
    unsigned_build,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
) + "\n").encode("ascii")
if hashlib.sha256(canonical_build).hexdigest() != build_sha256:
    raise SystemExit("build-provenance canonical digest is invalid")
build_source = build.get("source", {})
if (
    build_source.get("git_commit") != commit
    or build_source.get("git_tree") != git_tree
    or build_source.get("archive", {}).get("sha256") != archive_sha256
    or build_source.get("tree", {}).get("tree_sha256") != tree_sha256
    or build.get("container_digest") != container
):
    raise SystemExit("build provenance does not match the session contract")
campaign = load_strict(root / "science/arb_enclosure_campaign.json")
summary = campaign.get("summary", {})
expected_summary = {
    "case_count": 320,
    "proved": 320,
    "inconclusive": 0,
    "violation": 0,
    "refused": 0,
    "gate_pass": True,
}
for key, expected in expected_summary.items():
    if summary.get(key) != expected:
        raise SystemExit(f"Arb campaign {key}={summary.get(key)!r}, expected {expected!r}")
if campaign.get("source_archive_sha256") != archive_sha256:
    raise SystemExit("Arb campaign source-archive binding mismatch")
if campaign.get("build_manifest_sha256") != build_sha256:
    raise SystemExit("Arb campaign build-manifest binding mismatch")
campaign_corpus = root / "science/arb_enclosure_matrices.npz"
if campaign.get("corpus", {}).get("sha256") != sha256(campaign_corpus):
    raise SystemExit("Arb campaign corpus hash mismatch")
campaign_provenance = campaign.get("provenance", {})
if (
    campaign_provenance.get("commit") != commit
    or campaign_provenance.get("container_digest") != container
):
    raise SystemExit("Arb campaign provenance does not match the session contract")

baseline = load_strict(root / "science/torontonian_matched_baselines.json")
if baseline.get("schema_version") != 4:
    raise SystemExit("matched baseline schema version is not the registered version")
baseline_engine_rows = baseline.get("engines", [])
engines = {row.get("name") for row in baseline_engine_rows}
if engines != {"gbskernels_dd", "walrus", "piquasso"}:
    raise SystemExit(f"matched baseline engine set is incomplete: {engines}")
if len(baseline_engine_rows) != len(engines):
    raise SystemExit("matched baseline engine records are duplicated")
if baseline.get("commit") != commit or baseline.get("container_digest") != container:
    raise SystemExit("matched baseline provenance does not match the session contract")
if baseline.get("source_archive_sha256") != archive_sha256:
    raise SystemExit("matched baseline source-archive binding mismatch")
if baseline.get("build_manifest_sha256") != build_sha256:
    raise SystemExit("matched baseline build-manifest binding mismatch")
parameters = baseline.get("parameters", {})
expected_parameters = {
    "modes": [4, 8, 12, 16, 20],
    "matrices_per_size": 3,
    "repeats": 7,
    "warmups": 2,
    "regime": "loss",
    "seed": 20260726,
    "agreement_atol": 0.0,
    "agreement_rtol": 1e-8,
    "arb_oracle_max_modes": 20,
    "arb_target_bits": 80,
    "arb_max_precision_bits": 2048,
}
for key, expected in expected_parameters.items():
    if parameters.get(key) != expected:
        raise SystemExit(
            f"matched baseline {key}={parameters.get(key)!r}, expected {expected!r}"
        )
engine_rows = {row.get("name"): row for row in baseline_engine_rows}
if engine_rows.get("gbskernels_dd", {}).get("execution_device") != "gpu":
    raise SystemExit("matched GBSKernels engine did not record GPU execution")
if any(
    engine_rows.get(name, {}).get("execution_device") != "cpu"
    for name in ("walrus", "piquasso")
):
    raise SystemExit("matched recursive baselines did not record CPU execution")
performance = baseline.get("performance", {})
if len(performance.get("summary", [])) != 15 or len(performance.get("raw", [])) != 105:
    raise SystemExit("matched baseline performance row counts are incomplete")
agreement_rows = baseline.get("numerical_agreement", {}).get("rows", [])
if len(agreement_rows) != 15:
    raise SystemExit("matched baseline agreement row count is incomplete")

def finite_number(value, label, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label} is not a JSON number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise SystemExit(f"{label} is not finite") from None
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise SystemExit(f"{label} is not finite and nonnegative")
    return result

def complex_record(value, label):
    if not isinstance(value, dict) or set(value) != {"real", "imag"}:
        raise SystemExit(f"{label} is not a complex-number record")
    return complex(
        finite_number(value["real"], f"{label}.real"),
        finite_number(value["imag"], f"{label}.imag"),
    )

def dyadic_fraction(value, label):
    if not isinstance(value, dict) or set(value) != {"mantissa", "exponent"}:
        raise SystemExit(f"{label} is not a dyadic endpoint")
    mantissa_text = value["mantissa"]
    exponent = value["exponent"]
    if not isinstance(mantissa_text, str):
        raise SystemExit(f"{label} mantissa is not a decimal string")
    if isinstance(exponent, bool) or not isinstance(exponent, int):
        raise SystemExit(f"{label} exponent is not an integer")
    try:
        mantissa = int(mantissa_text)
    except ValueError:
        raise SystemExit(f"{label} mantissa is not an integer") from None
    if str(mantissa) != mantissa_text:
        raise SystemExit(f"{label} mantissa is not canonical")
    if exponent >= 0:
        return Fraction(mantissa << exponent, 1)
    return Fraction(mantissa, 1 << -exponent)

def rational_fraction(value, label):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise SystemExit(f"{label} is not a rational record")
    numerator_text = value["numerator"]
    denominator_text = value["denominator"]
    if not isinstance(numerator_text, str) or not isinstance(denominator_text, str):
        raise SystemExit(f"{label} rational components are not decimal strings")
    try:
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except ValueError:
        raise SystemExit(f"{label} rational components are not integers") from None
    if denominator <= 0:
        raise SystemExit(f"{label} denominator is not positive")
    result = Fraction(numerator, denominator)
    if (
        str(result.numerator) != numerator_text
        or str(result.denominator) != denominator_text
    ):
        raise SystemExit(f"{label} rational record is not canonical")
    return result

input_rows = baseline.get("inputs", {}).get("matrices", [])
if not isinstance(input_rows, list) or len(input_rows) != 15:
    raise SystemExit("matched baseline frozen input row count is incomplete")
inputs_by_id = {}
for row in input_rows:
    matrix_id = row.get("matrix_id") if isinstance(row, dict) else None
    matrix_hash = row.get("sha256") if isinstance(row, dict) else None
    if not isinstance(matrix_id, str) or not matrix_id or matrix_id in inputs_by_id:
        raise SystemExit("matched baseline frozen input identifiers are invalid")
    if (
        not isinstance(matrix_hash, str)
        or len(matrix_hash) != 64
        or any(character not in "0123456789abcdef" for character in matrix_hash)
    ):
        raise SystemExit(f"matched baseline frozen input hash is invalid: {matrix_id}")
    inputs_by_id[matrix_id] = row

expected_pairs = {
    frozenset(("gbskernels_dd", "walrus")),
    frozenset(("gbskernels_dd", "piquasso")),
    frozenset(("walrus", "piquasso")),
}
agreement_by_id = {}
for row in agreement_rows:
    matrix_id = row.get("matrix_id") if isinstance(row, dict) else None
    if matrix_id not in inputs_by_id or matrix_id in agreement_by_id:
        raise SystemExit("matched baseline agreement identifiers are invalid")
    input_row = inputs_by_id[matrix_id]
    if row.get("modes") != input_row.get("modes"):
        raise SystemExit(f"matched baseline agreement modes disagree for {matrix_id}")
    values_payload = row.get("values")
    if not isinstance(values_payload, dict) or set(values_payload) != engines:
        raise SystemExit("matched implementation value record is incomplete")
    values = {
        name: complex_record(payload, f"{matrix_id}.{name}")
        for name, payload in values_payload.items()
    }
    bounds = row.get("reported_abs_error_bounds")
    if not isinstance(bounds, dict) or set(bounds) != engines:
        raise SystemExit("matched implementation error-bound record is incomplete")
    finite_number(
        bounds.get("gbskernels_dd"),
        "matched GBSKernels DD error bound",
        nonnegative=True,
    )
    if bounds.get("walrus") is not None or bounds.get("piquasso") is not None:
        raise SystemExit("recursive baselines unexpectedly reported error bounds")
    pairs = row.get("pairwise")
    if not isinstance(pairs, list) or len(pairs) != 3:
        raise SystemExit("matched baseline pairwise record is incomplete")
    observed_pairs = []
    for pair in pairs:
        left = pair.get("left") if isinstance(pair, dict) else None
        right = pair.get("right") if isinstance(pair, dict) else None
        pair_key = frozenset((left, right))
        if left not in engines or right not in engines or left == right:
            raise SystemExit("matched baseline pairwise engine names are invalid")
        observed_pairs.append(pair_key)
        absolute = finite_number(
            pair.get("absolute_difference"),
            f"{matrix_id} pairwise absolute difference",
            nonnegative=True,
        )
        relative = finite_number(
            pair.get("relative_difference"),
            f"{matrix_id} pairwise relative difference",
            nonnegative=True,
        )
        expected_absolute = abs(values[left] - values[right])
        expected_relative = expected_absolute / max(
            abs(values[left]), abs(values[right]), 1e-300
        )
        if absolute != expected_absolute or relative != expected_relative:
            raise SystemExit(f"matched baseline pairwise difference is invalid: {matrix_id}")
        within = pair.get("within_tolerance")
        if not isinstance(within, bool):
            raise SystemExit(f"matched baseline tolerance flag is not Boolean: {matrix_id}")
        expected_within = bool(
            expected_absolute
            <= parameters["agreement_atol"]
            + parameters["agreement_rtol"]
            * max(abs(values[left]), abs(values[right]))
        )
        if within is not expected_within:
            raise SystemExit(f"matched baseline tolerance flag is invalid: {matrix_id}")
    if len(set(observed_pairs)) != 3 or set(observed_pairs) != expected_pairs:
        raise SystemExit(f"matched baseline engine-pair set is invalid: {matrix_id}")
    agreement_by_id[matrix_id] = row
if set(agreement_by_id) != set(inputs_by_id):
    raise SystemExit("matched baseline agreement rows do not cover the frozen inputs")

oracle = baseline.get("independent_arb_oracle")
if not isinstance(oracle, dict) or oracle.get("enabled") is not True:
    raise SystemExit("matched baseline independent Arb oracle is not enabled")
expected_oracle_settings = {
    "max_modes": 20,
    "target_bits": 80,
    "max_precision_bits": 2048,
}
for key, expected in expected_oracle_settings.items():
    if oracle.get(key) != expected:
        raise SystemExit(
            f"matched baseline Arb oracle {key}={oracle.get(key)!r}, expected {expected!r}"
        )
expected_oracle_summary = {
    "case_count": 15,
    "reported_bounds_checked": 15,
    "reported_bounds_containing_reference": 15,
    "reported_bounds_by_engine": {
        "gbskernels_dd": {"checked": 15, "containing_reference": 15},
        "walrus": {"checked": 0, "containing_reference": 0},
        "piquasso": {"checked": 0, "containing_reference": 0},
    },
}
oracle_summary = oracle.get("summary", {})
if not isinstance(oracle_summary, dict) or set(oracle_summary) != set(expected_oracle_summary):
    raise SystemExit("matched baseline Arb oracle summary fields are invalid")
for key, expected in expected_oracle_summary.items():
    if oracle_summary.get(key) != expected:
        raise SystemExit(
            f"matched baseline Arb oracle {key}={oracle_summary.get(key)!r}, "
            f"expected {expected!r}"
        )
oracle_rows = oracle.get("rows")
if not isinstance(oracle_rows, list) or len(oracle_rows) != 15:
    raise SystemExit("matched baseline Arb oracle row count is incomplete")
oracle_ids = set()
for row in oracle_rows:
    matrix_id = row.get("matrix_id") if isinstance(row, dict) else None
    if matrix_id not in inputs_by_id or matrix_id in oracle_ids:
        raise SystemExit("matched baseline Arb oracle identifiers are invalid")
    oracle_ids.add(matrix_id)
    input_row = inputs_by_id[matrix_id]
    if (
        row.get("modes") != input_row.get("modes")
        or row.get("matrix_sha256") != input_row.get("sha256")
    ):
        raise SystemExit(f"matched baseline Arb oracle input binding mismatch: {matrix_id}")
    finite_number(
        row.get("reference_seconds"),
        f"{matrix_id} Arb oracle reference seconds",
        nonnegative=True,
    )
    interval = row.get("arb_interval")
    modes = input_row.get("modes")
    if (
        not isinstance(interval, dict)
        or interval.get("schema") != "gbskernels.torontonian-arb-interval.v1"
        or interval.get("n_modes") != modes
        or interval.get("subset_count") != 1 << modes
        or interval.get("method") != "dense-subset-determinants"
        or interval.get("target_bits") != 80
    ):
        raise SystemExit(f"matched baseline Arb interval metadata is invalid: {matrix_id}")
    precision_bits = interval.get("precision_bits")
    if (
        isinstance(precision_bits, bool)
        or not isinstance(precision_bits, int)
        or not 128 <= precision_bits <= 2048
    ):
        raise SystemExit(f"matched baseline Arb interval precision is invalid: {matrix_id}")
    lower = dyadic_fraction(interval.get("lower"), f"{matrix_id} Arb lower")
    upper = dyadic_fraction(interval.get("upper"), f"{matrix_id} Arb upper")
    determinant_lower = dyadic_fraction(
        interval.get("minimum_determinant_lower"),
        f"{matrix_id} Arb minimum determinant lower",
    )
    if lower > upper or determinant_lower <= 0:
        raise SystemExit(f"matched baseline Arb interval is invalid: {matrix_id}")
    oracle_engines = row.get("engines")
    if not isinstance(oracle_engines, dict) or set(oracle_engines) != engines:
        raise SystemExit(f"matched baseline Arb engine records are incomplete: {matrix_id}")
    agreement = agreement_by_id[matrix_id]
    for name in engines:
        engine_oracle = oracle_engines[name]
        if not isinstance(engine_oracle, dict):
            raise SystemExit(f"matched baseline Arb engine record is invalid: {matrix_id}")
        center = complex_record(
            agreement["values"][name], f"{matrix_id}.{name}"
        )
        if center.imag != 0.0:
            raise SystemExit(f"matched baseline Arb engine center is not real: {matrix_id}")
        center_exact = Fraction.from_float(center.real)
        expected_error_lower = max(
            lower - center_exact,
            center_exact - upper,
            Fraction(0),
        )
        expected_error_upper = max(
            abs(center_exact - lower), abs(center_exact - upper)
        )
        if (
            rational_fraction(
                engine_oracle.get("center_error_lower"),
                f"{matrix_id}.{name} center-error lower",
            )
            != expected_error_lower
            or rational_fraction(
                engine_oracle.get("center_error_upper"),
                f"{matrix_id}.{name} center-error upper",
            )
            != expected_error_upper
        ):
            raise SystemExit(f"matched baseline Arb center error is invalid: {matrix_id}")
        contains = engine_oracle.get("reported_bound_contains_reference")
        if name == "gbskernels_dd":
            bound = finite_number(
                agreement["reported_abs_error_bounds"][name],
                f"{matrix_id} GBSKernels DD error bound",
                nonnegative=True,
            )
            radius = Fraction.from_float(bound)
            exact_contains = bool(
                center_exact - radius <= lower and upper <= center_exact + radius
            )
            if contains is not True or not exact_contains:
                raise SystemExit(
                    f"matched GBSKernels DD bound does not contain Arb: {matrix_id}"
                )
        elif contains is not None:
            raise SystemExit(
                f"matched recursive baseline unexpectedly reports Arb containment: {matrix_id}"
            )
if oracle_ids != set(inputs_by_id):
    raise SystemExit("matched baseline Arb rows do not cover the frozen inputs")

retained_extensions = sorted((root / "binary").glob("gbskernels_ext*.so"))
if len(retained_extensions) != 1:
    raise SystemExit(
        f"expected one retained compiled extension, found {len(retained_extensions)}"
    )
extension_records = build.get("build", {}).get("artifacts", {}).get("extensions", [])
if len(extension_records) != 1:
    raise SystemExit("build provenance does not contain exactly one extension record")
if extension_records[0].get("sha256") != sha256(retained_extensions[0]):
    raise SystemExit("retained extension hash disagrees with build provenance")

gate_logs = sorted((root / "gates").glob("*.log"))
if len(gate_logs) != 24:
    raise SystemExit(f"expected 24 gate logs, found {len(gate_logs)}")
for path in gate_logs:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[-1] != "PASS":
        raise SystemExit(f"gate evidence does not end in PASS: {path}")

files = []
for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
    if path == output or path.is_dir():
        continue
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"unsupported evidence entry: {path}")
    if path.suffix == ".json":
        load_strict(path)
    files.append({
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    })
if not files:
    raise SystemExit("evidence inventory is empty")
required_paths = {
    "build_provenance.json",
    "device/elf-list.txt",
    "device/embedded.sass",
    "device/extract-ptx.log",
    "device/nvidia-smi.txt",
    "device/ptx-list.txt",
    "environment/dpkg-query.txt",
    "environment/pip-freeze.txt",
    "environment/uv-lock-constraints.txt",
    "environment/uv-lock-sdists.txt",
    "science/arb_enclosure_campaign.json",
    "science/arb_enclosure_matrices.npz",
    "science/torontonian_matched_baselines.json",
    retained_extensions[0].relative_to(root).as_posix(),
}
inventoried_paths = {item["path"] for item in files}
missing_paths = sorted(required_paths - inventoried_paths)
if missing_paths:
    raise SystemExit(f"final evidence inventory is incomplete: {missing_paths}")

payload = {
    "schema": "gbskernels.publication-evidence.v1",
    "status": "pass",
    "source_tree_sha256": tree_sha256,
    "source_archive_sha256": archive_sha256,
    "build_manifest_sha256": build_sha256,
    "git_commit": commit,
    "git_tree": git_tree,
    "container_digest": container,
    "required_gpu_gates": 24,
    "arb_campaign_cases": 320,
    "matched_baseline_engines": sorted(engines),
    "matched_baseline_modes": expected_parameters["modes"],
    "files": files,
}
canonical = (json.dumps(
    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
) + "\n").encode("ascii")
payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
with output.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    handle.write("\n")
print(payload["manifest_sha256"])
PY

say "DONE"
printf 'Evidence root: %s\n' "$EVIDENCE_ROOT"
printf 'Build manifest: %s\n' "$GBS_BUILD_MANIFEST_SHA256"
printf 'Source tree: %s\n' "$SOURCE_TREE_SHA256"
