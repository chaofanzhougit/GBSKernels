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
if baseline.get("schema_version") != 3:
    raise SystemExit("matched baseline schema version is not the registered version")
engines = {row.get("name") for row in baseline.get("engines", [])}
if engines != {"gbskernels_dd", "walrus", "piquasso"}:
    raise SystemExit(f"matched baseline engine set is incomplete: {engines}")
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
}
for key, expected in expected_parameters.items():
    if parameters.get(key) != expected:
        raise SystemExit(
            f"matched baseline {key}={parameters.get(key)!r}, expected {expected!r}"
        )
engine_rows = {row.get("name"): row for row in baseline.get("engines", [])}
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
if any(
    len(row.get("pairwise", [])) != 3
    or not all(pair.get("within_tolerance") is True for pair in row["pairwise"])
    for row in agreement_rows
):
    raise SystemExit("matched implementations did not agree within the registered tolerance")
for row in agreement_rows:
    bounds = row.get("reported_abs_error_bounds")
    if not isinstance(bounds, dict) or set(bounds) != engines:
        raise SystemExit("matched implementation error-bound record is incomplete")
    dd_bound = bounds.get("gbskernels_dd")
    if (
        isinstance(dd_bound, bool)
        or not isinstance(dd_bound, (int, float))
        or not math.isfinite(float(dd_bound))
        or float(dd_bound) < 0.0
    ):
        raise SystemExit("matched GBSKernels DD error bound is invalid")
    if bounds.get("walrus") is not None or bounds.get("piquasso") is not None:
        raise SystemExit("recursive baselines unexpectedly reported error bounds")

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
