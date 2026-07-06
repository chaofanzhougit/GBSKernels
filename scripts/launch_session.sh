#!/usr/bin/env bash
# launch_session.sh -- drive a full rented-GPU session from the host in ONE command,
# the moment access is given. Captures provenance, pushes the repo, runs
# gpu_session.sh on the box, copies results back.
#
#   bash scripts/launch_session.sh -p <port> <user>@<host> [arch] [container-digest]
#
#   <arch>              optional: 80 (A100), 89 (4090), 90 (H100). Empty -> auto-detect.
#   [container-digest]  optional: the pinned image digest -> recorded in every artifact.
#
# Safety: the rsync excludes local-only files (private/, local
# settings) and .git unconditionally -- nothing untracked-by-convention and no
# history ever leaves the host. Idempotent; safe to re-run. The box's
# gpu_session.sh self-bootstraps its Python deps.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PORT=""
if [ "${1:-}" = "-p" ]; then PORT="$2"; shift 2; fi
TARGET="${1:-}"; ARCH="${2:-}"; DIGEST="${3:-}"
[ -n "$TARGET" ] || { echo "usage: $0 -p <port> <user>@<host> [arch] [digest]" >&2; exit 2; }
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ConnectTimeout=30)
[ -n "$PORT" ] && SSH+=(-p "$PORT")
RSH="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15${PORT:+ -p $PORT}"

echo "=== provenance ==="
git rev-parse --short HEAD > COMMIT_SHA && echo "commit: $(cat COMMIT_SHA)"
# Container digest is MANDATORY for an OFFICIAL run (frozen-experiment reproducibility -- every
# artifact must pin the exact image). If not passed as arg 4, try to resolve it from the image
# tag via docker; else ABORT (set ALLOW_NULL_DIGEST=1 only for a non-official dry run).
IMG="${GBS_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
if [ -z "$DIGEST" ] && command -v docker >/dev/null 2>&1; then
  D="$(docker buildx imagetools inspect "$IMG" --format '{{.Manifest.Digest}}' 2>/dev/null)"
  [ -n "$D" ] && { DIGEST="${IMG}@${D}"; echo "resolved digest (docker) for $IMG: $DIGEST"; }
fi
if [ -z "$DIGEST" ] && command -v curl >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  # No docker -> resolve the tag's digest straight from the Docker Hub registry API (curl only).
  REPO="${IMG%%:*}"; TAG="${IMG##*:}"
  TOK="$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${REPO}:pull" \
        | python3 -c 'import sys,json;print(json.load(sys.stdin).get("token",""))' 2>/dev/null)"
  if [ -n "$TOK" ]; then
    D="$(curl -sI -H "Authorization: Bearer $TOK" \
         -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json" \
         "https://registry-1.docker.io/v2/${REPO}/manifests/${TAG}" \
         | tr -d '\r' | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}')"
    [ -n "$D" ] && { DIGEST="${IMG}@${D}"; echo "resolved digest (registry) for $IMG: $DIGEST"; }
  fi
fi
if [ -n "$DIGEST" ]; then
  echo "$DIGEST" > CONTAINER_DIGEST; echo "digest: $DIGEST"
elif [ "${ALLOW_NULL_DIGEST:-0}" = "1" ]; then
  rm -f CONTAINER_DIGEST
  echo "[warn] ALLOW_NULL_DIGEST=1 -- artifacts record container_digest:null (NOT an official run)"
else
  echo "ABORT: no container digest -- an official run must pin the image. Pass it as arg 4" >&2
  echo "  (image@sha256:...), from the vast.ai instance page or:" >&2
  echo "  docker buildx imagetools inspect $IMG --format '{{.Manifest.Digest}}'" >&2
  echo "  (or set ALLOW_NULL_DIGEST=1 for a non-official dry run)." >&2
  exit 2
fi

echo "=== push repo (local-only files + .git ALWAYS excluded) ==="
rsync -az -e "$RSH" \
  --exclude .venv --exclude .git --exclude '__pycache__' \
  --exclude 'core/build' --exclude 'bindings/build*' --exclude '.pytest_cache' \
  --exclude '*.egg-info' --exclude '.hypothesis' --exclude 'dist/' \
  --exclude 'private/' \
  --exclude 'data/' \
  ./ "${TARGET}:~/GBSKernels/" || { echo "rsync up failed" >&2; exit 1; }

echo "=== safety: confirm no local-only files on the box ==="
"${SSH[@]}" "$TARGET" 'cd ~/GBSKernels && (find . -name .git -prune -o \( -name "private" \) -print | grep . && echo "LEAK" || echo "clean")' \
  | grep -q '^clean$' || { echo "ABORT: local-only file found on box" >&2; exit 1; }

echo "=== run session (arch: ${ARCH:-auto}; nohup on the box) ==="
# The session runs DETACHED on the box (nohup): a killed local driver no longer
# SIGPIPE-kills the remote mid-bench (it did, twice -- artifacts survived but the
# tail of the run was lost). This ssh merely streams the log and polls the
# process; if it dies, re-running this block resumes streaming harmlessly.
"${SSH[@]}" "$TARGET" "cd ~/GBSKernels && rm -f results/_session.log && nohup bash scripts/gpu_session.sh ${ARCH} > results/_session.log 2>&1 < /dev/null & echo \"session pid \$!\""
"${SSH[@]}" "$TARGET" 'cd ~/GBSKernels && off=0; while pgrep -f "[g]pu_session.sh" >/dev/null 2>&1; do sz=$(wc -c < results/_session.log 2>/dev/null || echo 0); if [ "$sz" -gt "$off" ]; then tail -c +$((off+1)) results/_session.log; off=$sz; fi; sleep 10; done; sz=$(wc -c < results/_session.log); [ "$sz" -gt "$off" ] && tail -c +$((off+1)) results/_session.log; exit 0'

echo "=== copy results back (append-only) ==="
rsync -az -e "$RSH" "${TARGET}:~/GBSKernels/results/" ./results/ || echo "[warn] results rsync failed; retry manually"
echo "=== DONE -- review results/ ; TERMINATE the instance ==="
