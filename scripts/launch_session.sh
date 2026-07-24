#!/usr/bin/env bash
# launch_session.sh -- drive a full rented-GPU session from the host in ONE command,
# the moment access is given. Captures provenance, pushes the repo, runs
# gpu_session.sh on the box, copies results back.
#
#   bash scripts/launch_session.sh -p <port> <user>@<host> [arch] [container-digest] [mode]
#
#   <arch>              optional: 80 (A100), 89 (4090), 90 (H100). Empty -> auto-detect.
#   [container-digest]  optional: the pinned image digest -> recorded in every artifact.
#   [mode]              optional: full (default) | jiuzhang (Jiuzhang regeneration +
#                       Gate C only) | campaign (stage-1 campaign) | validate
#                       (Item-1 DD-fix validation: adversarial enclosure + cost
#                       curve, no regeneration). jiuzhang/campaign/validate also
#                       push the small data payload. confirmatory is historical
#                       audit-only; confirmatory-v2 requires GBS_CONFIRM_V2_ARGS.
#
# Safety: the rsync excludes local-only files (private/, CLAUDE.local.md, local
# settings) and .git unconditionally -- nothing untracked-by-convention and no
# history ever leaves the host. Idempotent; safe to re-run. The box's
# gpu_session.sh self-bootstraps its Python deps.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PORT=""
if [ "${1:-}" = "-p" ]; then PORT="$2"; shift 2; fi
TARGET="${1:-}"; ARCH="${2:-}"; DIGEST="${3:-}"; MODE="${4:-full}"
[ -n "$TARGET" ] || { echo "usage: $0 -p <port> <user>@<host> [arch] [digest] [mode]" >&2; exit 2; }
if [ -n "$ARCH" ] && [[ ! "$ARCH" =~ ^[0-9]+$ ]]; then
  echo "ABORT: arch must be a numeric CUDA capability such as 89" >&2
  exit 2
fi
case "$MODE" in
  full|jiuzhang|campaign|validate|confirmatory|confirmatory-v2) ;;
  *) echo "ABORT: unsupported session mode: $MODE" >&2; exit 2 ;;
esac
if [ "$MODE" = "confirmatory-v2" ] && [[ "${GBS_CONFIRM_V2_ARGS:-}" == *"'"* || "${GBS_CONFIRM_V2_ARGS:-}" == *$'\n'* ]]; then
  echo "ABORT: GBS_CONFIRM_V2_ARGS may not contain quotes or newlines" >&2
  exit 2
fi
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ConnectTimeout=30)
[ -n "$PORT" ] && SSH+=(-p "$PORT")
RSH="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15${PORT:+ -p $PORT}"

echo "=== provenance ==="
if [ "$MODE" = "validate" ]; then
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || { echo "ABORT: validate mode must run from a Git worktree" >&2; exit 2; }
  if ! git diff --quiet HEAD -- || ! git diff --cached --quiet; then
    echo "ABORT: validate mode requires tracked files to match HEAD exactly" >&2
    exit 2
  fi
  if [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "ABORT: validate mode requires no untracked, non-ignored files" >&2
    git ls-files --others --exclude-standard >&2
    exit 2
  fi
fi
git rev-parse HEAD > COMMIT_SHA && echo "commit: $(cat COMMIT_SHA)"
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
if [ -n "$DIGEST" ] && { [[ "$DIGEST" == *"'"* || "$DIGEST" == *$'\n'* || "$DIGEST" == *" "* ]] \
     || [[ "$DIGEST" != *@sha256:* ]] \
     || [[ ! "${DIGEST##*@sha256:}" =~ ^[0-9a-fA-F]{64}$ ]]; }; then
  echo "ABORT: container digest must be image@sha256:<64-hex>" >&2
  exit 2
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
  --exclude 'private/' --exclude 'CLAUDE.local.md' --exclude '.claude/' \
  --exclude 'data/' \
  ./ "${TARGET}:~/GBSKernels/" || { echo "rsync up failed" >&2; exit 1; }

if [ "$MODE" = "jiuzhang" ] || [ "$MODE" = "campaign" ] || [ "$MODE" = "validate" ] || [ "$MODE" = "confirmatory" ] || [ "$MODE" = "confirmatory-v2" ]; then
  echo "=== push jiuzhang + q7 data payload (small; data/ is excluded above) ==="
  PAY="$(mktemp -d)/payload"
  mkdir -p "$PAY/data/jiuzhang1" "$PAY/data/q7_1076_zenodo/pattern_probs/patterns_exp"
  # Validate mode also runs Gate C on the retained event band. Historical and
  # v2 campaign modes receive their event patterns through their manifests.
if [ "$MODE" = "validate" ]; then
  J1_FILES=(T_full.npy events_band13_32.npy "squeezing parameters.txt")
  elif [ "$MODE" = "confirmatory" ] || [ "$MODE" = "confirmatory-v2" ]; then
    J1_FILES=(T_full.npy "squeezing parameters.txt")
  else
    J1_FILES=(T_full.npy empirical_click_rates.npy events_ge40.npy events_band13_32.npy "squeezing parameters.txt")
  fi
  for f in "${J1_FILES[@]}"; do
    cp "data/jiuzhang1/$f" "$PAY/data/jiuzhang1/" || { echo "ABORT: payload file missing: $f" >&2; exit 1; }
  done
  if [ "$MODE" = "campaign" ]; then
    cp data/jiuzhang1/campaign_events.npz "$PAY/data/jiuzhang1/" \
      || { echo "ABORT: campaign_events.npz missing (run decode_events.py)" >&2; exit 1; }
  fi
  if [ "$MODE" = "validate" ]; then
    mkdir -p "$PAY/data/q7_1076_zenodo/click_probs"
    cp data/q7_1076_zenodo/click_probs/click_probs_squeezed_0.npy \
       data/q7_1076_zenodo/click_probs/click_probs_squashed_0.npy \
       "$PAY/data/q7_1076_zenodo/click_probs/" \
      || { echo "ABORT: validate click-probability payload missing" >&2; exit 1; }
  else
    cp -R data/q7_1076_zenodo/sq_parameters data/q7_1076_zenodo/transfer_matrices \
          data/q7_1076_zenodo/click_probs data/q7_1076_zenodo/covariance_matrices \
          "$PAY/data/q7_1076_zenodo/" || { echo "ABORT: q7 zenodo payload missing" >&2; exit 1; }
    cp data/q7_1076_zenodo/pattern_probs/probs_sqz_0_clicks_*.npy \
       data/q7_1076_zenodo/pattern_probs/probs_sqs_0_clicks_*.npy \
       "$PAY/data/q7_1076_zenodo/pattern_probs/"
    cp data/q7_1076_zenodo/pattern_probs/patterns_exp/samples_0_clicks_*.npy \
       "$PAY/data/q7_1076_zenodo/pattern_probs/patterns_exp/"
  fi
  rsync -az -e "$RSH" "$PAY/data/" "${TARGET}:~/GBSKernels/data/" \
    || { echo "ABORT: payload rsync failed" >&2; exit 1; }
  rm -rf "$(dirname "$PAY")"
fi

echo "=== safety: confirm no local-only files on the box ==="
"${SSH[@]}" "$TARGET" 'cd ~/GBSKernels && (find . -name .git -prune -o \( -name "private" -o -name "CLAUDE.local.md" \) -print | grep . && echo "LEAK" || echo "clean")' \
  | grep -q '^clean$' || { echo "ABORT: local-only file found on box" >&2; exit 1; }

echo "=== run session (arch: ${ARCH:-auto}; nohup on the box) ==="
# The session runs DETACHED on the box (nohup): a killed local driver no longer
# SIGPIPE-kills the remote mid-bench (it did, twice -- artifacts survived but the
# tail of the run was lost). This ssh merely streams the log and polls the
# process; if it dies, re-running this block resumes streaming harmlessly.
# GBS_CONFIRM_ARGS (if set on the host) is forwarded so a confirmatory box can run
# a specific band-slice, e.g. GBS_CONFIRM_ARGS="--bands 30 --slice 150:300 --tag b4".
"${SSH[@]}" "$TARGET" "cd ~/GBSKernels && rm -f results/_session.log && nohup env GBS_CONFIRM_ARGS='${GBS_CONFIRM_ARGS:-}' GBS_CONFIRM_V2_ARGS='${GBS_CONFIRM_V2_ARGS:-}' GBS_ALLOW_LEGACY_CONFIRMATORY='${GBS_ALLOW_LEGACY_CONFIRMATORY:-0}' GBS_COMMIT='$(cat COMMIT_SHA)' GBS_CONTAINER_DIGEST='${DIGEST}' bash scripts/gpu_session.sh '${ARCH}' '${MODE}' > results/_session.log 2>&1 < /dev/null & echo \"session pid \$!\""

# GBS_NO_STREAM=1: fire-and-forget. The campaign is nohup'd on the box; return NOW
# so the caller can launch the next box atomically. A separate babysitter pulls
# checkpoints, aggregates, and destroys. (Default: stream the log + pull at the end.)
if [ -n "${GBS_NO_STREAM:-}" ]; then
  echo "=== fire-and-forget: campaign nohup'd on the box; no local stream/pull ==="
  exit 0
fi
"${SSH[@]}" "$TARGET" 'cd ~/GBSKernels && off=0; while pgrep -f "[g]pu_session.sh" >/dev/null 2>&1; do sz=$(wc -c < results/_session.log 2>/dev/null || echo 0); if [ "$sz" -gt "$off" ]; then tail -c +$((off+1)) results/_session.log; off=$sz; fi; sleep 10; done; sz=$(wc -c < results/_session.log); [ "$sz" -gt "$off" ] && tail -c +$((off+1)) results/_session.log; exit 0'

echo "=== copy results back (append-only) ==="
rsync -az -e "$RSH" "${TARGET}:~/GBSKernels/results/" ./results/ || echo "[warn] results rsync failed; retry manually"
echo "=== DONE -- review results/ ; TERMINATE the instance ==="
