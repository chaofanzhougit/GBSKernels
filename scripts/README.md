# scripts/ — GPU-session orchestration

Tooling to run the on-device validation and benchmarks on a rented CUDA host,
CPU-first by construction.

- **`gpu_session.sh`** — the on-box runner. In one command it runs the CPU
  pre-flight, builds `core/` with `nvcc`, and runs every mandatory differential
  gate before publishable throughput timing. Compiler and profiler diagnostics
  collected immediately after the build remain provisional until those gates
  pass. In ordinary benchmark modes, optional evidence harnesses may warn when
  unavailable. In release `validate` mode, the nanobind build/smoke, adversarial
  enclosure with physical-family coverage, and Jiuzhang Gate C probe are all
  hard gates; success writes a semantic manifest with hashes for all evidence.
- **`launch_session.sh`** — drives a session from a workstation: it captures the
  full commit and container digest, copies the working tree to the host
  (excluding local-only files), runs `gpu_session.sh` there, and copies
  `results/` back. Release `validate` mode refuses tracked changes, staged
  changes, or untracked non-ignored files before upload.
- **`session.py`** — an experimental local manifest wrapper around pluggable
  provision/terminate commands. Its safety contract is active (`--dry-run`,
  `--confirm`, CPU gate, cost caps), but manifest cells execute in the local
  checkout; use `launch_session.sh` for the implemented remote-host workflow.

```bash
bash scripts/gpu_session.sh                         # on the CUDA host
bash scripts/launch_session.sh -p PORT USER@HOST 89 # from a workstation
bash scripts/launch_session.sh -p PORT USER@HOST 89 IMAGE@sha256:DIGEST validate
```

Replace `PORT` and `USER@HOST` with the SSH endpoint for the rented host.

## Publication-evidence workflow (prospective)

The following tools define a new, source-to-binary-bound publication workflow.
They do not replace or retroactively strengthen the v0.2.1 evidence above, and
no GPU run or passing publication artifact from this workflow is claimed in the
repository at present.

- **`capture_build_provenance.py`** creates a strict manifest for an immutable
  archive extraction. It verifies the release-archive and source-tree hashes and
  records the full Git commit/tree, digest-pinned container, normalized compiler
  commands and caches, compiler/tool queries, the embedded-cubin inventory and
  SASS dump, available PTX, gate binaries, wheel, and the exact compiled
  extension. It re-hashes the source and
  archive after capture and refuses outputs inside the source tree or an existing
  output path.
- **`publication_gpu_session.sh`** runs on an already-provisioned CUDA host. It
  requires a `.git`-free archive extraction, the separate release archive, their
  expected hashes, Git and container identities, and a new external output
  directory. It derives exact registry constraints from the archive's `uv.lock`,
  installs the publication-only direct pins, builds the core and binding out of
  source with auditable floating-point flags, builds and installs the wheel from
  a second archive extraction, and verifies that Python loaded the recorded
  extension. The 24 device gates run before either science campaign.
- **`render_vast_publication_adapter.py`** bridges the Vast runner's deliberately
  small three-positional remote interface to the on-box driver's explicit flags.
  The renderer reproduces the supplied commit with `git archive`, verifies that
  its contents match the release archive, and derives the canonical source-tree
  hash directly from the safely inspected archive. An optional
  `--source-tree-sha256` is only an additional caller-supplied cross-check. The
  generated, create-only adapter binds the archive, tree, and driver hashes, Git
  objects, and container digest; verifies the uploaded archive; extracts it into
  a fresh directory; invokes the archive's driver; and packages the evidence and
  bootstrap/session logs for retrieval.
- **`vast_publication.py`** uses the official `vastai` CLI. It searches only
  on-demand offers and revalidates that the selected offer is verified,
  rentable, not already rented, and exactly one GPU, while enforcing the GPU,
  CUDA, reliability, disk, direct-port, hourly-cost, total-cost, and lifetime
  constraints. The image must be pinned by digest. A live run requires
  `--confirm-spend`; `--dry-run` performs local validation and writes a plan
  receipt without provisioning. Preflight requires the adapter's archive and
  container declarations to equal the uploaded archive and launched image.
  Before any provider command, the three upload inputs are copied into verified
  private snapshots. Upload, SSH readiness, and execution are bounded; failed
  uploads use partial remote names and are retried, then all three SHA-256 values
  are verified before atomic, idempotent promotion to their execution names. A
  separate reserved window retries retrieval before the teardown reserve begins.
  Outputs and receipts are create-only, and early remote failures still return a
  log bundle. An ambiguous creation is polled for a bounded interval and
  recovered only from one unique run-label match; failure to recover an ID is
  reported as a dedicated lifecycle error. Teardown targets the exact instance
  ID non-interactively, with delayed,
  recorded retries and a strict active-instance-list absence check on normal,
  failure, and handled-signal paths. Unconfirmed teardown is always the primary
  operator-facing error, while the preceding execution error remains in the
  receipt.

The registered on-box profile is fail-closed. It is configured to accept only a
24-gate pass, 320 Arb-contained enclosures, and a complete schema-v4 matched
artifact for GBSKernels on GPU plus The Walrus and Piquasso on CPU at
`4,8,12,16,20` modes. The matched corpus uses the real quadrature-basis loss
construction `O_x = I - ((cov + I)/2)^-1`. Each of its exact 15 frozen binary64
matrices receives an independent dense python-flint/Arb interval outside the
timed regions, and every GBSKernels DD-reported radius must contain the
corresponding interval. The Walrus and Piquasso fp64 values have no claimed
bounds. Pairwise tolerance flags are descriptive only: they are neither
acceptance criteria nor filters for timing rows. The heterogeneous GPU-DD versus
CPU-fp64 measurements do not isolate hardware, algorithm, or precision and
therefore cannot establish a causal speedup. This acceptance rule describes
what a future run must produce, not a result already obtained. The eight Arb
cases at `k = 25,...,32` are factorized quarter-identity matrices, not a general
dense large-`k` sample.

Generate the adapter only after the release archive, Git objects, and container
digest are final. The renderer derives the source-tree hash; pass
`--source-tree-sha256` only when independently cross-checking it. All named
outputs below must be new paths:

```bash
python scripts/render_vast_publication_adapter.py \
  --archive RELEASE_ARCHIVE.tar.gz \
  --git-commit FULL_GIT_COMMIT \
  --git-tree FULL_GIT_TREE \
  --container-digest IMAGE@sha256:DIGEST \
  --repository . \
  --output /new/path/publication-vast-adapter.sh

python scripts/vast_publication.py \
  --archive RELEASE_ARCHIVE.tar.gz \
  --checksum RELEASE_ARCHIVE.tar.gz.sha256 \
  --session-script /new/path/publication-vast-adapter.sh \
  --image IMAGE@sha256:DIGEST \
  --output /new/path/planned-output.tar.gz \
  --receipt /new/path/dry-run-receipt.json \
  --dry-run
```

For a paid run, choose fresh output and receipt paths, retain the same immutable
inputs, add `--confirm-spend`, and set explicit `--max-hourly-usd`,
`--max-total-usd`, and `--max-instance-seconds` limits. Review
[`envs/publication-requirements.txt`](../envs/publication-requirements.txt) and
[`envs/README.md`](../envs/README.md) before selecting the image. `--help` on
each tool is the authoritative option list.
