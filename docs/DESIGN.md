# GBSKernels — Design & Verification Contract

> The project's standing design document: intent, constraints, architecture, the
> precision strategy, and the five-layer verification contract. Code comments and
> docstrings cite it as **"docs/DESIGN.md §N"** (or `docs/DESIGN.md §N`) — section numbers here
> are stable for that reason. Operational protocols live next to it:
> `docs/benchmark_protocol.md` (the frozen benchmark experiment) and
> `docs/device_resident_contract.md` (the device-resident workspace contract).

---

## 0. Mission in one paragraph

Build the first **GPU-native, batched, numerically-characterized** open-source
library of the matrix functions that govern photonic quantum sampling — the
**hafnian** and **loop hafnian** (Gaussian boson sampling probabilities), the
**permanent** (standard boson sampling), and the **torontonian**
(threshold-detector statistics). The canonical library, Xanadu's *The Walrus*, is
an excellent CPU/C++ implementation; no clean, well-tested, GPU-native equivalent
exists that delivers the one thing photonic sampling actually needs at scale —
**high throughput on large batches of these functions** — while honestly
characterizing the **accuracy-vs-speed tradeoff** that GPU floating point forces.
GBSKernels fills that gap, validated against independent combinatorial ground
truth (not just against The Walrus), with the precision boundary measured rather
than hand-waved.

---

## 1. Why this project exists

Photonic quantum computing — Xanadu's Borealis, Quandela's Ascella, the broader
Gaussian-boson-sampling (GBS) program — rests computationally on a handful of
#P-hard matrix functions. Computing GBS output probabilities and drawing GBS
samples both reduce to evaluating **many hafnians of submatrices**; threshold
detectors replace the hafnian with the **torontonian**; standard
(Aaronson–Arkhipov) boson sampling uses the **permanent**. These evaluations are
the classical bottleneck — for verifying hardware, benchmarking quantum-advantage
claims, simulating devices during design, and computing application kernels
(graph similarity, molecular vibronic spectra, dense-subgraph and
perfect-matching problems encoded into GBS). The reference library, *The
Walrus*, is CPU-bound C++ with a Python interface; the field's GPU story is
fragmentary (§3). Yet the workload — thousands of independent matrix-function
evaluations per sample run — is almost perfectly suited to GPU throughput. The
gap between "what the science needs" and "what's available on the hardware
everyone now has" is the opening.

---

## 2. Background & basics (the project's canonical reference)

*Written to be the self-contained primer for the repo; precise enough to
implement from.*

### 2.1 The four matrix functions

For an $n\times n$ matrix $A$:

- **Permanent** $\mathrm{perm}(A) = \sum_{\sigma\in S_n} \prod_{i=1}^{n} A_{i,\sigma(i)}$ — like the determinant but with all-plus signs, which is exactly what removes the cancellation that makes determinants easy; computing it is **#P-hard** (Valiant 1979). For a 0/1 matrix it counts perfect matchings of the corresponding bipartite graph.
- **Hafnian** $\mathrm{haf}(A) = \sum_{M\in \mathrm{PMP}(n)} \prod_{(i,j)\in M} A_{i,j}$ for symmetric $A$ of even size $n$, where $\mathrm{PMP}(n)$ is the set of perfect matchings of $n$ points. It counts (weighted) perfect matchings of a general graph; $\mathrm{haf}$ generalizes $\mathrm{perm}$ (the permanent is the hafnian of a bipartite-structured matrix). Also #P-hard.
- **Loop hafnian** $\mathrm{lhaf}(A)$ — the hafnian generalized to allow "loops" (single points matched to themselves), encoded on the diagonal of $A$. Needed for Gaussian states with **nonzero displacement** (finite mean field); reduces to the ordinary hafnian when the diagonal is zero.
- **Torontonian** $\mathrm{tor}(O) = \sum_{S\subseteq[n]} (-1)^{n-|S|} \big/ \sqrt{\det(\mathbb{I} - O_S)}$ — arises for **threshold** (click/no-click) detectors, where one sums hafnian-like contributions over all photon numbers; it collapses that infinite sum into a single function of size $n$. (Sign convention: $(-1)^{n-|S|}$, matching The Walrus and the implementation; the form $(-1)^{|S|}$ differs by $(-1)^n$ — wrong for odd $n$. Indices in xxpp ordering: mode $i$ owns rows/cols $i$ and $i+n$.)

### 2.2 Why photonics produces exactly these (the physics link)

A Gaussian state of $m$ optical modes is fully described by its covariance
matrix and mean. **Gaussian boson sampling** (Hamilton et al. 2017) feeds
squeezed light through a linear-optical interferometer and measures photon
number. The probability of detecting a given photon pattern $\bar n$ is
proportional to the **hafnian** of a submatrix of a kernel matrix $A$ built from
the state, with the submatrix selected by $\bar n$ (each detected photon picks
out rows/columns). Three refinements map onto the three other functions: nonzero
displacement → **loop** hafnian; threshold detectors → **torontonian**; and the
historically prior **standard boson sampling** (single photons, not squeezed
light; Aaronson–Arkhipov 2011) → **permanent**. So the four functions are not a
grab-bag — they are the four corners of photonic sampling (squeezed vs
single-photon) × (number-resolving vs threshold), plus the displacement
generalization.

### 2.3 Why this is the computational bottleneck, and why GPU/batching is the lever

Two facts set the entire engineering problem:

**(a) Each function is exponential, but the modern algorithms are
$O(\mathrm{poly}(n)\,2^{n/2})$ and parallel.** Naïvely the permanent sums $n!$
terms; Ryser/Glynn reduce it to a signed sum over $2^n$ subsets. Naïvely the
hafnian sums over $(n-1)!!$ matchings; the Björklund–Cygan–Pilipczuk
"power-trace" algorithm used by The Walrus reduces it to
$\mathrm{haf}(A)=\sum_{S\subseteq[n/2]}(-1)^{|S|} f\big((AX)_S\big)$ running in
$O(n^3 2^{n/2})$ in polynomial space, where $f$ extracts a power-trace
polynomial coefficient. The torontonian is a signed subset sum over
**determinants**; the permanent a signed subset sum over **row-sum products**.
**The unifying structure (the central design insight): all four are a signed sum
over $2^k$ subsets of a per-subset dense-linear-algebra kernel, followed by a
reduction.** Subset enumeration is embarrassingly parallel — Gray-code ordering
makes each subset an $O(1)$ or $O(n)$ update of the previous — which is
precisely a GPU's strength.

**(b) Sampling needs *thousands* of these, not one.** Drawing a single GBS
sample uses a chain of conditional probabilities, each proportional to a hafnian
of a growing submatrix; a sampling run produces many samples; verification and
applications sweep many configurations. The real workload is **a large batch of
independent medium-sized matrix-function evaluations.** The Walrus computes
these on CPU, one (coarsely-threaded) call at a time. A GPU that maps *one
evaluation per warp/block and the whole batch across the grid* is the natural
fit — and the regime where the throughput win is largest and most useful. This
batched-throughput framing (not "one giant hafnian") is the project's product
thesis.

---

## 3. Gap analysis — what exists, what's missing

| Tool | What it is | Strength | Gap GBSKernels fills |
|---|---|---|---|
| **The Walrus** (Xanadu) | Canonical C++/Python library; hafnian (power-trace $O(n^3 2^{n/2})$, default), loop hafnian, permanent, torontonian, GBS samplers; extended (long-double/`__float128`) precision | The correctness oracle and feature reference; battle-tested; JOSS-published | **CPU-bound**; coarse parallelism; no GPU; no batched-throughput path; one-evaluation-at-a-time API shape |
| **Piquasso / Piquasso Boost** | Photonic simulator with an HPC "Boost" layer; permanent via data-flow engines (FPGA/HPC), extended precision | Serious high-performance permanents; numerical-accuracy focus | Permanent-centric; FPGA/HPC-oriented, not commodity-GPU; not a clean batched-GPU kernel library for *all four* functions |
| **DeepQuantum** (recent) | PyTorch photonic platform; demonstrated GPU-batched hafnian/torontonian beating CPU incumbents at large batch/size | Proves the GPU-batch thesis is real and wins | One feature inside a large framework; not a standalone, dependency-grade kernel library; precision characterization not the focus |
| **Strawberry Fields / Perceval / MrMustard** | Photonic circuit simulators (Xanadu / Quandela) | High-level simulation; call the kernels above | Consumers of the kernels, not optimized kernel providers; depend on The Walrus |

**The missing quadrant:** a *focused, standalone, GPU-native, batched,
dependency-grade* library of all four functions, **with the
FP64-vs-extended-precision accuracy boundary measured and exposed**, validated
against *independent* ground truth rather than only against the incumbent. The
precision dimension is the subtlest gap: these are alternating signed sums with
**catastrophic cancellation**, which The Walrus handles with CPU extended
precision that **has no native GPU equivalent** — so a credible GPU library must
solve the precision problem (double-double arithmetic) *and quantify when it is
needed*. Nobody has published that characterization cleanly. That is both the
hardest part and the most novel contribution.

### Concrete gap list (each → a section)

G1 no commodity-GPU batched library for all four functions → §5–7 · G2 the GPU
precision/cancellation problem unsolved-in-the-open (no native quad on GPU) →
§6 · G3 no published accuracy-vs-throughput characterization across precisions →
§6, §9 · G4 validation in the field leans on a single oracle (The Walrus);
little independent combinatorial ground-truthing → §8 · G5 sampling-throughput
(the real workload) under-served by one-call-at-a-time APIs → §7 batching API.

---

## 4. Scope

**In scope (phases 1–3):** permanent (Ryser + Glynn/BB-FG), hafnian (power-trace
algorithm), loop hafnian, torontonian; CPU reference + CUDA implementations;
FP64 and double-double precision paths; batched evaluation API; Python bindings;
a validated GBS-probability/sampling example; accuracy-vs-throughput benchmarks.

**Out of scope (candidate later):** AMD/ROCm; multi-GPU; FPGA; the
multidimensional-Hermite-polynomial machinery and exotic specialized hafnians
(banded, sparse-Laplace, Bristolian) beyond a basic sparse path; full GBS
application suite (graph kernels, vibronic spectra) — the library *enables*
these but does not implement them; autodiff through the kernels (interesting,
parked).

**Non-goals:** not a circuit simulator (it's a kernel library others call); not
a re-implementation of The Walrus's full surface; not a claim to beat The Walrus
on single small evaluations (CPU often wins there — say so).

---

## 5. Algorithms & GPU parallelization strategy

The shared skeleton (signed subset sum → per-subset linear algebra → reduction)
is the common design, with shared `subset_engine.cuh` utilities. **Status
(honest, 2026-06):** the Gray-code *delta-walk* is realized in the **permanent**
(each subset an $O(1)$/$O(n)$ update of the previous); the **hafnian, loop
hafnian, and torontonian** currently enumerate masks **independently** (one
evaluation per thread). The four are not unified onto one Gray-code delta-walk: the measured reason is
that the per-subset work for haf/lhaf/tor is memory-bound (per-thread
local-memory footprint), so the permanent's split-subset / delta-walk strategy
does **not** transfer automatically (measured, not assumed). Two execution
regimes throughout: **single-large** (one evaluation,
the $2^k$ subset space parallelized across the whole GPU) and **batched** (many
independent evaluations, one per warp/block, batch across the grid). The batched
regime is the priority.

**Permanent — Ryser / Glynn (BB-FG).**
$\mathrm{perm}(A)=(-1)^n\sum_{S\subseteq[n]}(-1)^{|S|}\prod_{i=1}^{n}\big(\sum_{j\in S}A_{ij}\big)$
(Ryser); Glynn/BB-FG is the standard refinement halving the work via a symmetry
and Gray-code delta updates. GPU: enumerate the $2^{n-1}$ Gray-code subsets
across threads; each step is an $O(n)$ rank-update of the running row-sum
vector; per-subset cost $O(n)$; parallel signed reduction. Complexity
$O(n\,2^n)$; the cleanest first kernel and the easiest to validate (closed forms
exist). **Implemented first.**

**Hafnian — power-trace (Björklund–Cygan–Pilipczuk, The Walrus default).**
$\mathrm{haf}(A)=\sum_{S\subseteq[n/2]}(-1)^{|S|} f\big((AX)_S\big)$,
$X=\big[\begin{smallmatrix}0&I\\I&0\end{smallmatrix}\big]$, where $f(C)$ is the
coefficient of $\eta^{n/2}$ in
$\exp\!\big(\sum_k \tfrac{\mathrm{tr}(C^k)}{2k}\eta^k\big)$ — computed from the
**power traces** $\mathrm{tr}(C),\dots,\mathrm{tr}(C^{n/2})$ via Newton's
identities, $O(n^3)$ per subset (matrix powers / Schur form). GPU: subset
enumeration parallel across the grid; per-subset power-trace computation either
per-thread (small $n$), per-warp with cooperative reduction (medium $n$), or
per-block (large $n$). Complexity $O(n^3 2^{n/2})$. The hard, high-value kernel.

**Loop hafnian.** Same formula family with the diagonal carried through the
power-trace polynomial (the generating function gains a linear term in the loop
weights). Implemented as a generalization of the hafnian kernel sharing the same
subset engine; validated by the diagonal-zero → hafnian reduction.

**Torontonian.**
$\mathrm{tor}(O)=\sum_{S\subseteq[n]}(-1)^{n-|S|}\det(\mathbb{I}-O_S)^{-1/2}$
(xxpp ordering; sign matches The Walrus — see §2.1). GPU: subset enumeration
parallel; per-subset cost is a Cholesky/LU determinant of a principal submatrix,
$O(|S|^3)$ (or incremental via rank-1 Cholesky updates along the Gray code).
Complexity $O(n^3 2^{n})$-ish before incremental tricks. Reuses the subset
engine; the per-subset kernel is a determinant instead of a power-trace.

**Sampling layer (thin, on top).** The GBS conditional-probability chain
implemented as orchestrated batched hafnian/torontonian calls — this is where
batched throughput pays off and where end-to-end statistical validation (§8 L4)
attaches.

---

## 6. Numerical precision — the technical core (G2, G3)

These functions are **alternating signed sums whose terms can dwarf the
result** → catastrophic cancellation; accuracy degrades with $n$ and with matrix
conditioning. The Walrus mitigates with CPU extended precision (`long double` /
`__float128`). **GPUs have no native quad precision** — this is the crux that
makes a naïve FP64 GPU port quietly wrong for large/ill-conditioned inputs, and
solving it well is the project's most novel contribution.

**Strategy — three precision tiers, selectable, and *characterized*:**

1. **FP64 fast path.** Native, fast, correct for small-to-moderate $n$ and
   well-conditioned matrices. The throughput workhorse.
2. **Double-double (DD) path.** Each value represented as an unevaluated sum of
   two FP64 (hi+lo), ~31 decimal digits, via error-free transformations (TwoSum,
   and TwoProd using FMA). ~8–20× slower than FP64 but fully GPU-parallel and
   well-understood (Dekker/Knuth EFTs; Bailey QD; campary/GQD as references —
   reimplement the needed ops, don't depend heavily). Restores accuracy where
   FP64 cancels.
3. **Mixed/adaptive (`precision="auto"`).** FP64 with an a-posteriori
   cancellation indicator ($\kappa=\Sigma|\mathrm{terms}|/|\mathrm{result}|$)
   that triggers a high-precision rerun per evaluation — best throughput at a
   target accuracy. Implemented for all four functions on the CPU (FP64→mpmath)
   and GPU (`*_kappa` kernels → per-element FP64→DD rerun); $\kappa$ is a
   **calibrated heuristic, not an error certificate** (`bench.calibrate_auto`).

**The deliverable isn't just "it's accurate" — it's the measured boundary.**
Produce **accuracy-vs-throughput curves** (relative error vs $n$, conditioning,
and precision tier) against an independent extended-precision reference. Users
learn *exactly* when FP64 suffices and when DD is required. Reference truth for
these curves comes from an **independent mpmath implementation** of each
function at 50+ digits (slow, tiny sizes) — not from The Walrus — so the
precision characterization shares no code with the thing it might otherwise just
echo.

---

## 7. Architecture

```
GBSKernels/
├── docs/DESIGN.md               # this file (cited in code as "docs/DESIGN.md §N")
├── core/                        # CUDA/C++ : the engine + the kernels
│   ├── subset_engine.cuh        #   Gray-code subset enumeration (shared utilities)
│   ├── dd.cuh                   #   double-double arithmetic (EFTs: TwoSum/TwoProd/FMA)
│   ├── permanent.cu  hafnian.cu  loop_hafnian.cu  torontonian.cu   (+ *_dd.cu DD tiers)
│   ├── permanent_coop.cu / permanent_warp.cu      # cooperative / fused-shuffle variants
│   ├── sampler_gather.cu / sampler_draw.cu / sampler_session.cu    # on-device sampler chain
│   ├── host_api.cu              #   host-pointer wrappers + dispatch
│   ├── check_*.cu               #   GPU-vs-CPU differential gates (run on the device)
│   └── preflight/               #   host shim: compiles/runs the kernels on CPU
├── cpu_ref/                     # independent CPU reference impls (FP64 + DD twin)
├── highprec_ref/                # mpmath ground truth (50+ digits, tiny sizes)
├── bindings/                    # nanobind Python extension over the kernels
├── gbskernels/                  # ergonomic Python layer + batching API + Workspace
├── sampling/                    # GBS orchestration + conditional sampler (validation target)
├── tests/                       # the five-layer verification suite (§8)
├── bench/                       # accuracy-vs-throughput harness (§9)
├── envs/  scripts/              # pinned container(s); scripted rented-GPU sessions
└── results/                     # raw accuracy + throughput data (append-only)
```

**API shape (batched-first):** `perm(A)` / `haf(A)` /
`lhaf(A)` / `tor(O)` for single calls, but the headline surface is
`haf_batched(stack_of_matrices, precision=...)` returning a vector, mapping one
evaluation per thread/warp/block. Precision tier is an explicit argument;
default FP64 with a documented accuracy caveat and a one-flag DD upgrade. CPU
reference reachable through the same API for differential testing.
`gbskernels.Workspace` is the device-resident handle for iterative workloads
(`docs/device_resident_contract.md`).

**Stack:** CUDA C++ for `core/`; **nanobind** for bindings; Python 3.12 + `uv`
for the Python layer, packaging, tests, and harness. Build via CMake.

---

## 8. Verification & testing — the five-layer contract (the heart of "useful not cute")

A kernel library that might be silently wrong on the inputs that matter is
worthless, so verification *is* the product. Five layers, increasingly
independent of any single oracle.

**Layer 1 — Independent combinatorial ground truth (shares no code with The
Walrus).** The whole point of these functions is counting, so count by brute
force at small sizes: permanent of a 0/1 matrix = number of perfect matchings of
its bipartite graph (enumerate directly); hafnian of a 0/1 adjacency = number of
perfect matchings of the graph (enumerate directly). Plus exact closed forms:
$\mathrm{perm}(J_n)=n!$, $\mathrm{perm}(I)=1$; $\mathrm{haf}$ of the all-ones
$2n\times2n$ matrix $=(2n-1)!!$ (perfect matchings of $K_{2n}$); direct-sum
multiplicativity $\mathrm{haf}(A\oplus B)=\mathrm{haf}(A)\,\mathrm{haf}(B)$ and
likewise for the permanent; small torontonian cases by explicit photon-number
summation. These are bedrock because they're independent of every existing
library.

**Layer 2 — The Walrus as differential oracle.** Exact agreement (to precision)
on large suites of random real/complex matrices across sizes and seeds, for all
four functions. The Walrus is the feature-complete reference; matching it
broadly is necessary but, per Layer 1, not sufficient.

**Layer 3 — Mathematical invariants / property-based (Hypothesis).** Permutation
invariance (simultaneous row/col perms leave $\mathrm{haf}$/$\mathrm{tor}$
unchanged; $\mathrm{perm}(PAQ)$ relations); scaling
$\mathrm{perm}(cA)=c^n\mathrm{perm}(A)$,
$\mathrm{haf}(cA)=c^{n/2}\mathrm{haf}(A)$; transpose invariance; loop hafnian →
hafnian when the diagonal is zero; real input → real output; block/triangular
structure identities. Properties hold across random inputs by construction —
they catch classes of bugs point tests miss.

**Layer 4 — Statistical / end-to-end (GBS).** Use the kernels to compute GBS
probabilities and verify they form a valid distribution (sum to 1 over a
truncated space within tolerance); validate sampled distributions against The
Walrus / Strawberry Fields on small instances via total-variation distance and
chi-square goodness-of-fit; check conditional/marginal consistency along the
sampling chain. This tests the functions *in their actual scientific use*, not
in isolation.

**Layer 5 — Numerical accuracy characterization & engineering meta-tests.**
Against the §6 mpmath reference: relative-error curves per precision tier vs
size and conditioning, including **deliberately cancellation-heavy matrices**
that must break FP64 and survive DD — and the boundary is *published*, not
hidden. Engineering invariants: **batched-equals-looped** (a batch result is
bit-identical, per precision mode, to the same inputs run singly),
**GPU-equals-CPU-reference** within tier tolerance, determinism (fixed input →
reproducible output), memory-bound checks, and a benchmark-honesty guard (a
post-sync checksum on returned values so no async early-return can fake a fast
time).

**CI policy:** Layers 1, 3, and the CPU-reference parts of Layer 2 run on **CPU
on every commit** (free GitHub runners) — these need no GPU. GPU correctness
(GPU-vs-CPU-equivalence, DD paths) and all timing run only in **scripted, manual
rented-GPU sessions**; nothing on a rented GPU starts until its CPU dry-run is
green. No published number ever comes from a shared CI runner.

---

## 9. Benchmarking (accuracy-normalized)

The benchmark claim is **throughput at a stated accuracy**, never raw
throughput. Report, against The Walrus (CPU) on the same instance:
batched-evaluation throughput (evals/sec) vs batch size and matrix size, at each
precision tier and at fixed achieved-accuracy levels; the FP64↔DD crossover;
single-eval latency (where CPU honestly often wins for small $n$ — publish it);
peak memory. Hygiene: pinned containers (digest-recorded), one shared input
generator so cross-engine timings are same-input, recorded GPU clocks/thermals,
repeated runs with median/IQR (never best-of-N), randomized order, raw data
published, **no composite "winner" number**. Honesty about where The Walrus wins
is what makes the wins credible. The full protocol — provenance fields, warm-up
policy, the three input regimes, the two-card GPU set, the same-instance
baseline, and the crossover figures — is frozen in `docs/benchmark_protocol.md`.

---

*End of design document. Code comments cite this file by section number
(`docs/DESIGN.md §N`); keep the section numbers stable.*
