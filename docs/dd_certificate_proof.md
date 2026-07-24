# Proof of the double-double torontonian enclosure (post-audit)

Status: **desk proof with release-gated device evidence.** This document
supplies the implementation-specific numerical proof required by the public
[`confirmatory_v2.md`](confirmatory_v2.md) contract and the historical-plan
reconciliation in
[`preregistration_jiuzhang1_confirmatory.md`](preregistration_jiuzhang1_confirmatory.md).
It covers the double-double (DD) single-instance recursive torontonian and
states and proves the bounds used by the corrected kernel
(`core/tor_recursive.cu`, `core/dd.cuh`, and
`core/certified_rounding.cuh`) relies on. It is the mathematical half of Item 1; the
empirical half -- enclosure against an independent-precision reference on
adversarial and physical inputs -- is the on-device test harness
(`examples/jiuzhang/dd_adversarial_enclosure.py`) and is not a theorem proved by
this document. A tagged release is permitted only when the clean exact commit
produces the hash-bound validation manifest described in the
[release evidence record](../results/README.md#v021-device-validation).

Notation follows the manuscript supplement (M1–M5): `u = 2^-53` is the binary64
unit roundoff; `fl(·)` is round-to-nearest-even; `ru_*` / `rd_*` are the
directed (round-up / round-down) intrinsics `__dadd_ru`, `__dmul_ru`,
`__ddiv_ru`, `__dsqrt_rd`, and the newly added `__dsub_rd` (host shims:
`nextafter(·, ±∞)`). For a tracked quantity `x̂` with exact counterpart `x`,
`e(x̂)` denotes a stored bound with the invariant `|x̂ − x| ≤ e(x̂)`, and every
operation that updates `e(·)` is directed outward (M5), so `e(·)` is always a
rigorous upper bound with no global inflation factor.

---

## 0. The defect this corrects

A double-word (double-double) value is a pair `x = (x.hi, x.lo)` of binary64
numbers representing the real number `⟦x⟧ = x.hi + x.lo`, with the non-overlap
invariant

    x.hi = fl(⟦x⟧),   hence   |x.lo| ≤ u·|x.hi|            (DW-inv)

(`|x.lo|` is a binary64 rounding residual of a number whose leading word is
`x.hi`; it is bounded by half an ulp of `x.hi`, i.e. `≤ u·|x.hi|`, **not** by
`u_DD·|x.hi|`).

The pre-audit lower-magnitude helper was

    md_lo_old(x) = rd_mul(|x.hi|, 1 − 2·u_DD),   u_DD = 2^-100.

This is **not** a valid lower bound on `|⟦x⟧|`. Two independent reasons:

1. **It ignores the sign of `x.lo`.** When `x.lo < 0`,
   `|⟦x⟧| = |x.hi| − |x.lo|` can be as small as `|x.hi|·(1 − u) = |x.hi|(1 −
   2^-53)`, whereas `md_lo_old` returns ≈ `|x.hi|·(1 − 2^-99)`. Since
   `2^-99 ≪ 2^-53`, `md_lo_old > |⟦x⟧|` — the "lower bound" exceeds the value.
2. **The factor rounds to one.** `2·u_DD = 2^-99` is far below the binary64
   spacing at 1.0 (`2^-52`), so `fl(1 − 2^-99) = 1.0` exactly; thus
   `md_lo_old(x) = rd_mul(|x.hi|, 1.0) = |x.hi|` (rounded down), which fails
   reason 1 by the full `|x.lo|`.

Because `md_lo` feeds every pivot guard and every denominator lower bound in
the Cholesky recursion, an over-large `md_lo` produces an **under**-large error
charge (dividing a numerator error by too large a denominator, or clearing a
pivot guard that should refuse), so a returned enclosure `E` could be smaller
than the true error -- the enclosure is unsound, not merely loose. This was
the first fatal numerical issue identified by the release audit.

---

## 1. Magnitude enclosure of a double-word value (the fix)

**Lemma DW1 (triangle magnitude bounds).** For a double-word `x` satisfying
(DW-inv), define

    md_hi(x) = ru_add(|x.hi|, |x.lo|)
    md_lo(x) = max(0, rd_sub(|x.hi|, |x.lo|)).

Then `md_lo(x) ≤ |⟦x⟧| ≤ md_hi(x)`.

*Proof.* By the triangle inequality in ℝ,

    ||x.hi| − |x.lo||  ≤  |x.hi + x.lo|  ≤  |x.hi| + |x.lo|.        (1)

*Upper.* `|x.hi| + |x.lo|` is a sum of two nonnegative binary64 numbers;
`ru_add` rounds it upward, so `md_hi(x) ≥ |x.hi| + |x.lo| ≥ |⟦x⟧|` by the right
half of (1). `|x.hi|` and `|x.lo|` are exact (negation and `fabs` are exact in
IEEE-754), so no error enters before the single directed `ru_add`.

*Lower.* By (DW-inv), `|x.lo| ≤ u·|x.hi| < |x.hi|` (for `x.hi ≠ 0`), so
`|x.hi| − |x.lo| ≥ 0` and the left side of (1) equals `|x.hi| − |x.lo|`.
`rd_sub` rounds `|x.hi| − |x.lo|` downward, so
`rd_sub(|x.hi|,|x.lo|) ≤ |x.hi| − |x.lo| ≤ |⟦x⟧|` by the left half of (1).
The `max(0, ·)` only raises the bound toward 0 when the subtraction underflows
to a tiny negative (it cannot exceed the true `|⟦x⟧| ≥ 0`), preserving
`md_lo(x) ≤ |⟦x⟧|`. If `x.hi = 0` then `x.lo = 0` by (DW-inv) and both bounds
are 0. ∎

**Remark.** DW1 uses the *stored* low word, so it is tight to within one
directed rounding regardless of the relative sign of `x.hi`, `x.lo`. This is
the operation that `md_hi`/`md_lo` now implement in `core/tor_recursive.cu`.

**Lemma DW0 (directed subtraction is outward).** For binary64 `a ≥ b ≥ 0`,
`rd_sub(a,b) = __dsub_rd(a,b) ≤ a − b`, and on the host shim
`nextafter(a−b, −∞) ≤ a − b`. *Proof.* `__dsub_rd` is correctly rounded toward
−∞ by IEEE-754, so its result is `≤ a − b`. `a − b` is a real number; its
round-to-nearest value `fl(a−b)` may exceed `a − b`, but
`nextafter(fl(a−b), −∞) ≤ a − b` because at most one binary64 lies strictly
between `a − b` and `fl(a−b)` and `nextafter` steps past `fl(a−b)` toward −∞
by one ulp, which brackets `a − b` (Sterbenz cancellation makes `a−b` exact
when `a/2 ≤ b ≤ 2a`, in which case both sides are equal). ∎

---

## 2. Operation and residual bounds used by the kernel

The add/multiply accumulation chains use the published double-word analysis for
error-free-transform implementations and charge the conservative relative
constant

    u_DD = 2^-100.

That charge is used only for DD add/multiply chains. The kernel does **not**
assign a literature constant to `dd_sqrt`, and it does not rely on the forward
constant of `dd_div`. Both nonlinear operations are checked a posteriori from
the residual of the value actually returned by `dd.cuh`.

The accumulation factor is formed as

    gamma_k = ru_div(ru_mul(k,u_DD), rd_sub(1,ru_mul(k,u_DD))).

This construction is materially important: the former multiplier
`1 + 4*u_DD` rounds to exactly `1` in binary64 and therefore supplied no upward
slack. `core/check_certified.cu` probes the directed expression on device and
requires `gamma_64 > 64*u_DD`.

Every binary64 value operation also receives an absolute term

    eta = 8e-323 = 16 * 2^-1074.

Thus the local model is relative-plus-absolute, not a relative-only model:
`|fl(x op y) - (x op y)| <= u|fl(x op y)| + eta`. The Cholesky dot updates
charge `(8c+16) eta` at column `c`, which exceeds the number of binary64
primitives in the corresponding DD multiply/add chain. Determinant products,
division, square root, reciprocal, signed accumulation, and final collapse
each carry their own explicit `eta`. If underflow destroys a relative bound,
the absolute term remains; if it overtakes a positive pivot or determinant,
the existing guard refuses with `+inf`.

**Lemma RES (implemented product residual).** Let `a`, `b`, and `c` be stored
double words and define

    R(a,b,c) = |<a><b> - <c>|,       <x> = x.hi + x.lo.

`dd_mul_sub_residual_absu` evaluates the six-term expansion

    a.hi*b.hi + a.hi*b.lo + a.lo*b.hi + a.lo*b.lo - c.hi - c.lo

as five explicit binary64 FMAs. After each rounded result `q_i`, it outwardly
adds `2u|q_i| + eta` to an error accumulator. The returned

    delta_R = ru_add(|q_5|, sum_i (2u|q_i| + eta))

satisfies `R(a,b,c) <= delta_R`.

*Proof.* An explicit FMA rounds the exact product-plus-add once. For a rounded
binary64 result `q`, `|q-q_exact| <= u/(1-u)|q| + eta < 2u|q| + eta`.
The prior step's uncertainty enters the next FMA addend with coefficient one,
so induction over the five FMAs gives the accumulated error. The expansion is
algebraically `<a><b>-<c>`. Directed `ru_*` operations make the stored sum an
upper bound. This ordering retains the high-product cancellation; unlike first
collapsing the product to binary64, it resolves an `O(u^2)` DD residual. ∎

**Lemma DD-B1 (DD pair product, tracked).** If `|<a>-a_*| <= e_a`,
`|<b>-b_*| <= e_b`, and `p = dd_mul(a,b)`, then

    |<p> - a_* b_*| <= M(a)e_b + M(b)e_a + e_a e_b + delta_R(a,b,p),

with every term rounded outward and `M = md_hi`.

*Proof.* Expand the input perturbations and apply the triangle inequality; Lemma
RES bounds the implementation error of the actual `dd_mul` result. ∎

**Lemma DD-B6 (residual-certified division).** Under
`e_b < 0.5 md_lo(b)`, let `v = dd_div(a,b)` and
`d_lo = rd_sub(md_lo(b),e_b)>0`. Then the implemented bound

    ( delta_R(v,b,a) + e_a + M(v)e_b ) / d_lo

is at least `|<v>-a_*/b_*|`.

*Proof.* Multiply the quotient error by the exact denominator:

    |<v>b_* - a_*|
      <= |<v><b>-<a>| + M(v)|b_*-<b>| + |a_*-<a>|.

Lemma RES bounds the first term; the tracked input errors bound the other two;
and `|b_*| >= d_lo`. Division by the downward lower bound and upward rounding
complete the result. No forward-error property of `dd_div` is assumed. ∎

**Lemma DD-B5 (residual-certified square root).** Let
`|<s>-s_*| <= e_s`, `e_s < 0.5 md_lo(s)`, and `r = dd_sqrt(s)`. Define

    z_lo = rd_sub(md_lo(s),e_s),
    d_lo = rd_add(md_lo(r),rd_sqrt(z_lo)),
    delta = delta_R(r,r,s).

If the positive guards hold, the implemented bound

    (delta + e_s) / d_lo

is at least `|<r>-sqrt(s_*)|`.

*Proof.* Factor the difference of squares:

    |<r>-sqrt(s_*)|
      = |<r>^2-s_*| / (|<r>|+sqrt(s_*))
      <= (delta+e_s) / (md_lo(r)+sqrt(z_lo)).

Lemma RES supplies `delta`; DW1 and directed downward operations supply the
denominator lower bound. No forward-error constant for `dd_sqrt` is used. In
the source, `rd_add(x,y)` is written equivalently as `rd_sub(x,-y)`. ∎

The same residual lemma certifies the leaf reciprocal `c=dd_div(1,r)`: with a
tracked root error `t`,

    |<c>-1/r_*| <= (delta_R(c,r,1) + M(c)t) / (md_lo(r)-t).

Thus every DD division and square root in the recursive certificate is tied to
the actual returned bits rather than an undocumented implementation constant.

---

## 3. Propagation through the Cholesky DFS

The single-large kernel evaluates `tor(O) = Σ_{S⊆[n]} (−1)^{n−|S|} /
√det(I − O_S)` by a depth-first walk over click subsets, maintaining a
double-word Cholesky factor `L` of the current principal minor and a parallel
binary64 error factor `EL` with the invariant `EL[i][j] ≥ |⟦L[i][j]⟧ −
L_exact[i][j]|`. At each visited node:

* **off-diagonal** `L[r][c] = dd_div(s_acc, L[c][c])`, error by Lemma DD-B6 with
  `e_b = EL[c][c]`, guard `EL[c][c] < ½·md_lo(L[c][c])` (else refuse);
* **diagonal** `L[r][r] = dd_sqrt(s_acc)`, error by Lemma DD-B5 with
  `e_s = EL`-accumulated, guard `e_s < ½·md_lo(s_acc)` (else refuse);
* **subset term** `det = Π L[i][i]²`, each actual `dd_mul` error by Lemma RES
  plus the tracked input polydisc terms, then the square root by DD-B5 and the
  reciprocal by the residual formula following DD-B5; the running signed sum
  is charged `u_DD·|S_k| + eta` per `dd_add`.

**Proposition (subtree enclosure).** If no guard refuses along the walk, the
per-subtree partial `p̂_t` returned to the host and its bound `b_t` satisfy
`|p̂_t − p_t| ≤ b_t`, where `p_t` is the exact-arithmetic value of the
implemented subtree formula. *Proof.* Induction over DFS depth: the input minor
is initialized with its explicit DD-add and underflow charge; each step composes
Lemma RES and DD-B1/B5/B6 (all outward, M5) with the
inductive `EL` invariant; the saved/restored `(det, e_det)` stack reuses a
bound already proven for the shared prefix, so prefix reuse is sound. A refused
guard writes `NaN`/`+∞` and the subtree contributes `+∞` to `b_t` (Section 5),
never a finite under-charge. ∎

---

## 4. DD → binary64 collapse (host reduction)

Each subtree partial is collapsed from double-word to binary64 before the host
sums the `2^g` subtrees. The pre-audit code folded a single `u·|·|` charge into
the leaf; the corrected code recovers the collapse residual and adds explicit
underflow slack.

**Lemma COLL (collapse residual).** The kernel collapses a double-word partial
`p̂` to binary64 by `TwoSum`: `val = fl(⟦p̂⟧)` with the exact residual
`r = ⟦p̂⟧ − val` recovered (up to the explicit `eta` underflow term). The
charged bound is
`b_t ← ru_add(ru_add(b_t, u·|val|), ru_add(|r|,eta))`, giving
`|val − p_t| ≤ b_t + u·|val| + |r| + eta`.
*Proof.* `|val − p_t| ≤ |val − ⟦p̂⟧| + |⟦p̂⟧ − p_t|`; `TwoSum` gives the first
term exactly absent underflow and within `eta` otherwise,
and `|⟦p̂⟧ − p_t| ≤ b_t` by the subtree induction. The additional `u·|val|`
term is conservative rather than necessary for this triangle bound; it records
the binary64 output granularity explicitly and matches the host reduction's
per-add charge. ∎

The gate `core/check_tor_recursive.cu` isolates collapse at `n=1, g=1`, exercises
the sqrt/division residual path on a correlated SPD matrix, and checks that a
determinant product driven below the subnormal range refuses instead of returning
a finite bound. The host then sums the collapsed `(c_t,b_t)` values in
`core/host_api.cu`; each value addition is charged `u|total|+eta`, and each bound
addition is rounded upward with `nextafter`, giving the final `(y,E)` with
`|y-tor(O)| <= E`.

### Generic permanent and hafnian certificates

The batched FP64 and DD permanent/hafnian certificates use the same absolute
underflow policy. In `core/certified.cu`, every row, trace, Newton, Gray-code,
subset-sum, halving, and final-scaling stage carries an explicit `eta` in
addition to its relative `gamma_k` term. Permanent products are enclosed by an
operation-by-operation pair recurrence. If `p` with bound `e_p` is multiplied
by row value `r` with bound `e_r`, the next bound is

    M(p)e_r + M(r)e_p + e_p e_r + c_mul M(p)M(r) + eta.

The absolute term is therefore multiplied by every later row magnitude. This
is required for mixed-scale chains: an early product may underflow and a later
large factor may amplify the lost value back into the normal range. A flat
underflow allowance applied only after the full product would not be valid.

In `core/certified_dd.cu`, the outward complex magnitude is

    M(z) = ru( safe*hypot(z_hi) + safe*hypot(z_lo) + eta ),

so a zero or subnormal high word cannot hide a nonzero low word. DD additions,
halvings, subset sums, and the final DD-to-binary64 collapse likewise carry
`eta`; the final permanent scaling uses directed division plus `eta`.

The hafnian Newton recurrence certifies its actual divide-by-integer result
`q = ddc_div_d(a,j)` a posteriori. For each real component it evaluates
`q_hi*j - a_hi + q_lo*j - a_lo` with three FMAs, charging each rounded FMA
`2u*|rounded| + eta`. If `R` is the outward sum of the real and imaginary
residual bounds and `e_a` encloses the recurrence accumulator, then

    |q - a_exact/j| <= ru((R + e_a)/j) + eta.

This is independent of a nominal DD-division relative constant. The focused
gate in `core/check_certified.cu` runs FP64 and DD permanent/hafnian kernels on
minimum-subnormal complex inputs and rejects any non-finite value or finite
bound below the absolute floor. It also checks the exact mixed-scale permanent
`diag(2^-600,2^-500,2^600)`, whose first product underflows but whose exact final
value is `2^-500`, against both permanent certificate tiers. It is part of the
host preflight and the CUDA device gate; the broader device-validation status
below remains unchanged.

The generic complex-LU division uses the component formula with denominator
`c^2+d^2` only when outward lower bounds prove both `B_lo^2` and
`B_lo(B_lo-e_b)` remain in the binary64 normal range. It additionally proves
that every nonzero numerator component product and every nontrivial numerator
dot result is normal. Otherwise it writes a `NaN` value and `+inf` bound. These
fail-closed guards are necessary because either the denominator or a much
smaller numerator can enter the subnormal range, where the relative-error
derivation used by `tor_p_div` does not apply. The perturbation term is
evaluated in the equivalent factorized form

    ru(e_a/(B_lo-e_b)) ru(B_hi/B_lo)
      + ru(e_b/B_lo) ru(A_hi/(B_lo-e_b)),

with every positive operation rounded upward. Thus an `e_a B_hi` product that
would underflow cannot disappear before division by the small denominator.
The focused probe covers `s=2^-538` with divisors `s+2si` and `s+si`, the exact
mixed-scale numerator regression
`a=(-4.072261452359005-4.94162777767184i)2^-572` and
`b=(12.307195185197452+9.424233382422166i)2^-500`, and the amplification case
`a=0`, `e_a=8e-323`, `b=2^-500`. The first three must refuse on the host shim
and device; the last must return a bound at least `e_a/|b|`.

At the public API boundary, a certified tier is accepted only when its complex
value and nonnegative bound are finite. A non-finite result triggers the next
tier when `rtol` requests the escalation ladder, or raises a refusal when no
fallback was requested. Non-finite inputs are rejected before evaluation, so a
`NaN` relative error can never compare false and be reported as certified.

---

## 5. Refusal conditions (never a finite overclaim)

A node refuses (writes `NaN` value, `+∞` bound) when any of:

* a pivot guard fails: `EL[c][c] ≥ ½·md_lo(L[c][c])` or
  `e_s ≥ ½·md_lo(s_acc)` — the certified interval of the pivot/radicand
  includes 0, so no finite relative bound is provable;
* a determinant lower bound is non-positive (`e_det ≥ det̂`) — the minor is not
  certifiably SPD;
* a residual, magnitude, or propagated bound is non-finite.

`+∞` propagates through `ru_add` to `E = +∞`, which the caller reports as a
refusal. **No path converts a failed guard into a finite bound.** Because the
guards use the corrected `md_lo` (Lemma DW1), they now refuse exactly the cases
the pre-audit guards could wrongly clear.

---

## 6. Required device-validation evidence (Item 1 empirical half)

This proof establishes soundness of the *stated operations*. Per audit Gate
1.4, the following must hold **on device** before an official release claims
the DD implementation passed its empirical gate. They are exercised by
`examples/jiuzhang/dd_adversarial_enclosure.py` against a 50-digit `mpmath`
reference:

1. negative low words in denominator/pivot bounds (forces the DW1 lower branch);
2. cancellation within and across subtrees (large κ);
3. pivots within a factor of 2 of the refusal boundary (guard exactness);
4. DD→binary64 collapse residuals near half an ulp (Lemma COLL exactness);
5. normal/subnormal transition cases (finite enclosure or explicit refusal);
6. dimensions through `k=14` in the independent-reference harness, together
   with the separate physical Gate C probe at its release-probe sizes.

The invariant to check is `|ŷ − y_ref| ≤ E` with **zero** violations, plus a
tightness distribution (bound / actual error) that stays finite on the
physical inputs and widens honestly (never inverts) on the adversarial ones.
The v0.2.1 tag is conditioned on this gate passing at the exact release commit,
together with the 24 device gates, binding smoke, and physical Gate C probe.
That release validation does not retroactively regenerate the older frontier,
parity, or historical event artifacts; those keep their legacy qualifiers.
