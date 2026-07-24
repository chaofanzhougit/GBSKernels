"""Adversarial enclosure harness for the certified double-double torontonian.

This is the empirical half of the certificate audit
(docs/dd_certificate_proof.md section 6): it stresses the corrected DD
lower-bound and residual arithmetic against an independent 50-digit mpmath reference and asserts the
enclosure invariant |value - reference| <= bound with ZERO violations.

Two backends, one harness:
  --backend cpu   cpu_ref.certified.certified_torontonian  (the CPU twin; runs
                  ANYWHERE, no CUDA -> this is the CPU-dry-run that proves the
                  families, the reference, and the enclosure logic are correct)
  --backend gpu   gbskernels.tor_single(O, dd=True)        (the actual fixed
                  kernel; run this ON DEVICE to validate the fix before any DD
                  artifact is regenerated or trusted)

The reference is affordable only while 2^k subsets are tractable (k <= ~14), so
the enclosure test runs there; larger dimensions are covered on-device by the
internal-consistency gates in tests/test_tor_recursive.py and the closed forms
in core/check_tor_recursive.cu, not here.

    # CPU dry-run (no GPU):
    uv run python examples/jiuzhang/dd_adversarial_enclosure.py --backend cpu --kmax 10
    # on the box, after building the extension:
    python examples/jiuzhang/dd_adversarial_enclosure.py --backend gpu --kmax 14

Exit status 0 requires zero violations, nonempty coverage at every requested
dimension and adversarial family, and a bounded refusal fraction. Official GPU
validation additionally requires the physical Jiuzhang family. Refusals
(``bound = +inf``) are reported and never counted as passes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO),):
    if p not in sys.path:
        sys.path.insert(0, p)

from cpu_ref.diagnostics import torontonian_abs_term_sum  # noqa: E402
from highprec_ref import torontonian_mp  # noqa: E402

TINY = 5e-324


# --------------------------------------------------------------------------
# adversarial input families (real, xxpp, dimension 2k, with I - O_S SPD)
# --------------------------------------------------------------------------
def _sym(a: np.ndarray) -> np.ndarray:
    return (a + a.T) / 2.0


def _scale_spectral(O: np.ndarray, rho: float) -> np.ndarray:
    """Scale a symmetric O so its spectral radius is exactly rho (rho < 1 keeps
    I - O positive definite; rho -> 1 pushes pivots toward the refusal boundary
    and forces DD low words of opposite sign -- the regime the fix corrects)."""
    w = np.linalg.eigvalsh(O)
    m = np.max(np.abs(w))
    return O * (rho / m) if m > 0 else O


def family_random_spd(k: int, rng, rho: float = 0.4) -> np.ndarray:
    return _scale_spectral(_sym(rng.standard_normal((2 * k, 2 * k))), rho)


def family_near_refusal(k: int, rng, rho: float) -> np.ndarray:
    """Spectral radius rho just below 1 -> the smallest I - O_S pivots are tiny,
    so md_lo(pivot) - e must stay a valid positive lower bound (the fix)."""
    return _scale_spectral(_sym(rng.standard_normal((2 * k, 2 * k))), rho)


def family_cancellation(k: int, rng, rho: float = 0.9) -> np.ndarray:
    """Sign-structured O whose subset terms partially cancel: large
    torontonian_abs_term_sum / |tor| (high condition number)."""
    A = rng.standard_normal((2 * k, 2 * k))
    signs = np.where(rng.random((2 * k, 2 * k)) < 0.5, -1.0, 1.0)
    return _scale_spectral(_sym(A * signs), rho)


def family_physical(k: int, rng, states) -> np.ndarray | None:
    """A size-k mode subset of the real Jiuzhang xxpp O (the actual campaign
    regime).  Only available if the state data files are present."""
    if states is None:
        return None
    kind = "squeezed" if rng.random() < 0.5 else "squashed"
    O = states[kind]
    n = O.shape[0] // 2
    S = rng.choice(n, size=k, replace=False)
    idx = list(S) + [int(j) + n for j in S]
    return O[np.ix_(idx, idx)]


def _try_load_states():
    """Best-effort load of the real Jiuzhang state O matrices; returns None if
    the data files are absent (keeps the CPU dry-run self-contained)."""
    try:
        sys.path.insert(0, str(HERE))
        import q7_construction as q7  # noqa

        return {"squeezed": q7.build_state("squeezed")["O"],
                "squashed": q7.build_state("squashed")["O"]}
    except Exception as e:  # pragma: no cover - data-dependent
        print(f"  (physical family skipped: {type(e).__name__}: {e})", flush=True)
        return None


# --------------------------------------------------------------------------
# backends: (value, bound) for a real xxpp O
# --------------------------------------------------------------------------
def backend_cpu(O: np.ndarray):
    from cpu_ref.certified import certified_torontonian

    v, b = certified_torontonian(O)
    return float(np.real(v)), float(b)


def backend_gpu(O: np.ndarray):
    import gbskernels

    k = O.shape[0] // 2
    try:
        v, d = gbskernels.tor_single(np.ascontiguousarray(O, dtype=np.float64),
                                     groups=min(k, 13), dd=True)
    except ValueError:               # kernel refuses (non-finite / uncertifiable)
        return float("nan"), float("inf")
    return float(v), float(d["abs_error_bound"])


BACKENDS = {"cpu": backend_cpu, "gpu": backend_gpu}


# --------------------------------------------------------------------------
def run(
    backend: str,
    kmax: int,
    per_cell: int,
    dps: int,
    seed: int,
    *,
    require_physical: bool = False,
    max_refusal_fraction: float = 0.75,
) -> int:
    if kmax < 2 or per_cell <= 0 or dps < 20:
        raise ValueError("kmax >= 2, per_cell > 0, and dps >= 20 are required")
    if not 0.0 <= max_refusal_fraction < 1.0:
        raise ValueError("max_refusal_fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    evaluate = BACKENDS[backend]
    states = _try_load_states()

    # (name, builder) ; rho ladder pushes toward the refusal boundary
    cells = []
    for k in range(2, kmax + 1):
        cells.append(("random_spd", k, lambda k=k: family_random_spd(k, rng)))
        for rho in (0.9, 0.99, 0.999):
            cells.append((f"near_refusal_{rho}", k,
                          lambda k=k, rho=rho: family_near_refusal(k, rng, rho)))
        cells.append(("cancellation", k, lambda k=k: family_cancellation(k, rng)))
        cells.append(("physical", k, lambda k=k: family_physical(k, rng, states)))

    rows = []
    violations = []
    refusals = 0
    physical_skipped = 0
    t0 = time.time()
    for name, k, build in cells:
        for _ in range(per_cell):
            O = build()
            if O is None:
                if name == "physical":
                    physical_skipped += 1
                continue
            ref = complex(torontonian_mp(O, dps=dps)).real
            v, b = evaluate(O)
            kappa = torontonian_abs_term_sum(O) / max(abs(ref), TINY)
            refused = not math.isfinite(b)
            if refused:
                refusals += 1
                enclosed = None
                tight = None
            else:
                err = abs(v - ref)
                enclosed = err <= b
                tight = b / max(err, TINY)
                if not enclosed:
                    violations.append((name, k, err, b, kappa))
            rows.append({"family": name, "k": k, "kappa": kappa,
                         "value": v, "bound": b, "ref": ref,
                         "err": None if refused else abs(v - ref),
                         "enclosed": enclosed, "tightness": tight,
                         "refused": refused})

    n_checked = sum(1 for r in rows if not r["refused"])
    n_enclosed = sum(1 for r in rows if r["enclosed"])
    physical_cases = sum(1 for r in rows if r["family"] == "physical")
    checked_by_family = {
        name: sum(1 for row in rows if row["family"] == name and not row["refused"])
        for name, _, _ in cells
    }
    checked_by_k = {
        str(k): sum(1 for row in rows if row["k"] == k and not row["refused"])
        for k in range(2, kmax + 1)
    }
    required_families = {
        "random_spd", "near_refusal_0.9", "near_refusal_0.99",
        "near_refusal_0.999", "cancellation",
    }
    if require_physical or physical_cases:
        required_families.add("physical")
    refusal_fraction = refusals / len(rows) if rows else 1.0
    gate_failures = []
    if violations:
        gate_failures.append("one or more finite certificates violated enclosure")
    if n_checked == 0:
        gate_failures.append("no finite certificate was checked")
    missing_families = sorted(
        name for name in required_families if checked_by_family.get(name, 0) == 0
    )
    if missing_families:
        gate_failures.append(f"no finite check for families {missing_families}")
    missing_k = [k for k, count in checked_by_k.items() if count == 0]
    if missing_k:
        gate_failures.append(f"no finite check at k={missing_k}")
    expected_physical = (kmax - 1) * per_cell
    if require_physical and physical_cases != expected_physical:
        gate_failures.append(
            f"physical family incomplete: {physical_cases}/{expected_physical} cases"
        )
    if refusal_fraction > max_refusal_fraction:
        gate_failures.append(
            f"refusal fraction {refusal_fraction:.3f} exceeds {max_refusal_fraction:.3f}"
        )
    gate_pass = not gate_failures
    kappas = [r["kappa"] for r in rows if math.isfinite(r["kappa"])]
    tights = [r["tightness"] for r in rows if r["tightness"] is not None]
    summary = {
        "kind": "dd_adversarial_enclosure",
        "backend": backend,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_cases": len(rows), "n_checked": n_checked, "n_enclosed": n_enclosed,
        "n_violations": len(violations), "n_refusals": refusals,
        "refusal_fraction": refusal_fraction,
        "max_refusal_fraction": max_refusal_fraction,
        "physical_required": require_physical,
        "physical_cases": physical_cases,
        "physical_skipped": physical_skipped,
        "checked_by_family": checked_by_family,
        "checked_by_k": checked_by_k,
        "gate_pass": gate_pass,
        "gate_failures": gate_failures,
        "kmax": kmax, "dps": dps, "seed": seed,
        "kappa_max": max(kappas) if kappas else None,
        "tightness_median": float(np.median(tights)) if tights else None,
        "tightness_worst": max(tights) if tights else None,
        "elapsed_s": round(time.time() - t0, 2),
    }

    out = REPO / "results" / "jiuzhang"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = out / f"dd_adversarial_enclosure_{backend}_{stamp}.json"
    from bench._provenance import provenance

    path.write_text(json.dumps({
        "summary": summary,
        "provenance": provenance(),
        "rows": rows,
    }, indent=1))

    print(f"\n== dd adversarial enclosure ({backend}) ==")
    print(f"  cases {len(rows)}  checked {n_checked}  enclosed {n_enclosed}  "
          f"refusals {refusals}  violations {len(violations)}")
    if kappas:
        print(f"  kappa up to {max(kappas):.2e}   tightness median "
              f"{summary['tightness_median']:.2e} worst {summary['tightness_worst']:.2e}")
    if violations:
        print("  ENCLOSURE VIOLATIONS (fix is NOT sound on this backend):")
        for name, k, err, b, kappa in violations[:10]:
            print(f"    {name} k={k}: err {err:.3e} > bound {b:.3e} (kappa {kappa:.2e})")
    if gate_failures:
        print("  GATE FAILURES:")
        for failure in gate_failures:
            print(f"    {failure}")
    print(f"  -> {path}")
    return 0 if gate_pass else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=list(BACKENDS), default="cpu")
    ap.add_argument("--kmax", type=int, default=10, help="largest mode count k (dim 2k)")
    ap.add_argument("--per-cell", type=int, default=3, help="random draws per (family,k)")
    ap.add_argument("--dps", type=int, default=50, help="mpmath reference digits")
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--require-physical", action="store_true",
                    help="fail unless every requested physical-family case is present")
    ap.add_argument("--max-refusal-fraction", type=float, default=0.75)
    args = ap.parse_args()
    return run(
        args.backend,
        args.kmax,
        args.per_cell,
        args.dps,
        args.seed,
        require_physical=args.require_physical,
        max_refusal_fraction=args.max_refusal_fraction,
    )


if __name__ == "__main__":
    raise SystemExit(main())
