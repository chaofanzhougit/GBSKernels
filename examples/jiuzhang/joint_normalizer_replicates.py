"""Produce raw paired grouped-click replicates for registered model points.

The Walrus API returns a point estimate and across-group spread, but not the raw
group vectors required for cross-model/cross-band covariance. This driver calls
one group per independent seed and retains every paired probability vector.
Using the same replicate seed for all registered models implements common random
numbers and preserves the covariance the confirmatory analysis consumes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from confirmatory_contract import load_registration  # noqa: E402
from confirmatory_common import (analysis_source_hash, current_commit,
                                 valid_container_digest,
                                 write_npz_exclusive)  # noqa: E402
import coherence_family  # noqa: E402
import q7_construction as q7  # noqa: E402


def coherence_inputs_from_config(
    coherence: float, *, r25: np.ndarray, transfer: np.ndarray,
    parameterization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r25 = np.asarray(r25, dtype=float)
    transfer = np.asarray(transfer)
    if r25.ndim != 1 or transfer.ndim != 2 or transfer.shape[1] != 2 * len(r25):
        raise ValueError("normalizer state configuration has inconsistent dimensions")
    nbar = np.sinh(r25) ** 2
    if parameterization == "classical_excess":
        moment = coherence_family.excess_moment(nbar, coherence)
    elif parameterization == "physical_fraction":
        moment = coherence_family.anomalous_moment(nbar, coherence)
    else:
        raise ValueError(f"unknown parameterization {parameterization!r}")
    phn = np.repeat(nbar, 2)
    chn = np.empty_like(phn)
    chn[0::2], chn[1::2] = +moment, -moment
    pairing = q7.pair_beamsplitter(len(phn))[: len(phn), : len(phn)]
    output_transfer = np.ascontiguousarray(transfer @ pairing)
    if output_transfer.shape[1] != len(phn):
        raise ValueError("registered transfer matrix has the wrong input width")
    return phn, chn, output_transfer


def coherence_inputs(coherence: float, *, exp_id: int,
                     parameterization: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r25, transfer = q7.load_config(exp_id)
    return coherence_inputs_from_config(
        coherence, r25=r25, transfer=transfer,
        parameterization=parameterization)


def generate(model_points: dict[str, float], *, exp_id: int, parameterization: str,
             replicates: int, samples_per_replicate: int, seed: int) \
        -> dict[str, np.ndarray]:
    if replicates < 2 or samples_per_replicate < 1:
        raise ValueError("need >=2 replicates and >=1 sample per replicate")
    from thewalrus.grouped_click_probabilities import grouped_click_probabilities

    inputs = {name: coherence_inputs(value, exp_id=exp_id,
                                     parameterization=parameterization)
              for name, value in model_points.items()}
    output: dict[str, list[np.ndarray]] = {name: [] for name in inputs}
    for rep in range(replicates):
        rep_seed = seed + rep
        for name, (phn, chn, transfer) in inputs.items():
            probabilities, _ = grouped_click_probabilities(
                phn, chn + 0j * chn, transfer.astype(np.complex128),
                samples_per_replicate, 1, rep_seed)
            values = np.asarray(probabilities, dtype=float)
            if (values.ndim != 1 or np.any(~np.isfinite(values))
                    or np.any(values < 0) or np.any(values > 1)):
                raise ValueError("grouped-click generator returned invalid probabilities")
            output[name].append(values)
    return {name: np.stack(rows) for name, rows in output.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    reg = load_registration(args.registration)
    plan = reg["plan"]
    commit = current_commit()
    if commit != plan["analysis_commit"]:
        raise SystemExit("normalizer generator commit differs from registration")
    source_hash = analysis_source_hash()
    if source_hash != plan["numerical_contract"]["analysis_source_sha256"]:
        raise SystemExit("normalizer generator source bytes differ from registration")
    container_digest = os.environ.get("GBS_CONTAINER_DIGEST")
    if (plan["external_requirements"].get("container_digest_required", True)
            and not valid_container_digest(container_digest)):
        raise SystemExit("normalizer generation requires GBS_CONTAINER_DIGEST")
    cfg = plan["analysis"]["normalizer_replicates"]
    bands = [int(x) for x in plan["selection"]["bands"]]
    models = plan["models"]
    reference = str(models["reference_model"])
    alternative = str(models["alternative_model"])
    points = {str(k): float(v) for k, v in models["coherence_points"].items()}
    order = [reference, alternative] + sorted(set(points) - {reference, alternative})
    values = generate({name: points[name] for name in order}, exp_id=int(models["exp_id"]),
                      parameterization=str(models.get("parameterization", "classical_excess")),
                      replicates=int(cfg["count"]),
                      samples_per_replicate=int(cfg["samples_per_replicate"]),
                      seed=int(cfg["seed"]))
    import thewalrus
    model_full = np.stack([values[name] for name in order], axis=1)
    if max(bands) >= model_full.shape[2]:
        raise ValueError("normalizer distribution does not cover a registered band")
    if np.any(model_full[:, :, bands] <= 0):
        raise ValueError("registered normalizer mass is zero in a requested band")
    p0_full, p1_full = values[reference], values[alternative]
    meta = {"schema": "gbskernels.joint-normalizer-replicates.v1", "bands": bands,
            "registration_id": reg["public"]["plan_sha256"],
            "analysis_commit": commit, "container_digest": container_digest,
            "analysis_source_sha256": source_hash,
            "thewalrus_version": getattr(thewalrus, "__version__", "unknown"),
            "numpy_version": np.__version__,
            "reference_model": reference, "alternative_model": alternative,
            "model_names": order, "coherence_points": [points[name] for name in order],
            "parameterization": str(models.get("parameterization", "classical_excess")),
            "exp_id": int(models["exp_id"]),
            "replicate_count": len(p0_full),
            "samples_per_replicate": int(cfg["samples_per_replicate"]),
            "seed": int(cfg["seed"]),
            "seed_rule": "seed + replicate; common random numbers across models"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    write_npz_exclusive(
        args.out, meta=json.dumps(meta, sort_keys=True),
        p_reference=p0_full[:, bands], p_alternative=p1_full[:, bands],
        p_reference_full=p0_full, p_alternative_full=p1_full,
        p_models=model_full[:, :, bands], p_models_full=model_full)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
