"""Generate stratum-specific click normalizers for calibration posterior draws.

Block drift changes both event probabilities and click-count probabilities.
Consequently reconstruction normalizers must retain draw and acquisition-
stratum axes; a single ``(draw, band)`` value cannot represent the registered
nuisance model. This driver writes paired probabilities with shape
``(draw, stratum, model, band)`` using common random numbers across models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from confirmatory_common import (analysis_source_hash, current_commit, sha256_file,
                                 valid_container_digest,
                                 write_npz_exclusive)  # noqa: E402
from confirmatory_contract import load_registration  # noqa: E402
from joint_normalizer_replicates import coherence_inputs_from_config  # noqa: E402
from reconstruction_replicates import (CALIBRATION_FINGERPRINT_METHOD,
                                       PAIRED_NORMALIZER_FINGERPRINT_METHOD,
                                       _validate_transfer,
                                       calibration_draw_fingerprints,
                                       paired_normalizer_draw_fingerprints,
                                       load_calibration)  # noqa: E402


GroupedEvaluator = Callable[[np.ndarray, np.ndarray, np.ndarray, int, int, int], tuple]


def generate(
    calibration: dict[str, np.ndarray], model_points: dict[str, float], *,
    parameterization: str, bands: list[int], samples_per_draw: int, seed: int,
    grouped_evaluator: GroupedEvaluator | None = None,
) -> dict[str, np.ndarray]:
    """Return calibration-paired click distributions for every stratum/model."""
    if samples_per_draw < 1 or not bands or any(band < 0 for band in bands):
        raise ValueError("normalizer bands/samples are invalid")
    r_draws = np.asarray(calibration["r25"], dtype=float)
    transfers = np.asarray(calibration["T"])
    drift = np.asarray(calibration["block_drift"])
    if (r_draws.ndim != 2 or transfers.ndim != 3 or drift.ndim != 4
            or len(r_draws) != len(transfers) or len(r_draws) != len(drift)):
        raise ValueError("calibration draw axes are inconsistent")
    if grouped_evaluator is None:
        from thewalrus.grouped_click_probabilities import grouped_click_probabilities

        grouped_evaluator = grouped_click_probabilities

    names = list(model_points)
    rows: list[list[list[np.ndarray]]] = []
    for draw_index, (r25, transfer) in enumerate(
            zip(r_draws, transfers, strict=True)):
        draw_rows: list[list[np.ndarray]] = []
        for stratum in range(drift.shape[1]):
            transfer_s = np.asarray(transfer, dtype=np.complex128).copy()
            if "efficiency" in calibration:
                transfer_s *= np.sqrt(np.asarray(calibration["efficiency"])[
                    draw_index])[:, None]
            transfer_s += drift[draw_index, stratum]
            _validate_transfer(transfer_s)
            replicate_seed = int(seed) + draw_index * drift.shape[1] + stratum
            model_rows: list[np.ndarray] = []
            for name in names:
                phn, chn, output_transfer = coherence_inputs_from_config(
                    float(model_points[name]), r25=r25, transfer=transfer_s,
                    parameterization=parameterization)
                probabilities, _ = grouped_evaluator(
                    phn, chn.astype(np.complex128),
                    output_transfer.astype(np.complex128),
                    int(samples_per_draw), 1, replicate_seed)
                values = np.asarray(probabilities, dtype=float)
                if (values.ndim != 1 or np.any(~np.isfinite(values))
                        or np.any(values < 0) or np.any(values > 1)):
                    raise ValueError("grouped-click generator returned invalid probabilities")
                model_rows.append(values)
            lengths = {len(values) for values in model_rows}
            if len(lengths) != 1:
                raise ValueError("grouped-click model outputs have inconsistent support")
            draw_rows.append(model_rows)
        rows.append(draw_rows)
    full = np.asarray(rows, dtype=float)
    if full.ndim != 4 or max(bands) >= full.shape[-1]:
        raise ValueError("calibration normalizers do not cover every registered band")
    selected = full[..., bands]
    if np.any(selected <= 0):
        raise ValueError("calibration normalizer mass is zero in a registered band")
    calibration_fingerprints = calibration_draw_fingerprints(calibration)
    paired_fingerprints = paired_normalizer_draw_fingerprints(
        calibration_fingerprints, selected)
    return {"p_models": selected, "p_models_full": full,
            "p_reference": selected[:, :, 0, :],
            "p_alternative": selected[:, :, 1, :],
            "calibration_draw_sha256": np.asarray(calibration_fingerprints),
            "paired_normalizer_draw_sha256": np.asarray(paired_fingerprints)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    registration = load_registration(args.registration)
    plan = registration["plan"]
    calibration_cfg = plan["analysis"]["calibration_draws"]
    calibration_hash = sha256_file(args.calibration)
    if calibration_hash != calibration_cfg["posterior_sha256"]:
        raise SystemExit("calibration posterior hash differs from registration")
    commit = current_commit()
    if commit != plan["analysis_commit"]:
        raise SystemExit("calibration-normalizer commit differs from registration")
    source_hash = analysis_source_hash()
    if source_hash != plan["numerical_contract"]["analysis_source_sha256"]:
        raise SystemExit("calibration-normalizer source bytes differ from registration")
    container_digest = os.environ.get("GBS_CONTAINER_DIGEST")
    if (plan["external_requirements"].get("container_digest_required", True)
            and not valid_container_digest(container_digest)):
        raise SystemExit("calibration normalizers require GBS_CONTAINER_DIGEST")

    bands = [int(x) for x in plan["selection"]["bands"]]
    n_strata = int(plan["selection"]["n_strata"])
    calibration = load_calibration(args.calibration, bands, n_strata=n_strata)
    if len(calibration["r25"]) != int(calibration_cfg["count"]):
        raise SystemExit("calibration draw count differs from registration")
    models = plan["models"]
    reference = str(models["reference_model"])
    alternative = str(models["alternative_model"])
    points = {str(k): float(v) for k, v in models["coherence_points"].items()}
    order = [reference, alternative] + sorted(
        set(points) - {reference, alternative})
    normalizer_cfg = plan["analysis"]["normalizer_replicates"]
    output = generate(
        calibration, {name: points[name] for name in order},
        parameterization=str(models.get("parameterization", "classical_excess")),
        bands=bands,
        samples_per_draw=int(normalizer_cfg["samples_per_replicate"]),
        seed=int(calibration_cfg["seed"]),
    )
    import thewalrus

    meta = {
        "schema": "gbskernels.calibration-normalizer-draws.v1",
        "bands": bands, "n_strata": n_strata,
        "draws": int(len(calibration["r25"])),
        "registration_id": registration["public"]["plan_sha256"],
        "analysis_commit": commit, "analysis_source_sha256": source_hash,
        "container_digest": container_digest,
        "calibration_posterior_sha256": calibration_hash,
        "pairing": "calibration_draw_and_common_stratum",
        "calibration_draw_fingerprint_method": CALIBRATION_FINGERPRINT_METHOD,
        "paired_normalizer_fingerprint_method": PAIRED_NORMALIZER_FINGERPRINT_METHOD,
        "model_names": order,
        "coherence_points": [points[name] for name in order],
        "parameterization": str(models.get("parameterization", "classical_excess")),
        "exp_id": int(models["exp_id"]),
        "samples_per_draw_stratum": int(normalizer_cfg["samples_per_replicate"]),
        "seed": int(calibration_cfg["seed"]),
        "seed_rule": "seed + draw_index * n_strata + stratum; common across models",
        "thewalrus_version": getattr(thewalrus, "__version__", "unknown"),
        "numpy_version": np.__version__,
    }
    write_npz_exclusive(args.out, meta=json.dumps(meta, sort_keys=True), **output)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
