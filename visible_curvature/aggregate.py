from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .mechanism import (
    build_elasticity_prediction_rows,
    summarize_elasticity_predictions,
)

FOCUSED_TABLES = (
    "block_metrics.csv",
    "bootstrap_metrics.csv",
    "interventions.csv",
    "alpha_sweep.csv",
    "damping_sweep.csv",
    "ridge_sweep.csv",
    "block_failures.csv",
)

CANONICAL_TABLES = {
    "block_metrics.csv": "canonical_block_metrics.csv",
    "interventions.csv": "canonical_interventions.csv",
    "alpha_sweep.csv": "canonical_alpha_sweep.csv",
    "damping_sweep.csv": "canonical_damping_sweep.csv",
}


def _read_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def consensus_label(values: Sequence[object], minimum_seed_count: int = 1) -> tuple[str, str]:
    normalized = [str(value) if value is not None else "inconclusive" for value in values]
    if len(normalized) < int(minimum_seed_count):
        return "inconclusive", "insufficient_seed_count"
    if normalized and all(value == "positive" for value in normalized):
        return "positive", ""
    if normalized and all(value == "negative" for value in normalized):
        return "negative", ""
    signed = {value for value in normalized if value in {"positive", "negative"}}
    if signed == {"positive", "negative"}:
        return "inconclusive", "seed_sign_conflict"
    if not signed:
        return "inconclusive", "no_signed_seed"
    return "inconclusive", "seed_inconclusive"


def _seed_labels(group: pd.DataFrame) -> list[str]:
    label_column = (
        "balanced_reliable_ordering"
        if "balanced_reliable_ordering" in group
        else "reliable_ordering"
    )
    if label_column not in group:
        return []
    if "seed" not in group:
        return group[label_column].fillna("inconclusive").astype(str).tolist()
    labels: list[str] = []
    for _, seed_group in group.groupby("seed", dropna=False):
        values = seed_group[label_column].fillna("inconclusive").astype(str).tolist()
        label, _ = consensus_label(values, minimum_seed_count=1)
        labels.append(label)
    return labels


def _resolve_source_directory(path: Path) -> tuple[Path, bool, dict]:
    candidates = [path]
    if (path / "final").is_dir():
        candidates.insert(0, path / "final")
    for candidate in candidates:
        canonical = candidate / CANONICAL_TABLES["block_metrics.csv"]
        if canonical.exists():
            summary = _read_json(candidate / "balanced_reliability_summary.json")
            return candidate, True, summary
    for candidate in candidates:
        if (candidate / "block_metrics.csv").exists():
            return candidate, False, {}
    return path, False, {}


def _paired_seed_summary(metrics: pd.DataFrame, minimum_seed_count: int) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "protocol_hash", "runtime_identity_sha256", "block_name", "block_type",
                "covariance_moment", "assignment", "alpha", "n_seeds", "delta_g_median",
                "delta_g_q025", "delta_g_q975", "reliable_ordering",
                "reliable_ordering_consensus_reason",
                "condition_metric_consensus", "fallback_tau_consensus",
                "metric_consensus_reason", "bootstrap_ci_low_median",
                "bootstrap_ci_high_median", "tail_localized_consensus",
            ]
        )
    keys = [
        key for key in (
            "protocol_hash", "runtime_identity_sha256", "block_name", "block_type",
            "covariance_moment", "assignment", "alpha",
        ) if key in metrics.columns
    ]
    rows: list[dict] = []
    for key, group in metrics.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row["n_seed_rows"] = int(len(group))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        for column in (
            "delta_g",
            "delta_g_predicted",
            "K_H",
            "K_adam",
            "K_shampoo",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ):
            if column not in group:
                continue
            values = group[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if values.size:
                row[f"{column}_median"] = float(np.median(values))
                row[f"{column}_q025"] = float(np.quantile(values, 0.025))
                row[f"{column}_q975"] = float(np.quantile(values, 0.975))
        metric_reason = ""
        metric_consensus = "not_recorded"
        fallback_tau_consensus = float("nan")
        if "condition_metric" in group.columns:
            metric_values = {
                (
                    "truncated"
                    if str(value).strip().lower() == "truncated_condition"
                    else str(value).strip().lower()
                )
                for value in group["condition_metric"].dropna().tolist()
                if str(value).strip() and str(value).strip().lower() != "nan"
            }
            if len(metric_values) == 1:
                metric_consensus = next(iter(metric_values))
                if metric_consensus == "truncated":
                    if "fallback_tau" not in group.columns:
                        metric_consensus = "incompatible"
                        metric_reason = "fallback_tau_missing"
                    else:
                        taus = pd.to_numeric(group["fallback_tau"], errors="coerce").dropna().to_numpy(dtype=float)
                        if taus.size and np.allclose(taus, taus[0], rtol=0.0, atol=1.0e-12):
                            fallback_tau_consensus = float(taus[0])
                        else:
                            metric_consensus = "incompatible"
                            metric_reason = "fallback_tau_disagreement"
            else:
                metric_consensus = "incompatible"
                metric_reason = "condition_metric_disagreement"
        row["condition_metric_consensus"] = metric_consensus
        row["fallback_tau_consensus"] = fallback_tau_consensus
        row["metric_consensus_reason"] = metric_reason
        tail_consensus = "not_recorded"
        if "tail_localized_ordering" in group.columns:
            tail_values = {
                _truthy(value)
                for value in group["tail_localized_ordering"].dropna().tolist()
            }
            if tail_values == {True}:
                tail_consensus = "yes"
            elif tail_values == {False}:
                tail_consensus = "no"
            elif tail_values:
                tail_consensus = "mixed"
        row["tail_localized_consensus"] = tail_consensus
        labels = _seed_labels(group)
        label, reason = consensus_label(labels, minimum_seed_count=minimum_seed_count)
        if metric_consensus == "incompatible":
            label = "inconclusive"
            reason = metric_reason
        row["reliable_ordering"] = label
        row["reliable_ordering_consensus_reason"] = reason
        rows.append(row)
    return pd.DataFrame(rows)



def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _control_group_keys(frame: pd.DataFrame, extra: Sequence[str]) -> list[str]:
    return [
        key
        for key in (
            "protocol_hash",
            "runtime_identity_sha256",
            "block_name",
            "block_type",
            "seed",
            "covariance_moment",
            *extra,
        )
        if key in frame.columns
    ]


def _control_value(group: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(group[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return float(values.median()) if not values.empty else float("nan")


def _control_reliable(group: pd.DataFrame) -> bool:
    if "balanced_reliable_for_inference" not in group.columns:
        return False
    return bool(group["balanced_reliable_for_inference"].map(_truthy).all())


def _base_contrast_row(group: pd.DataFrame) -> dict[str, object]:
    if group.empty:
        return {}
    source = group.iloc[0]
    return {
        key: source[key]
        for key in (
            "protocol_hash",
            "runtime_identity_sha256",
            "block_name",
            "block_type",
            "seed",
            "covariance_moment",
        )
        if key in group.columns
    }


def _paired_control_contrasts(
    interventions: pd.DataFrame,
    alpha_sweep: pd.DataFrame,
    damping_sweep: pd.DataFrame,
    *,
    practical_alpha: float = 0.25,
) -> pd.DataFrame:
    """Build within-block, within-seed contrasts for all empirical controls."""
    rows: list[dict[str, object]] = []

    if not interventions.empty and {"assignment", "delta_g"}.issubset(interventions.columns):
        keys = _control_group_keys(
            interventions,
            ["alpha", "damping_coefficient", "sweep_mode"],
        )
        for _, group in interventions.groupby(keys, dropna=False):
            aligned = group[group["assignment"].astype(str) == "aligned"]
            reversed_rows = group[group["assignment"].astype(str) == "reversed"]
            if aligned.empty or reversed_rows.empty:
                continue
            aligned_value = _control_value(aligned, "delta_g")
            reversed_value = _control_value(reversed_rows, "delta_g")
            paired_reliable = _control_reliable(aligned) and _control_reliable(reversed_rows)
            sign_reversal = bool(
                np.isfinite(aligned_value)
                and np.isfinite(reversed_value)
                and aligned_value * reversed_value < 0.0
            )
            rows.append(
                {
                    **_base_contrast_row(group),
                    "contrast_type": "assignment_aligned_minus_reversed",
                    "assignment": "aligned_vs_reversed",
                    "reference_label": "reversed",
                    "comparison_label": "aligned",
                    "reference_value": reversed_value,
                    "comparison_value": aligned_value,
                    "contrast_value": aligned_value - reversed_value,
                    "control_estimand": "delta_g",
                    "expected_direction": "positive",
                    "expectation_satisfied": bool(aligned_value > reversed_value),
                    "sign_reversal": sign_reversal,
                    "paired_reliable": paired_reliable,
                }
            )

    if not alpha_sweep.empty and {"assignment", "alpha", "delta_g"}.issubset(alpha_sweep.columns):
        keys = _control_group_keys(
            alpha_sweep,
            ["assignment", "damping_coefficient", "sweep_mode"],
        )
        for _, group in alpha_sweep.groupby(keys, dropna=False):
            alpha_values = pd.to_numeric(group["alpha"], errors="coerce")
            reference = group[np.isclose(alpha_values, practical_alpha, rtol=0.0, atol=1.0e-12)]
            if reference.empty:
                continue
            reference_value = _control_value(reference, "delta_g")
            for alpha_value in sorted(
                value for value in alpha_values.dropna().unique() if not np.isclose(value, practical_alpha)
            ):
                comparison = group[np.isclose(alpha_values, alpha_value, rtol=0.0, atol=1.0e-12)]
                comparison_value = _control_value(comparison, "delta_g")
                rows.append(
                    {
                        **_base_contrast_row(group),
                        "contrast_type": "alpha_signed_change_from_practical",
                        "assignment": str(group.iloc[0].get("assignment", "")),
                        "reference_label": f"{practical_alpha:g}",
                        "comparison_label": f"{float(alpha_value):g}",
                        "reference_value": reference_value,
                        "comparison_value": comparison_value,
                        "contrast_value": comparison_value - reference_value,
                        "control_estimand": "signed_delta_g_change",
                        "expected_direction": "diagnostic",
                        "expectation_satisfied": np.nan,
                        "sign_reversal": bool(reference_value * comparison_value < 0.0),
                        "paired_reliable": _control_reliable(reference)
                        and _control_reliable(comparison),
                    }
                )

    required_damping = {"assignment", "damping_coefficient"}
    if not damping_sweep.empty and required_damping.issubset(damping_sweep.columns):
        value_column = "control_value" if "control_value" in damping_sweep.columns else "delta_g"
        keys = _control_group_keys(
            damping_sweep,
            ["assignment", "alpha", "sweep_mode", "control_estimand"],
        )
        for _, group in damping_sweep.groupby(keys, dropna=False):
            coefficients = pd.to_numeric(group["damping_coefficient"], errors="coerce")
            finite_coefficients = sorted(coefficients.dropna().unique())
            if len(finite_coefficients) < 2:
                continue
            reference_coefficient = float(finite_coefficients[0])
            reference = group[np.isclose(coefficients, reference_coefficient, rtol=0.0, atol=1.0e-15)]
            reference_value = _control_value(reference, value_column)
            for coefficient in finite_coefficients[1:]:
                comparison = group[np.isclose(coefficients, coefficient, rtol=0.0, atol=1.0e-15)]
                comparison_value = _control_value(comparison, value_column)
                contrast = comparison_value - reference_value
                rows.append(
                    {
                        **_base_contrast_row(group),
                        "contrast_type": "damping_change_from_minimum",
                        "assignment": str(group.iloc[0].get("assignment", "")),
                        "sweep_mode": str(group.iloc[0].get("sweep_mode", "")),
                        "reference_label": f"{reference_coefficient:g}",
                        "comparison_label": f"{float(coefficient):g}",
                        "reference_value": reference_value,
                        "comparison_value": comparison_value,
                        "contrast_value": contrast,
                        "control_estimand": str(
                            group.iloc[0].get("control_estimand", value_column)
                        ),
                        "expected_direction": "nonpositive",
                        "expectation_satisfied": bool(
                            np.isfinite(contrast) and contrast <= 0.0
                        ),
                        "sign_reversal": False,
                        "paired_reliable": _control_reliable(reference)
                        and _control_reliable(comparison),
                    }
                )

    columns = [
        "protocol_hash",
        "runtime_identity_sha256",
        "block_name",
        "block_type",
        "seed",
        "covariance_moment",
        "contrast_type",
        "assignment",
        "sweep_mode",
        "reference_label",
        "comparison_label",
        "reference_value",
        "comparison_value",
        "contrast_value",
        "control_estimand",
        "expected_direction",
        "expectation_satisfied",
        "sign_reversal",
        "paired_reliable",
    ]
    return pd.DataFrame(rows, columns=columns)

def _immutable_manifest(manifest: dict) -> bool:
    model = dict(manifest.get("model", {}))
    data = dict(manifest.get("data", {}))
    return bool(
        model.get("resolved_model_commit")
        and data.get("dataset_revision")
        and data.get("tokenizer_revision")
        and data.get("source_order_sha256")
        and data.get("selected_chunk_content_sha256")
    )


def aggregate_frozen_runs(
    run_dirs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    minimum_seed_count: int = 1,
    allow_incompatible: bool = False,
) -> Path:
    requested_runs = [Path(value) for value in run_dirs]
    if not requested_runs:
        raise ValueError("At least one focused frozen run directory is required")
    resolved = [_resolve_source_directory(path) for path in requested_runs]
    runs = [item[0] for item in resolved]
    balanced_flags = [item[1] for item in resolved]
    balanced_summaries = [item[2] for item in resolved]
    all_sources_balanced = bool(runs) and all(balanced_flags)
    any_sources_balanced = any(balanced_flags)
    if any_sources_balanced and not all_sources_balanced and not allow_incompatible:
        raise ValueError("Refusing to mix balanced canonical and legacy run sources")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifests = [_read_json(run / "run_manifest.json") for run in runs]
    source_manifests_present = all(bool(manifest) for manifest in manifests)
    if not source_manifests_present and not allow_incompatible:
        raise ValueError("Every source run must contain run_manifest.json")

    protocol_hashes = {str(m.get("protocol_hash", "")) for m in manifests if m}
    runtime_ids = {str(m.get("runtime_identity_sha256", "")) for m in manifests if m}
    runtime_environment_ids = {
        str(m.get("runtime_environment_sha256", "")) for m in manifests if m
    }
    tiers = {str(m.get("experiment_tier", "")) for m in manifests if m}
    compatible_protocol_hashes = len(protocol_hashes) == 1 and "" not in protocol_hashes
    compatible_runtime_identities = len(runtime_ids) == 1 and "" not in runtime_ids
    compatible_runtime_environments = (
        len(runtime_environment_ids) == 1 and "" not in runtime_environment_ids
    )
    if not allow_incompatible and not compatible_protocol_hashes:
        raise ValueError("Refusing to aggregate incompatible protocol hashes")
    if not allow_incompatible and not compatible_runtime_identities:
        raise ValueError("Refusing to aggregate incompatible runtime identities")
    if not allow_incompatible and not compatible_runtime_environments:
        raise ValueError("Refusing to aggregate incompatible runtime environments")

    collected: dict[str, list[pd.DataFrame]] = {name: [] for name in FOCUSED_TABLES}
    for run, manifest, balanced in zip(runs, manifests, balanced_flags):
        for name in FOCUSED_TABLES:
            source_name = CANONICAL_TABLES.get(name, name) if balanced else name
            frame = _read_csv(run / source_name)
            if frame is None:
                continue
            frame = frame.copy()
            frame["source_run_dir"] = str(run.resolve())
            if "runtime_identity_sha256" not in frame:
                frame["runtime_identity_sha256"] = manifest.get("runtime_identity_sha256")
            if "runtime_environment_sha256" not in frame:
                frame["runtime_environment_sha256"] = manifest.get(
                    "runtime_environment_sha256"
                )
            collected[name].append(frame)

    combined: dict[str, pd.DataFrame] = {}
    for name, frames in collected.items():
        nonempty = [frame for frame in frames if not frame.empty]
        if nonempty:
            frame = pd.concat(nonempty, ignore_index=True, sort=False)
        elif frames:
            frame = frames[0].iloc[0:0].copy()
        else:
            frame = pd.DataFrame()
        combined[name] = frame
        frame.to_csv(output / name, index=False)
        if all_sources_balanced and name in CANONICAL_TABLES:
            frame.to_csv(output / CANONICAL_TABLES[name], index=False)

    metrics = combined["block_metrics.csv"]
    paired = _paired_seed_summary(metrics, minimum_seed_count=int(minimum_seed_count))
    paired.to_csv(output / "paired_seed_summary.csv", index=False)

    elasticity_rows = build_elasticity_prediction_rows(metrics)
    elasticity_summary = summarize_elasticity_predictions(elasticity_rows)
    elasticity_rows.to_csv(output / "elasticity_prediction_rows.csv", index=False)
    elasticity_summary.to_csv(
        output / "elasticity_prediction_summary.csv", index=False
    )
    paired_controls = _paired_control_contrasts(
        combined["interventions.csv"],
        combined["alpha_sweep.csv"],
        combined["damping_sweep.csv"],
    )
    paired_controls.to_csv(output / "paired_control_contrasts.csv", index=False)

    primary = paired.copy()
    if not primary.empty:
        if "covariance_moment" in primary:
            primary = primary[primary["covariance_moment"].astype(str) == "centered"]
        if "assignment" in primary:
            primary = primary[primary["assignment"].astype(str) == "observed"]
        if "alpha" in primary:
            primary = primary[np.isclose(primary["alpha"].astype(float), 0.25)]
    minimum_seed_count_met = bool(
        len(primary) > 0 and (primary["n_seeds"].astype(int) >= int(minimum_seed_count)).all()
    )
    all_primary_metric_compatible = bool(
        len(primary) > 0
        and (
            "condition_metric_consensus" not in primary.columns
            or (primary["condition_metric_consensus"].astype(str) != "incompatible").all()
        )
    )
    all_primary_rows_numerically_accepted = False
    if all_sources_balanced and not metrics.empty:
        required_acceptance = [
            column
            for column in (
                "final_endpoint_numerically_accepted",
                "final_diagnostic_agreement",
            )
            if column in metrics.columns
        ]
        all_primary_rows_numerically_accepted = bool(
            len(required_acceptance) == 2
            and all(metrics[column].astype(bool).all() for column in required_acceptance)
        )
    failures = combined["block_failures.csv"]
    no_block_failures = failures.empty and all(int(m.get("num_failed_blocks", 0)) == 0 for m in manifests if m)
    scientific_run = source_manifests_present and all(bool(m.get("scientific_run", False)) for m in manifests)
    synthetic_backend = any(bool(m.get("synthetic_backend", False)) for m in manifests if m)
    immutable_revisions = source_manifests_present and all(_immutable_manifest(m) for m in manifests)
    experiment_tier = next(iter(tiers)) if len(tiers) == 1 else "mixed"

    summary = {
        "n_source_runs": len(runs),
        "n_seed_values": int(metrics["seed"].nunique()) if "seed" in metrics else 0,
        "n_named_blocks": int(primary["block_name"].nunique()) if "block_name" in primary else 0,
        "n_primary_positive": int((primary.get("delta_g_median", pd.Series(dtype=float)) > 0).sum()),
        "n_primary_negative": int((primary.get("delta_g_median", pd.Series(dtype=float)) < 0).sum()),
    }
    pd.DataFrame([summary]).to_csv(output / "aggregate_summary.csv", index=False)

    aggregate_manifest = {
        "schema_version": "2.0",
        "status": "complete",
        "source_manifests_present": source_manifests_present,
        "compatible_protocol_hashes": compatible_protocol_hashes,
        "compatible_runtime_identities": compatible_runtime_identities,
        "compatible_runtime_environments": compatible_runtime_environments,
        "no_block_failures": no_block_failures,
        "scientific_run": scientific_run,
        "synthetic_backend": synthetic_backend,
        "immutable_revisions": immutable_revisions,
        "experiment_tier": experiment_tier,
        "minimum_seed_count": int(minimum_seed_count),
        "minimum_seed_count_met": minimum_seed_count_met,
        "all_primary_metric_compatible": all_primary_metric_compatible,
        "reliability_mode": (
            "balanced_canonical" if all_sources_balanced else "legacy"
        ),
        "all_sources_balanced": all_sources_balanced,
        "canonical_tables_used": all_sources_balanced,
        "all_primary_rows_numerically_accepted": all_primary_rows_numerically_accepted,
        "all_balanced_sources_scientifically_accepted": bool(
            all_sources_balanced
            and all(
                summary.get("scientific_status") == "accepted"
                for summary in balanced_summaries
            )
        ),
        "n_named_blocks": summary["n_named_blocks"],
        "n_elasticity_prediction_rows": int(len(elasticity_rows)),
        "n_elasticity_prediction_eligible": int(
            elasticity_rows.get(
                "prediction_eligible", pd.Series(dtype=bool)
            ).map(_truthy).sum()
        ),
        "n_paired_control_contrasts": int(len(paired_controls)),
        "n_reliable_paired_control_contrasts": int(
            paired_controls.get(
                "paired_reliable", pd.Series(dtype=bool)
            ).map(_truthy).sum()
        ),
        "protocol_hashes": sorted(protocol_hashes),
        "runtime_identity_sha256": sorted(runtime_ids),
        "runtime_environment_sha256": sorted(runtime_environment_ids),
        "source_run_dirs": [str(run.resolve()) for run in runs],
        "requested_source_dirs": [
            str(run.resolve()) for run in requested_runs
        ],
        "required_tables": list(FOCUSED_TABLES),
    }
    (output / "aggregate_manifest.json").write_text(
        json.dumps(aggregate_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output
