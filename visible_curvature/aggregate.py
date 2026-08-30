from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

FOCUSED_TABLES = (
    "block_metrics.csv",
    "bootstrap_metrics.csv",
    "interventions.csv",
    "alpha_sweep.csv",
    "damping_sweep.csv",
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
        for column in ("delta_g", "delta_g_predicted", "K_adam", "K_shampoo"):
            if column not in group:
                continue
            values = group[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if values.size:
                row[f"{column}_median"] = float(np.median(values))
                row[f"{column}_q025"] = float(np.quantile(values, 0.025))
                row[f"{column}_q975"] = float(np.quantile(values, 0.975))
        labels = _seed_labels(group)
        label, reason = consensus_label(labels, minimum_seed_count=minimum_seed_count)
        row["reliable_ordering"] = label
        row["reliable_ordering_consensus_reason"] = reason
        rows.append(row)
    return pd.DataFrame(rows)


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
    tiers = {str(m.get("experiment_tier", "")) for m in manifests if m}
    compatible_protocol_hashes = len(protocol_hashes) == 1 and "" not in protocol_hashes
    compatible_runtime_identities = len(runtime_ids) == 1 and "" not in runtime_ids
    if not allow_incompatible and not compatible_protocol_hashes:
        raise ValueError("Refusing to aggregate incompatible protocol hashes")
    if not allow_incompatible and not compatible_runtime_identities:
        raise ValueError("Refusing to aggregate incompatible runtime identities")

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
        "no_block_failures": no_block_failures,
        "scientific_run": scientific_run,
        "synthetic_backend": synthetic_backend,
        "immutable_revisions": immutable_revisions,
        "experiment_tier": experiment_tier,
        "minimum_seed_count": int(minimum_seed_count),
        "minimum_seed_count_met": minimum_seed_count_met,
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
        "protocol_hashes": sorted(protocol_hashes),
        "runtime_identity_sha256": sorted(runtime_ids),
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
