#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

FOCUSED_OUTPUTS = (
    "block_metrics.csv",
    "bootstrap_metrics.csv",
    "interventions.csv",
    "alpha_sweep.csv",
    "damping_sweep.csv",
    "block_failures.csv",
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def validate_run(output_dir: str | Path) -> tuple[list[str], list[str]]:
    output = Path(output_dir)
    failures: list[str] = []
    warnings: list[str] = []

    aggregate_manifest = _read_json(output / "aggregate_manifest.json")
    if aggregate_manifest:
        for key in (
            "source_manifests_present",
            "compatible_protocol_hashes",
            "compatible_runtime_identities",
            "no_block_failures",
        ):
            if not bool(aggregate_manifest.get(key)):
                failures.append(f"aggregate manifest check failed: {key}")
        if aggregate_manifest.get("status") != "complete":
            failures.append("aggregate status is not complete")
        return failures, warnings

    manifest = _read_json(output / "run_manifest.json")
    if not manifest:
        return ["run_manifest.json is missing"], warnings
    if manifest.get("status") != "complete":
        failures.append("run status is not complete")
    for name in FOCUSED_OUTPUTS:
        if not (output / name).exists():
            failures.append(f"required output is missing: {name}")

    metrics = _read_csv(output / "block_metrics.csv")
    if metrics.empty:
        failures.append("block_metrics.csv has no successful block rows")
    else:
        required = {
            "block_name", "covariance_moment", "assignment", "alpha", "delta_g",
            "endpoint_numerically_reliable", "ordering_inferentially_reliable", "reliable_ordering",
        }
        missing = sorted(required.difference(metrics.columns))
        if missing:
            failures.append("block_metrics.csv missing columns: " + ", ".join(missing))
        if "delta_g" in metrics and not all(math.isfinite(float(value)) for value in metrics["delta_g"]):
            failures.append("block_metrics.csv contains non-finite delta_g")
        if {"covariance_moment", "assignment", "alpha"}.issubset(metrics.columns):
            primary = metrics[
                (metrics["covariance_moment"].astype(str) == "centered")
                & (metrics["assignment"].astype(str) == "observed")
                & ((metrics["alpha"].astype(float) - 0.25).abs() <= 1.0e-12)
            ]
            if primary.empty:
                failures.append("primary centered observed alpha=0.25 endpoint is missing")

    captured = _read_csv(output / "block_failures.csv")
    if not captured.empty:
        failures.append(f"captured block failures: {len(captured)}")

    tier = str(manifest.get("experiment_tier", "debug"))
    if tier == "confirmatory":
        for name in ("bootstrap_metrics.csv", "interventions.csv", "alpha_sweep.csv", "damping_sweep.csv"):
            if _read_csv(output / name).empty:
                failures.append(f"confirmatory output is empty: {name}")
    elif tier == "screening":
        for name in ("bootstrap_metrics.csv", "interventions.csv", "alpha_sweep.csv", "damping_sweep.csv"):
            if not _read_csv(output / name).empty:
                warnings.append(f"screening run unexpectedly populated secondary control: {name}")

    if bool(manifest.get("scientific_run", False)):
        model = dict(manifest.get("model", {}))
        data = dict(manifest.get("data", {}))
        for key, value in (
            ("resolved_model_commit", model.get("resolved_model_commit")),
            ("dataset_revision", data.get("dataset_revision")),
            ("tokenizer_revision", data.get("tokenizer_revision")),
            ("source_order_sha256", data.get("source_order_sha256")),
            ("selected_chunk_content_sha256", data.get("selected_chunk_content_sha256")),
        ):
            if not value:
                failures.append(f"scientific provenance field is missing: {key}")
    return failures, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a focused frozen run or aggregate.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    failures, warnings = validate_run(args.output_dir)
    for warning in warnings:
        print(f"WARNING: {warning}")
    for failure in failures:
        print(f"ERROR: {failure}")
    print("PASS" if not failures else "FAIL")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
