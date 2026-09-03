from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class PredictionResult:
    predictions: pd.DataFrame
    summary: dict[str, float | int]


def _validate_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _cluster_keys(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    _validate_columns(frame, columns)
    if len(columns) == 1:
        return frame[columns[0]].map(lambda value: (value,))
    return frame[list(columns)].apply(lambda row: tuple(row.tolist()), axis=1)


def _sign_labels(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))


def cluster_bootstrap_sign_fractions(
    frame: pd.DataFrame,
    *,
    delta_column: str,
    cluster_columns: Sequence[str],
    threshold: float,
    replicates: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    _validate_columns(frame, [delta_column, *cluster_columns])
    working = frame[[delta_column, *cluster_columns]].dropna(subset=[delta_column]).copy()
    if working.empty:
        raise ValueError("No finite gain values are available")
    keys = _cluster_keys(working, cluster_columns)
    working["__cluster__"] = keys
    clusters = list(dict.fromkeys(keys.tolist()))
    grouped = {key: working[working["__cluster__"] == key] for key in clusters}

    def fractions(sample: pd.DataFrame) -> np.ndarray:
        labels = _sign_labels(sample[delta_column].to_numpy(dtype=float), threshold)
        return np.array(
            [
                np.mean(labels > 0),
                np.mean(labels < 0),
                np.mean(labels == 0),
            ],
            dtype=float,
        )

    empirical = fractions(working)
    generator = np.random.default_rng(seed)
    draws = np.empty((replicates, 3), dtype=float)
    for replicate in range(replicates):
        sampled_keys = generator.choice(len(clusters), size=len(clusters), replace=True)
        sample = pd.concat(
            [grouped[clusters[int(index)]] for index in sampled_keys],
            ignore_index=True,
        )
        draws[replicate] = fractions(sample)

    tail = 0.5 * (1.0 - confidence)
    lower = np.quantile(draws, tail, axis=0)
    upper = np.quantile(draws, 1.0 - tail, axis=0)
    return {
        label: {
            "estimate": float(empirical[index]),
            "lower": float(lower[index]),
            "upper": float(upper[index]),
        }
        for index, label in enumerate(("positive", "negative", "neutral"))
    }


def paired_intervention_effects(
    frame: pd.DataFrame,
    *,
    intervention: str,
    reference_branch: str,
    treatment_branch: str,
    value_column: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    _validate_columns(
        frame,
        ["intervention", "branch", value_column, *group_columns],
    )
    subset = frame[
        (frame["intervention"] == intervention)
        & frame["branch"].isin([reference_branch, treatment_branch])
    ].copy()
    pivot = subset.pivot_table(
        index=list(group_columns),
        columns="branch",
        values=value_column,
        aggfunc="first",
    )
    if reference_branch not in pivot or treatment_branch not in pivot:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "reference_value",
                "treatment_value",
                "paired_difference",
                "paired_log_ratio",
            ]
        )
    paired = pivot[[reference_branch, treatment_branch]].dropna().reset_index()
    paired = paired.rename(
        columns={
            reference_branch: "reference_value",
            treatment_branch: "treatment_value",
        }
    )
    paired["paired_difference"] = (
        paired["treatment_value"] - paired["reference_value"]
    )
    positive = (paired["reference_value"] > 0) & (paired["treatment_value"] > 0)
    paired["paired_log_ratio"] = np.nan
    paired.loc[positive, "paired_log_ratio"] = np.log(
        paired.loc[positive, "treatment_value"]
        / paired.loc[positive, "reference_value"]
    )
    return paired


def leave_one_cluster_out_prediction(
    frame: pd.DataFrame,
    *,
    target_column: str,
    predictor_columns: Sequence[str],
    cluster_columns: Sequence[str],
    sign_threshold: float = 0.0,
) -> PredictionResult:
    if sign_threshold < 0:
        raise ValueError("sign_threshold must be nonnegative")
    required = [target_column, *predictor_columns, *cluster_columns]
    _validate_columns(frame, required)
    working = frame[required].dropna().copy()
    if working.empty:
        raise ValueError("No complete rows are available for prediction")
    working["__cluster__"] = _cluster_keys(working, cluster_columns)
    clusters = list(dict.fromkeys(working["__cluster__"].tolist()))
    if len(clusters) < 2:
        raise ValueError("At least two clusters are required")

    predicted_parts: list[pd.DataFrame] = []
    parameter_count = len(predictor_columns) + 1
    for cluster in clusters:
        train = working[working["__cluster__"] != cluster]
        test = working[working["__cluster__"] == cluster]
        if len(train) < parameter_count:
            raise ValueError("Insufficient training rows for leave-one-cluster-out fit")
        train_design = np.column_stack(
            [
                np.ones(len(train), dtype=float),
                train[list(predictor_columns)].to_numpy(dtype=float),
            ]
        )
        coefficients = np.linalg.lstsq(
            train_design,
            train[target_column].to_numpy(dtype=float),
            rcond=None,
        )[0]
        test_design = np.column_stack(
            [
                np.ones(len(test), dtype=float),
                test[list(predictor_columns)].to_numpy(dtype=float),
            ]
        )
        part = test.copy()
        part["prediction"] = test_design @ coefficients
        predicted_parts.append(part)

    predictions = pd.concat(predicted_parts, ignore_index=True)
    observed = predictions[target_column].to_numpy(dtype=float)
    predicted = predictions["prediction"].to_numpy(dtype=float)
    correlation = float(spearmanr(observed, predicted).statistic)
    if not math.isfinite(correlation):
        correlation = math.nan
    observed_sign = _sign_labels(observed, sign_threshold)
    predicted_sign = _sign_labels(predicted, sign_threshold)
    sign_accuracy = float(np.mean(observed_sign == predicted_sign))
    root_mean_square_error = float(np.sqrt(np.mean((observed - predicted) ** 2)))
    predictions = predictions.drop(columns=["__cluster__"])
    return PredictionResult(
        predictions=predictions,
        summary={
            "rows": int(len(predictions)),
            "clusters": int(len(clusters)),
            "spearman": correlation,
            "sign_accuracy": sign_accuracy,
            "rmse": root_mean_square_error,
        },
    )
