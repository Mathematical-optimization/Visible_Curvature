from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


_HYPOTHESES = ("H1", "H2", "H3", "H4", "H5", "H6")
_ID_COLUMNS = (
    "run_id",
    "run_name",
    "architecture",
    "model_family",
    "model",
    "model_name",
    "model_scale",
    "seed",
    "checkpoint_step",
    "checkpoint",
    "block_name",
)


def _column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _wide_column(
    frame: pd.DataFrame,
    *,
    exact: Sequence[str],
    prefixes: Sequence[str] = (),
    preferred_suffix: str = "0.25",
) -> str | None:
    direct = _column(frame, exact)
    if direct is not None:
        return direct
    for prefix in prefixes:
        preferred = f"{prefix}{preferred_suffix}"
        if preferred in frame.columns:
            return preferred
        candidates = sorted(column for column in frame.columns if column.startswith(prefix))
        if candidates:
            return candidates[0]
    return None


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.where(np.isfinite(values))


def _filter_censored(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out = out[~out[column].fillna(True).astype(bool)]
    return out


def _geometry_panel(frame: pd.DataFrame, delta_column: str | None) -> pd.DataFrame:
    relevant = ["censored", "curvature_censored", "adam_censored"]
    if delta_column and delta_column.startswith("delta_G_"):
        alpha_suffix = delta_column.removeprefix("delta_G_")
        relevant.extend(
            [
                f"shampoo_{alpha_suffix}_censored",
                f"shampoo_censored_{alpha_suffix}",
            ]
        )
    else:
        relevant.extend(["shampoo_censored", "shampoo_0.25_censored"])
    return _filter_censored(frame, relevant)


def _intervention_panel(frame: pd.DataFrame) -> pd.DataFrame:
    return _filter_censored(
        frame,
        (
            "censored",
            "curvature_censored",
            "effective_censored",
            "condition_censored",
        ),
    )


def _group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in _ID_COLUMNS if column in frame.columns]


def _spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan")
    return float(pair["x"].corr(pair["y"], method="spearman"))


def _row(hypothesis: str, status: str, metric: str, **evidence: Any) -> dict[str, Any]:
    return {
        "hypothesis": hypothesis,
        "status": status,
        "metric": metric,
        "evidence": evidence,
    }


def _status_from_trials(passed: int, total: int) -> str:
    if total == 0:
        return "insufficient_data"
    return "supported" if passed == total else "not_supported"


def _subset_intervention(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    intervention_column = _column(frame, ("intervention", "intervention_type", "kind"))
    if intervention_column is None:
        return frame.copy()
    return frame[frame[intervention_column].astype(str).str.lower() == name.lower()].copy()


def _signed_assignment_metric(frame: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    delta_column = _wide_column(
        frame,
        exact=("delta_G", "delta_gain", "DeltaG", "delta_g"),
        prefixes=("delta_G_", "delta_gain_"),
    )
    if delta_column is not None:
        return _finite_series(frame, delta_column), delta_column

    gain_column = _wide_column(
        frame,
        exact=("G_shampoo", "gain_shampoo", "shampoo_gain", "gain"),
        prefixes=("G_shampoo_", "gain_shampoo_"),
    )
    if gain_column is None:
        return None, None
    shampoo_gain = _finite_series(frame, gain_column)
    adam_column = _wide_column(
        frame,
        exact=("G_adam", "gain_adam", "adam_gain"),
        prefixes=("G_adam_", "gain_adam_"),
    )
    if adam_column is not None:
        return shampoo_gain - _finite_series(frame, adam_column), f"{gain_column}-{adam_column}"
    return shampoo_gain, gain_column


def evaluate_hypotheses(
    geometry: pd.DataFrame,
    interventions: pd.DataFrame | None = None,
    *,
    correlation_threshold: float = 0.3,
    sign_tolerance: float = 1e-8,
    h2_is_held_out: bool = False,
) -> pd.DataFrame:
    """Evaluate H1--H6 with the paper's signed, paired semantics.

    The function accepts both the package's historical wide tables
    (``delta_G_0.25``/``G_shampoo_0.25``) and canonical long tables. Rows
    censored for unrelated optional operators are not removed from the primary
    Adam--Shampoo panel.
    """

    rows: list[dict[str, Any]] = []
    delta_column = _wide_column(
        geometry,
        exact=("delta_G", "delta_gain", "DeltaG", "delta_g", "delta_G_0.25"),
        prefixes=("delta_G_", "delta_gain_"),
    )
    primary = _geometry_panel(geometry, delta_column)

    # H1: both signs recur in valid primary rows.
    if delta_column is None:
        rows.append(_row("H1", "insufficient_data", "no delta_G column"))
    else:
        delta = _finite_series(primary, delta_column).dropna()
        positive = int((delta > sign_tolerance).sum())
        negative = int((delta < -sign_tolerance).sum())
        status = (
            "insufficient_data"
            if len(delta) == 0
            else ("supported" if positive > 0 and negative > 0 else "not_supported")
        )
        rows.append(
            _row(
                "H1",
                status,
                f"positive={positive}, negative={negative}, valid={len(delta)}",
                positive=positive,
                negative=negative,
                valid_rows=int(len(delta)),
                threshold=sign_tolerance,
                delta_column=delta_column,
            )
        )

    # H2: direction is positive, never absolute correlation. In-sample evidence
    # is descriptive; only held-out evaluation may be confirmatory support.
    response_column = _wide_column(
        primary,
        exact=(
            "predicted_signed_response",
            "visible_response_score",
            "response_score",
            "response_shampoo",
        ),
    )
    if delta_column is None or response_column is None:
        rows.append(_row("H2", "insufficient_data", "missing response or delta_G column"))
    else:
        correlation = _spearman(
            _finite_series(primary, response_column),
            _finite_series(primary, delta_column),
        )
        if not math.isfinite(correlation):
            status = "insufficient_data"
        elif correlation < correlation_threshold:
            status = "not_supported"
        elif h2_is_held_out:
            status = "supported"
        else:
            status = "descriptive_only"
        rows.append(
            _row(
                "H2",
                status,
                f"spearman={correlation:.6g}; held_out={h2_is_held_out}",
                spearman=correlation,
                held_out=h2_is_held_out,
                threshold=correlation_threshold,
                response_column=response_column,
                delta_column=delta_column,
            )
        )

    iv = _intervention_panel(interventions) if interventions is not None else pd.DataFrame()

    # H3: aligned positive and reversed negative within every paired unit.
    assignment = _subset_intervention(iv, "assignment") if not iv.empty else iv
    branch_column = _column(assignment, ("branch", "assignment", "intervention_branch"))
    signed_metric, signed_metric_name = _signed_assignment_metric(assignment)
    h3_passed = h3_total = 0
    if branch_column is not None and signed_metric is not None:
        assignment = assignment.assign(_signed_metric=signed_metric)
        groups = _group_columns(assignment)
        iterator = assignment.groupby(groups, dropna=False) if groups else [((), assignment)]
        for _, group in iterator:
            labels = group[branch_column].astype(str).str.lower()
            aligned = group.loc[labels.str.contains("align"), "_signed_metric"].dropna()
            reversed_values = group.loc[labels.str.contains("revers"), "_signed_metric"].dropna()
            if aligned.empty or reversed_values.empty:
                continue
            h3_total += 1
            h3_passed += int(
                float(aligned.mean()) > sign_tolerance
                and float(reversed_values.mean()) < -sign_tolerance
            )
    rows.append(
        _row(
            "H3",
            _status_from_trials(h3_passed, h3_total),
            f"signed_reversals={h3_passed}/{h3_total}",
            passed=h3_passed,
            total=h3_total,
            metric_column=signed_metric_name,
        )
    )

    # H4: factor damping contracts |G_shampoo|. With fixed Adam, delta_G tends
    # to -G_adam rather than zero in general.
    damping = _subset_intervention(iv, "damping") if not iv.empty else iv
    damping_column = _column(
        damping,
        (
            "rho_over_min",
            "normalized_damping",
            "damping_ratio",
            "rho_left_over_min",
            "damping",
            "rho",
        ),
    )
    gain_column = _wide_column(
        damping,
        exact=("G_shampoo", "gain_shampoo", "shampoo_gain", "gain"),
        prefixes=("G_shampoo_", "gain_shampoo_"),
    )
    h4_passed = h4_total = 0
    if damping_column is not None and gain_column is not None:
        curve_groups = _group_columns(damping)
        for extra in ("branch", "alpha", "moment_kind", "curvature_kind"):
            if extra in damping.columns and extra not in curve_groups:
                curve_groups.append(extra)
        iterator = damping.groupby(curve_groups, dropna=False) if curve_groups else [((), damping)]
        for _, group in iterator:
            curve = group.assign(
                _damping=_finite_series(group, damping_column),
                _gain=_finite_series(group, gain_column),
            ).dropna(subset=["_damping", "_gain"])
            curve = curve.sort_values("_damping")
            if len(curve) < 2 or float(curve["_damping"].iloc[-1]) <= float(curve["_damping"].iloc[0]):
                continue
            h4_total += 1
            h4_passed += int(
                abs(float(curve["_gain"].iloc[-1]))
                <= abs(float(curve["_gain"].iloc[0])) + sign_tolerance
            )
    rows.append(
        _row(
            "H4",
            _status_from_trials(h4_passed, h4_total),
            f"|G_shampoo| contracted={h4_passed}/{h4_total}; delta_G tends to -G_adam if Adam is fixed",
            passed=h4_passed,
            total=h4_total,
            damping_column=damping_column,
            gain_column=gain_column,
        )
    )

    # H5: alpha=1/2 amplifies the alpha=1/4 signed gain in both regimes.
    alpha_frame = _subset_intervention(iv, "alpha") if not iv.empty else iv
    alpha_column = _column(alpha_frame, ("alpha", "exponent", "shampoo_exponent"))
    alpha_gain_column = _wide_column(
        alpha_frame,
        exact=("G_shampoo", "gain_shampoo", "shampoo_gain", "gain"),
        prefixes=("G_shampoo_", "gain_shampoo_"),
    )
    h5_passed = h5_total = favorable = unfavorable = 0
    if alpha_column is not None and alpha_gain_column is not None:
        curve_groups = _group_columns(alpha_frame)
        for extra in ("branch", "moment_kind", "curvature_kind", "damping_left", "damping_right"):
            if extra in alpha_frame.columns and extra not in curve_groups:
                curve_groups.append(extra)
        iterator = alpha_frame.groupby(curve_groups, dropna=False) if curve_groups else [((), alpha_frame)]
        for _, group in iterator:
            alpha_values = _finite_series(group, alpha_column)
            gains = _finite_series(group, alpha_gain_column)
            valid = group.assign(_alpha=alpha_values, _gain=gains).dropna(subset=["_alpha", "_gain"])
            a25 = valid[np.isclose(valid["_alpha"], 0.25)]
            a50 = valid[np.isclose(valid["_alpha"], 0.50)]
            if a25.empty or a50.empty:
                continue
            gain25 = float(a25["_gain"].mean())
            gain50 = float(a50["_gain"].mean())
            if abs(gain25) <= sign_tolerance:
                continue
            h5_total += 1
            favorable += int(gain25 > sign_tolerance)
            unfavorable += int(gain25 < -sign_tolerance)
            same_sign = gain25 * gain50 > 0
            amplified = abs(gain50) > abs(gain25) + sign_tolerance
            h5_passed += int(same_sign and amplified)
    h5_status = (
        "insufficient_data"
        if h5_total == 0
        else (
            "supported"
            if h5_passed == h5_total and favorable > 0 and unfavorable > 0
            else "not_supported"
        )
    )
    rows.append(
        _row(
            "H5",
            h5_status,
            f"alpha_1/2_amplified={h5_passed}/{h5_total}, favorable={favorable}, unfavorable={unfavorable}",
            passed=h5_passed,
            total=h5_total,
            favorable=favorable,
            unfavorable=unfavorable,
            gain_column=alpha_gain_column,
        )
    )

    # H6: scalar grafting leaves condition number invariant within each full
    # experimental unit; checkpoints must never be pooled into one baseline.
    grafting = _subset_intervention(iv, "grafting") if not iv.empty else iv
    scale_column = _column(grafting, ("graft_scale", "scalar_graft", "scale"))
    condition_column = _wide_column(
        grafting,
        exact=("K_shampoo", "condition_shampoo", "condition_number"),
        prefixes=("K_shampoo_",),
    )
    h6_passed = h6_total = 0
    maximum_relative_error = float("nan")
    if scale_column is not None and condition_column is not None:
        curve_groups = _group_columns(grafting)
        for extra in ("branch", "preconditioner", "alpha"):
            if extra in grafting.columns and extra not in curve_groups:
                curve_groups.append(extra)
        errors: list[float] = []
        iterator = grafting.groupby(curve_groups, dropna=False) if curve_groups else [((), grafting)]
        for _, group in iterator:
            scales = _finite_series(group, scale_column)
            conditions = _finite_series(group, condition_column)
            valid = group.assign(_scale=scales, _condition=conditions).dropna(
                subset=["_scale", "_condition"]
            )
            if valid["_scale"].nunique() < 2:
                continue
            values = valid["_condition"].to_numpy(dtype=float)
            relative = float((values.max() - values.min()) / max(abs(values.mean()), 1e-300))
            errors.append(relative)
            h6_total += 1
            h6_passed += int(relative <= 1e-8)
        if errors:
            maximum_relative_error = max(errors)
    rows.append(
        _row(
            "H6",
            _status_from_trials(h6_passed, h6_total),
            f"checkpoint_safe_invariance={h6_passed}/{h6_total}; max_relative_error={maximum_relative_error:.6g}",
            passed=h6_passed,
            total=h6_total,
            max_relative_error=maximum_relative_error,
        )
    )

    table = pd.DataFrame(rows)
    # Keep a stable order even when individual hypotheses have insufficient data.
    table["hypothesis"] = pd.Categorical(table["hypothesis"], categories=_HYPOTHESES, ordered=True)
    return table.sort_values("hypothesis").reset_index(drop=True)


def summary_dict(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Convert the canonical evaluation table to the legacy JSON shape."""

    output: dict[str, dict[str, Any]] = {}
    for _, record in table.iterrows():
        hypothesis = str(record["hypothesis"])
        evidence = record.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {"metric": str(record.get("metric", ""))}
        evidence = {**evidence, "metric": str(record.get("metric", ""))}
        output[hypothesis] = {
            "status": str(record["status"]),
            "evidence": evidence,
        }
    for hypothesis in _HYPOTHESES:
        output.setdefault(
            hypothesis,
            {"status": "insufficient_data", "evidence": {"metric": "not evaluated"}},
        )
    return output


def patch_legacy_summary(legacy: pd.DataFrame, hardened: pd.DataFrame) -> pd.DataFrame:
    """Compatibility helper retained for external callers of v1.1.0."""

    if not isinstance(legacy, pd.DataFrame) or legacy.empty:
        return hardened
    hypothesis_column = _column(legacy, ("hypothesis", "name", "id"))
    status_column = _column(legacy, ("status", "supported", "result"))
    detail_column = _column(legacy, ("metric", "detail", "details", "summary"))
    if hypothesis_column is None or status_column is None:
        return hardened
    out = legacy.copy()
    for _, record in hardened.iterrows():
        mask = out[hypothesis_column].astype(str).str.upper() == str(record["hypothesis"]).upper()
        if not mask.any():
            new = {column: None for column in out.columns}
            new[hypothesis_column] = record["hypothesis"]
            new[status_column] = record["status"]
            if detail_column:
                new[detail_column] = record["metric"]
            out = pd.concat([out, pd.DataFrame([new])], ignore_index=True)
        else:
            out.loc[mask, status_column] = record["status"]
            if detail_column:
                out.loc[mask, detail_column] = record["metric"]
    return out


def evaluate_h2_leave_one_cluster_out(
    frame: pd.DataFrame,
    *,
    predictor: str,
    target: str,
    cluster_columns: Sequence[str] = ("model_name", "seed", "checkpoint_step"),
    correlation_threshold: float = 0.3,
) -> dict[str, Any]:
    """Linear held-out prediction across prespecified model/seed/checkpoint clusters."""

    data = _geometry_panel(frame, target).dropna(subset=[predictor, target]).copy()
    columns = [column for column in cluster_columns if column in data.columns]
    if not columns:
        raise ValueError("held-out H2 requires at least one cluster column")
    data["_cluster"] = data[columns].astype(str).agg("|".join, axis=1)
    predictions = pd.Series(index=data.index, dtype=float)
    for cluster in data["_cluster"].unique():
        train = data[data["_cluster"] != cluster]
        test = data[data["_cluster"] == cluster]
        if len(train) < 3 or train[predictor].nunique() < 2:
            continue
        design = np.column_stack((np.ones(len(train)), train[predictor].to_numpy(dtype=float)))
        beta, *_ = np.linalg.lstsq(design, train[target].to_numpy(dtype=float), rcond=None)
        predictions.loc[test.index] = beta[0] + beta[1] * test[predictor].to_numpy(dtype=float)
    valid = predictions.notna()
    rho = _spearman(predictions[valid], data.loc[valid, target])
    return {
        "hypothesis": "H2",
        "status": (
            "supported"
            if math.isfinite(rho) and rho >= correlation_threshold
            else "not_supported"
        ),
        "spearman": rho,
        "held_out_rows": int(valid.sum()),
        "clusters": int(data["_cluster"].nunique()),
        "predictions": predictions,
    }
