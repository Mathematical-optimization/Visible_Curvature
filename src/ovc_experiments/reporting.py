from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _ensure_parent(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def delta_gain_column(frame: pd.DataFrame, *, alpha: float = 0.25) -> str:
    """Return the canonical or legacy Shampoo-minus-Adam gain column."""

    preferred = (f"delta_G_{alpha:g}", "delta_G_0.25", "delta_G")
    for column in preferred:
        if column in frame.columns:
            return column
    fallback = sorted(column for column in frame.columns if column.startswith("delta_G_"))
    if fallback:
        return fallback[0]
    raise ValueError("Geometry frame has no delta_G column")


def plot_geometry_delta_gain(frame: pd.DataFrame, path: str | Path, *, alpha: float = 0.25) -> Path:
    destination = _ensure_parent(path)
    column = delta_gain_column(frame, alpha=alpha)
    values = frame[column].fillna(0.0).to_numpy()
    labels = frame["block_name"].astype(str).tolist()
    figure, axis = plt.subplots(figsize=(max(5.0, 0.55 * len(labels)), 3.4))
    axis.bar(np.arange(len(values)), values)
    axis.axhline(0.0, linewidth=1.0)
    axis.set_ylabel(r"$\Delta G_b$")
    axis.set_xlabel("parameter block")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    axis.set_title("Frozen Shampoo-form gain relative to Adam form")
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination


def plot_intervention_conditions(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = _ensure_parent(path)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    plotted = False
    for (block, intervention, branch), group in frame.groupby(
        ["block_name", "intervention", "branch"], dropna=False
    ):
        if intervention == "alpha":
            x = group["alpha"].to_numpy(dtype=float)
            label = f"{block}: {branch}, alpha"
        elif intervention == "damping":
            x = group["rho_over_min"].to_numpy(dtype=float)
            x = np.where(x == 0.0, 1e-8, x)
            label = f"{block}: {branch}, damping"
        else:
            continue
        order = np.argsort(x)
        axis.plot(x[order], group["condition_number"].to_numpy(dtype=float)[order], marker="o", label=label)
        if intervention == "damping":
            axis.set_xscale("log")
        plotted = True
    if not plotted:
        assignment = frame[frame["intervention"] == "assignment"]
        if not assignment.empty:
            labels = (assignment["block_name"].astype(str) + ":" + assignment["branch"].astype(str)).tolist()
            axis.bar(np.arange(len(assignment)), assignment["condition_number"].to_numpy(dtype=float))
            axis.set_xticks(np.arange(len(labels)))
            axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
            plotted = True
    axis.set_yscale("log")
    axis.set_ylabel("effective condition number")
    axis.set_xlabel("intervention value")
    axis.set_title("Assignment, utilization, and damping interventions")
    if plotted and len(frame) <= 80:
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination


def plot_dynamics_curves(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = _ensure_parent(path)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for (block, preconditioner, method), group in frame.groupby(
        ["block_name", "preconditioner", "method"]
    ):
        group = group.sort_values("iteration")
        y = group["relative_objective"].to_numpy(dtype=float)
        y = np.maximum(y, np.finfo(float).tiny)
        axis.plot(
            group["iteration"].to_numpy(dtype=int),
            y,
            label=f"{block}: {preconditioner}/{method}",
        )
    axis.set_yscale("log")
    axis.set_xlabel("iteration")
    axis.set_ylabel("relative quadratic objective")
    axis.set_title("Frozen local quadratic dynamics")
    if len(frame.groupby(["block_name", "preconditioner", "method"])) <= 16:
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination


def plot_synthetic_fan(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = _ensure_parent(path)
    paired = frame[frame["experiment"] == "flat_kron_pair"].copy()
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for (alpha, r), group in paired.groupby(["alpha", "r"]):
        group = group.sort_values("kappa")
        axis.plot(group["K_H"], group["K_plus"], marker="o", label=f"aligned a={alpha:g}, r={r:g}")
        axis.plot(group["K_H"], group["K_minus"], marker="x", label=f"reversed a={alpha:g}, r={r:g}")
    axis.plot(paired["K_H"].drop_duplicates().sort_values(), paired["K_H"].drop_duplicates().sort_values(), linestyle="--", label="scalar")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$K_H$")
    axis.set_ylabel("effective condition number")
    axis.set_title("Synthetic assignment-utilization fan")
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination


def summarize_hypotheses(
    geometry: pd.DataFrame,
    interventions: pd.DataFrame,
    *,
    sign_threshold: float,
) -> dict[str, Any]:
    """Return the canonical H1--H6 evaluation in the historical JSON shape."""

    from .hardened_reporting import evaluate_hypotheses, summary_dict

    table = evaluate_hypotheses(
        geometry,
        interventions,
        sign_tolerance=sign_threshold,
        h2_is_held_out=False,
    )
    return summary_dict(table)

def aggregate_geometry_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("No geometry files supplied")
    return pd.concat(frames, ignore_index=True, sort=False)



def plot_checkpoint_heatmap(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    alpha: float = 0.25,
) -> Path:
    destination = _ensure_parent(path)
    column = delta_gain_column(frame, alpha=alpha)
    pivot = frame.pivot_table(
        index="block_name",
        columns="checkpoint_step",
        values=column,
        aggfunc="mean",
    ).sort_index()
    figure, axis = plt.subplots(
        figsize=(max(5.0, 0.75 * max(1, pivot.shape[1])), max(2.8, 0.42 * max(1, pivot.shape[0])))
    )
    finite = pivot.to_numpy(dtype=float)
    maximum = float(np.nanmax(np.abs(finite))) if np.isfinite(finite).any() else 1.0
    maximum = max(maximum, np.finfo(float).eps)
    image = axis.imshow(finite, aspect="auto", vmin=-maximum, vmax=maximum, cmap="coolwarm")
    axis.set_xticks(np.arange(pivot.shape[1]))
    axis.set_xticklabels([str(value) for value in pivot.columns])
    axis.set_yticks(np.arange(pivot.shape[0]))
    axis.set_yticklabels([str(value) for value in pivot.index], fontsize=7)
    axis.set_xlabel("checkpoint step")
    axis.set_ylabel("parameter block")
    axis.set_title(r"Checkpoint evolution of $\Delta G_b$")
    figure.colorbar(image, ax=axis, label=r"$\Delta G_b$")
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination


def plot_staleness(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = _ensure_parent(path)
    figure, axis = plt.subplots(figsize=(6.8, 4.0))
    plotted = False
    if not frame.empty:
        for (block, target_step), group in frame.groupby(["block_name", "target_step"]):
            group = group.sort_values("checkpoint_lag")
            axis.plot(
                group["checkpoint_lag"].to_numpy(dtype=int),
                group["condition_ratio_to_fresh"].to_numpy(dtype=float),
                marker="o",
                label=f"{block} @ {target_step}",
            )
            plotted = True
    axis.axhline(1.0, linewidth=1.0, linestyle="--")
    axis.set_xlabel("checkpoint lag")
    axis.set_ylabel(r"$K_{\mathrm{stale}}/K_{\mathrm{fresh}}$")
    axis.set_title("Frozen Shampoo factor staleness")
    if plotted and len(frame.groupby(["block_name", "target_step"])) <= 16:
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination



def plot_continuation_curves(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = _ensure_parent(path)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for (block, preconditioner), group in frame.groupby(["block_name", "preconditioner"]):
        group = group.sort_values("iteration")
        values = np.maximum(
            group["relative_loss"].to_numpy(dtype=float), np.finfo(float).tiny
        )
        axis.plot(
            group["iteration"].to_numpy(dtype=int),
            values,
            marker="o",
            label=f"{block}: {preconditioner}",
        )
    axis.set_yscale("log")
    axis.set_xlabel("frozen continuation step")
    axis.set_ylabel("loss / initial loss")
    axis.set_title("Fixed-batch one-block continuations")
    if len(frame.groupby(["block_name", "preconditioner"])) <= 16:
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination

