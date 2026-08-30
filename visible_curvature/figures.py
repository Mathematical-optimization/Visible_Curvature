from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required focused result table is missing or empty: {path}")
    return pd.read_csv(path)


def _accepted_controls(frame: pd.DataFrame) -> pd.DataFrame:
    if "balanced_reliable_for_inference" not in frame.columns:
        return frame
    values = frame["balanced_reliable_for_inference"]
    if values.dtype == bool:
        mask = values
    else:
        mask = values.astype(str).str.lower().isin({"true", "1", "yes"})
    return frame[mask].copy()


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_block_signed_gain(output_dir: str | Path, output_path: str | Path) -> Path:
    root = Path(output_dir)
    summary_path = root / "paired_seed_summary.csv"
    if summary_path.exists() and summary_path.stat().st_size:
        frame = pd.read_csv(summary_path)
        value = "delta_g_median"
    else:
        frame = _read(root / "block_metrics.csv")
        value = "delta_g"
    frame = frame[(frame["covariance_moment"] == "centered") & (frame["assignment"] == "observed")]
    frame = frame[np.isclose(frame["alpha"].astype(float), 0.25)]
    if frame.empty:
        raise ValueError("No centered observed alpha=0.25 rows for the primary block-gain figure")
    grouped = frame.groupby("block_name", as_index=False)[value].median().sort_values(value)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(grouped)), 4.0))
    ax.bar(np.arange(len(grouped)), grouped[value].to_numpy(dtype=float))
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(np.arange(len(grouped)), [str(v).split(".")[-2] for v in grouped["block_name"]], rotation=45, ha="right")
    ax.set_ylabel(r"$\Delta G=\log K_{\rm Adam}-\log K_{\rm Shampoo}$")
    ax.set_title("Frozen blockwise signed gain")
    return _save(fig, Path(output_path))


def plot_assignment_intervention(output_dir: str | Path, output_path: str | Path) -> Path:
    frame = _accepted_controls(_read(Path(output_dir) / "interventions.csv"))
    grouped = frame.groupby("assignment", as_index=False)["delta_g"].median()
    order = [name for name in ("observed", "aligned", "reversed") if name in set(grouped["assignment"])]
    grouped = grouped.set_index("assignment").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.bar(grouped["assignment"], grouped["delta_g"])
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel(r"median $\Delta G$")
    ax.set_title("Factor–curvature assignment intervention")
    return _save(fig, Path(output_path))


def plot_alpha_response(output_dir: str | Path, output_path: str | Path) -> Path:
    frame = _accepted_controls(_read(Path(output_dir) / "alpha_sweep.csv"))
    grouped = frame.groupby(["assignment", "alpha"], as_index=False)["delta_g"].median()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for assignment in ("observed", "aligned", "reversed"):
        subset = grouped[grouped["assignment"] == assignment].sort_values("alpha")
        if not subset.empty:
            ax.plot(subset["alpha"], subset["delta_g"], marker="o", label=assignment)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"factor utilization exponent $\alpha$")
    ax.set_ylabel(r"median $\Delta G$")
    ax.set_title("Utilization-strength response")
    ax.legend()
    return _save(fig, Path(output_path))


def plot_damping_attenuation(output_dir: str | Path, output_path: str | Path) -> Path:
    frame = _accepted_controls(_read(Path(output_dir) / "damping_sweep.csv"))
    group_keys = ["assignment", "damping_coefficient"]
    if "sweep_mode" in frame.columns:
        group_keys.insert(0, "sweep_mode")
    grouped = frame.groupby(group_keys, as_index=False)["delta_g"].median()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    modes = (
        list(dict.fromkeys(grouped["sweep_mode"].astype(str)))
        if "sweep_mode" in grouped.columns
        else [""]
    )
    for mode in modes:
        mode_frame = (
            grouped[grouped["sweep_mode"].astype(str) == mode]
            if mode
            else grouped
        )
        for assignment in ("observed", "aligned", "reversed"):
            subset = mode_frame[mode_frame["assignment"] == assignment].sort_values("damping_coefficient")
            if not subset.empty:
                label = f"{mode}: {assignment}" if mode else assignment
                ax.plot(subset["damping_coefficient"], subset["delta_g"].abs(), marker="o", label=label)
    ax.set_xscale("symlog", linthresh=1.0e-4)
    ax.set_xlabel("normalized damping coefficient")
    ax.set_ylabel(r"median $|\Delta G|$")
    ax.set_title("Damping attenuation")
    ax.legend()
    return _save(fig, Path(output_path))


def make_all_frozen_figures(output_dir: str | Path, figure_dir: str | Path | None = None) -> list[Path]:
    root = Path(output_dir)
    target = Path(figure_dir) if figure_dir is not None else root / "figures"
    return [
        plot_block_signed_gain(root, target / "block_signed_gain.pdf"),
        plot_assignment_intervention(root, target / "assignment_intervention.pdf"),
        plot_alpha_response(root, target / "alpha_response.pdf"),
        plot_damping_attenuation(root, target / "damping_attenuation.pdf"),
    ]
