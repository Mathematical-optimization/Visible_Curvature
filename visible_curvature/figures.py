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


def _alpha_plot_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Aggregate the signed alpha response relative to practical alpha=1/4."""
    value = (
        "alpha_delta_from_practical"
        if "alpha_delta_from_practical" in frame.columns
        else "delta_g"
    )
    grouped = frame.groupby(["assignment", "alpha"], as_index=False)[value].median()
    return grouped, value


def _damping_plot_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each damping mode using its explicitly declared estimand."""
    work = frame.copy()
    if "sweep_mode" not in work.columns:
        work["sweep_mode"] = "joint"
    if "control_estimand" not in work.columns:
        work["control_estimand"] = np.where(
            work["sweep_mode"].astype(str).eq("shampoo_only")
            & ("G_shampoo" in work.columns),
            "abs_g_shampoo",
            "abs_delta_g",
        )
    if "control_value" not in work.columns:
        values = []
        for _, row in work.iterrows():
            if str(row.get("sweep_mode", "joint")) == "shampoo_only" and "G_shampoo" in work.columns:
                values.append(abs(float(row["G_shampoo"])))
            else:
                values.append(abs(float(row["delta_g"])))
        work["control_value"] = values
    group_keys = [
        "sweep_mode",
        "assignment",
        "damping_coefficient",
        "control_estimand",
    ]
    return work.groupby(group_keys, as_index=False)["control_value"].median()



def plot_alpha_response(output_dir: str | Path, output_path: str | Path) -> Path:
    frame = _accepted_controls(_read(Path(output_dir) / "alpha_sweep.csv"))
    grouped, value = _alpha_plot_data(frame)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for assignment in ("observed", "aligned", "reversed"):
        subset = grouped[grouped["assignment"] == assignment].sort_values("alpha")
        if not subset.empty:
            ax.plot(subset["alpha"], subset[value], marker="o", label=assignment)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"factor utilization exponent $\alpha$")
    ylabel = (
        r"median $\Delta G(\alpha)-\Delta G(1/4)$"
        if value == "alpha_delta_from_practical"
        else r"median $\Delta G$ (legacy rows)"
    )
    ax.set_ylabel(ylabel)
    ax.set_title("Signed utilization-strength response")
    ax.legend()
    return _save(fig, Path(output_path))


def plot_damping_attenuation(output_dir: str | Path, output_path: str | Path) -> Path:
    frame = _accepted_controls(_read(Path(output_dir) / "damping_sweep.csv"))
    grouped = _damping_plot_data(frame)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    modes = list(dict.fromkeys(grouped["sweep_mode"].astype(str)))
    for mode in modes:
        mode_frame = grouped[grouped["sweep_mode"].astype(str) == mode]
        for assignment in ("observed", "aligned", "reversed"):
            subset = mode_frame[mode_frame["assignment"] == assignment].sort_values("damping_coefficient")
            if not subset.empty:
                estimand = str(subset["control_estimand"].iloc[0])
                label = f"{mode}: {assignment} ({estimand})"
                ax.plot(subset["damping_coefficient"], subset["control_value"], marker="o", label=label)
    ax.set_xscale("symlog", linthresh=1.0e-4)
    ax.set_xlabel("normalized damping coefficient")
    ax.set_ylabel(r"median declared attenuation target")
    ax.set_title("Theory-aligned damping attenuation")
    ax.legend(fontsize="small")
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
