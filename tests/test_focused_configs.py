from pathlib import Path

import yaml

from visible_curvature.config import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
EXPECTED = {
    "smoke.yaml",
    "synthetic_theory.yaml",
    "hf_opt125m_screening.yaml",
    "hf_opt125m_confirmatory.yaml",
    "balanced_reliability_smoke.yaml",
    "hf_opt125m_balanced_reliability.yaml",
}


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_only_focused_configs_are_shipped():
    assert {path.name for path in CONFIGS.glob("*.yaml")} == EXPECTED


def test_smoke_config_is_small_and_exercises_all_retained_controls():
    cfg = validate_config(_load("smoke.yaml"), mode="frozen")
    assert cfg["model"]["backend"] == "tiny_causal_lm"
    assert cfg["analysis"]["compute_tier"] == "debug"
    assert cfg["analysis"]["bootstrap"]["diagnostics"] == "delta_only"
    assert cfg["analysis"]["interventions"]["enabled"] is True
    assert cfg["analysis"]["alpha_sweep"]["values"] == [0.25, 0.5]
    assert cfg["analysis"]["damping_sweep"]["enabled"] is True
    assert cfg["analysis"]["damping_sweep"]["modes"] == [
        "joint",
        "shampoo_only",
    ]
    assert cfg["model"]["d_model"] <= 2
    assert cfg["analysis"]["covariance"]["num_batches"] <= 2
    assert cfg["analysis"]["curvature"]["lanczos_steps"] <= 4
    assert cfg["analysis"]["curvature"]["partial_trace_probes"] <= 2


def test_screening_disables_secondary_controls_and_bootstrap():
    cfg = validate_config(_load("hf_opt125m_screening.yaml"), mode="frozen")
    analysis = cfg["analysis"]
    assert cfg["scientific_run"] is True
    assert analysis["compute_tier"] == "screening"
    assert analysis["bootstrap"]["reps"] == 0
    assert analysis["interventions"]["enabled"] is False
    assert analysis["alpha_sweep"]["enabled"] is False
    assert analysis["damping_sweep"]["enabled"] is False
    assert cfg["blocks"]["max_blocks"] <= 4


def test_confirmatory_has_only_required_controls_and_bounded_budget():
    cfg = validate_config(_load("hf_opt125m_confirmatory.yaml"), mode="frozen")
    analysis = cfg["analysis"]
    assert cfg["scientific_run"] is True
    assert analysis["compute_tier"] == "confirmatory"
    assert analysis["bootstrap"]["reps"] == 100
    assert analysis["bootstrap"]["diagnostics"] == "delta_only"
    assert analysis["assignments"] == ["observed", "aligned", "reversed"]
    assert analysis["alpha_sweep"]["values"] == [0.25, 0.5]
    assert analysis["damping_sweep"]["coefficients"] == [0.0, 0.01, 0.1, 1.0]
    assert analysis["damping_sweep"]["modes"] == ["joint", "shampoo_only"]
    assert cfg["blocks"]["max_blocks"] <= 2
    assert analysis["curvature"]["lanczos_steps"] <= 96
    assert analysis["curvature"]["partial_trace_probes"] <= 48
