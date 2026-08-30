import pytest

from visible_curvature.config import validate_config


def _valid_scientific_config():
    return {
        "scientific_run": True,
        "model": {
            "backend": "hf_causal_lm",
            "revision": "a" * 40,
        },
        "data": {
            "backend": "hf_text",
            "revision": "b" * 40,
            "tokenizer_revision": "c" * 40,
            "order_seed": 1729,
        },
        "analysis": {
            "compute_tier": "confirmatory",
            "covariance": {"skip_batches": 0, "num_batches": 16},
            "curvature": {"skip_batches": 16, "num_batches": 2},
            "bootstrap": {"reps": 100},
            "assignments": ["observed", "aligned", "reversed"],
            "alpha_sweep": {"values": [0.25, 0.5]},
        },
    }


def test_scientific_hf_run_requires_full_immutable_revisions():
    cfg = _valid_scientific_config()
    assert validate_config(cfg, mode="frozen")["data"]["order_seed"] == 1729

    for path in (("model", "revision"), ("data", "revision"), ("data", "tokenizer_revision")):
        bad = _valid_scientific_config()
        bad[path[0]][path[1]] = "main"
        with pytest.raises(ValueError, match="40-character"):
            validate_config(bad, mode="frozen")


def test_scientific_confirmatory_requires_100_delta_only_bootstrap_replicates():
    cfg = _valid_scientific_config()
    cfg["analysis"]["bootstrap"]["reps"] = 99
    with pytest.raises(ValueError, match="at least 100"):
        validate_config(cfg, mode="frozen")


def test_removed_modes_and_gold_tier_are_rejected():
    cfg = _valid_scientific_config()
    cfg["analysis"]["compute_tier"] = "gold"
    with pytest.raises(ValueError, match="compute_tier"):
        validate_config(cfg, mode="frozen")
    with pytest.raises(ValueError, match="mode must be 'frozen'"):
        validate_config(cfg, mode="online")
