from pathlib import Path

import visible_curvature


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_and_dependencies_are_focused():
    assert visible_curvature.__version__ == "1.3.0"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert 'version = "1.3.0"' in pyproject
    assert "ViT" not in pyproject
    assert "timm" not in pyproject + requirements
    assert "torchvision" not in pyproject + requirements


def test_public_scripts_are_only_the_focused_workflow():
    expected = {
        "run_synthetic_theory.py",
        "run_frozen_analysis.py",
        "validate_run.py",
        "aggregate_runs.py",
        "make_figures.py",
        "export_paper_assets.py",
        "run_balanced_reliability.py",
        "validate_balanced_run.py",
        "summarize_balanced_results.py",
        "make_balanced_policies.py",
    }
    assert {path.name for path in (ROOT / "scripts").glob("*.py")} == expected


def test_public_docs_do_not_advertise_removed_experiment_families():
    for name in (
        "README.md",
        "QUICKSTART_KO.md",
        "EXPERIMENT_PROTOCOL_KO.md",
        "AUTHOR_RUN_CHECKLIST_KO.md",
        "PACKAGE_MANIFEST.md",
    ):
        text = (ROOT / name).read_text(encoding="utf-8").lower()
        for forbidden in ("matched online", "run_online.py", "vit", "gold tier", "frozen_trajectories.csv"):
            assert forbidden not in text, f"{forbidden!r} remains in {name}"


def test_smoke_script_uses_exact_debug_watermark_contract():
    text = (ROOT / "reproduce_smoke.sh").read_text(encoding="utf-8")
    assert "run_synthetic_theory.py" in text
    assert "run_frozen_analysis.py" in text
    assert "integrated_theorem3_all_checks_passed" in text
    assert "DEBUG EXPORT -- NOT SCIENTIFIC EVIDENCE" in text
    assert "run_online.py" not in text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" in text


def test_release_has_no_legacy_plan_or_smoke_wrapper():
    forbidden = (
        ROOT / "visible_curvature" / "smoke.py",
        ROOT / "docs" / "superpowers" / "plans" / "2026-08-27-visible-curvature-v0.5.0.md",
        ROOT / "docs" / "superpowers" / "specs" / "2026-08-27-visible-curvature-v0.5.0-design.md",
    )
    assert not [str(path.relative_to(ROOT)) for path in forbidden if path.exists()]


def test_protocol_hash_excludes_removed_or_run_only_top_level_sections():
    from visible_curvature.utils import protocol_config_hash

    focused = {
        "data": {"backend": "synthetic_tokens", "order_seed": 7},
        "blocks": {"include": ["q_proj"]},
        "analysis": {"compute_tier": "screening"},
    }
    decorated = {
        **focused,
        "online": {"arm": "removed"},
        "model": {"name": "checkpoint-A"},
        "seed": 123,
        "output_dir": "somewhere",
    }
    decorated["analysis"] = {
        **focused["analysis"],
        "curvature": {"shift_overrides_path": "/seed-specific/path.json"},
    }
    focused["analysis"] = {
        **focused["analysis"],
        "curvature": {},
    }
    assert protocol_config_hash(focused) == protocol_config_hash(decorated)


def test_release_does_not_bundle_generated_validation_artifacts():
    assert not (ROOT / "validation_artifacts").exists()
