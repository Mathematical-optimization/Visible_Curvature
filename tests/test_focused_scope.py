from pathlib import Path

import pytest
import torch

from visible_curvature.adapters import load_model_bundle
from visible_curvature.data import build_dataloader_factory


ROOT = Path(__file__).resolve().parents[1]


def test_release_contains_no_online_or_image_entrypoints():
    forbidden = [
        ROOT / "visible_curvature" / "online_optim.py",
        ROOT / "visible_curvature" / "online_runner.py",
        ROOT / "visible_curvature" / "provenance.py",
        ROOT / "scripts" / "run_online.py",
        ROOT / "configs" / "timm_vit_frozen_template.yaml",
    ]
    assert not [str(path.relative_to(ROOT)) for path in forbidden if path.exists()]
    assert not list((ROOT / "configs").glob("online*.yaml"))
    assert not list((ROOT / "configs").glob("*online*.yaml"))


def test_model_adapter_rejects_removed_image_backends_before_importing_dependencies():
    with pytest.raises(ValueError, match="Supported model backends"):
        load_model_bundle({"model": {"backend": "timm_classifier", "name": "vit_tiny_patch16_224"}}, torch.device("cpu"))
    with pytest.raises(ValueError, match="Supported model backends"):
        load_model_bundle({"model": {"backend": "hf_image_classifier", "name": "fake/image"}}, torch.device("cpu"))


def test_data_factory_rejects_removed_image_backends_before_importing_dependencies():
    for backend in ("synthetic_images", "imagefolder", "cifar10"):
        with pytest.raises(ValueError, match="Supported data backends"):
            build_dataloader_factory(
                {"data": {"backend": backend}},
                model_backend="tiny_causal_lm",
                seed=0,
            )
