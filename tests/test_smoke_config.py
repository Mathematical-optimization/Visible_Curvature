from pathlib import Path

from visible_curvature.config import load_yaml
from visible_curvature.block_registry import discover_matrix_blocks
from visible_curvature.adapters import load_model_bundle
from visible_curvature.utils import get_device


def test_smoke_block_discovery():
    root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(root / 'configs' / 'smoke.yaml')
    device = get_device('cpu')
    bundle = load_model_bundle(cfg, device)
    bcfg = cfg['blocks']
    blocks = discover_matrix_blocks(bundle.model, include=bcfg['include'], max_blocks=bcfg['max_blocks'])
    assert len(blocks) == 1
