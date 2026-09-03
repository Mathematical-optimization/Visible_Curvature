from __future__ import annotations
from pathlib import Path
import torch
from ovc_experiments.hardened_runner import HardenedBlockConfig, analyze_block_streaming
from ovc_experiments.safe_operators import DiagonalOperator


def gradient_factory():
    # Replace this replayable generator with the model's per-example block-gradient function.
    generator = torch.Generator().manual_seed(7)
    for _ in range(128):
        yield torch.randn(4, 4, generator=generator, dtype=torch.float64)


curvature = DiagonalOperator(torch.logspace(0, 2, 16, dtype=torch.float64))
config = HardenedBlockConfig(
    curvature_kind='ggn',
    shampoo_damping=1e-3,
    adam_damping=1e-8,
    subspace_policy='strict_spd',
)
result = analyze_block_streaming(
    curvature_operator=curvature,
    gradient_factory=gradient_factory,
    rows=4,
    cols=4,
    example_count=128,
    config=config,
    output_dir='../outputs/example',
    config_path=Path(__file__),
    metadata={'block_name': 'example.weight', 'checkpoint_step': 0, 'seed': 7},
)
print(dict(result.row))
