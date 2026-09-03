"""Experiment toolkit for optimizer-visible curvature."""

from .blocks import BlockSpec, MatrixLayout, discover_matrix_blocks
from .config import ExperimentConfig, load_config, save_config

__all__ = [
    "BlockSpec",
    "ExperimentConfig",
    "MatrixLayout",
    "discover_matrix_blocks",
    "load_config",
    "save_config",
]

__version__ = "1.2.0"

# OVC_HARDENED_EXPORTS_V1_2
from .curvature_policy import CurvaturePolicy, validate_curvature_policy
from .hardened_runner import HardenedBlockConfig, HardenedBlockResult, analyze_block_streaming
from .streaming_moments import MatrixMomentStatistics, StreamingMatrixMoments, accumulate_matrix_moments
from .theorem_validation import validate_flat_kron_pair, validate_weighted_chebyshev

from .functional_safe import functional_call_tied, tied_parameter_groups
from .paths import resolve_output_dir
from .hardened_reporting import evaluate_h2_leave_one_cluster_out
