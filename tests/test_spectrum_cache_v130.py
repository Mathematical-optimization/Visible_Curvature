from __future__ import annotations

import torch

from visible_curvature.analysis_runner import (
    _LanczosSpectrumCache,
    _evaluate_preconditioner_pair,
)
from visible_curvature.curvature import LinearMatrixOperator


class _IdentityPreconditioner:
    def apply_half(self, matrix: torch.Tensor) -> torch.Tensor:
        return matrix


def _spectrum(call_index: int, *, steps: int, starts: int) -> dict:
    return {
        "min_ritz": 1.0,
        "max_ritz": 2.0 + 0.01 * call_index,
        "min_ritz_residual": 0.0,
        "max_ritz_residual": 0.0,
        "steps": steps,
        "starts": starts,
    }


def test_pair_evaluation_reuses_cached_adam_and_shampoo_spectra(monkeypatch):
    calls: list[int] = []

    def fake_lanczos(_matvec, _dim, *, steps, device, dtype, starts):
        del device, dtype
        calls.append(len(calls))
        return _spectrum(len(calls), steps=steps, starts=len(starts))

    monkeypatch.setattr(
        "visible_curvature.analysis_runner.multi_start_lanczos", fake_lanczos
    )
    operator = LinearMatrixOperator(
        (1, 1), lambda matrix: matrix, torch.device("cpu"), torch.float32
    )
    preconditioner = _IdentityPreconditioner()
    starts = [torch.ones(1)]
    cache = _LanczosSpectrumCache()

    first = _evaluate_preconditioner_pair(
        operator,
        preconditioner,
        preconditioner,
        starts=starts,
        steps=4,
        condition_cfg={"relative_floor": 1.0e-8, "fallback_tau": 1.0e-4},
        spectrum_cache=cache,
        adam_cache_key=("adam", "centered", "base"),
        shampoo_cache_key=("shampoo", "observed", 0.25),
    )
    second = _evaluate_preconditioner_pair(
        operator,
        preconditioner,
        preconditioner,
        starts=starts,
        steps=4,
        condition_cfg={"relative_floor": 1.0e-8, "fallback_tau": 1.0e-4},
        spectrum_cache=cache,
        adam_cache_key=("adam", "centered", "base"),
        shampoo_cache_key=("shampoo", "aligned", 0.25),
    )
    third = _evaluate_preconditioner_pair(
        operator,
        preconditioner,
        preconditioner,
        starts=starts,
        steps=4,
        condition_cfg={"relative_floor": 1.0e-8, "fallback_tau": 1.0e-4},
        spectrum_cache=cache,
        adam_cache_key=("adam", "centered", "base"),
        shampoo_cache_key=("shampoo", "observed", 0.25),
    )

    assert len(calls) == 3
    assert first[1] is second[1] is third[1]
    assert first[2] is third[2]
    assert second[2] is not first[2]
    assert cache.diagnostics() == {"entries": 3, "hits": 3, "misses": 3}
