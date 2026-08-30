from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from visible_curvature.analysis_runner import iter_block_analyses
from scripts.validate_run import validate_run


@dataclass
class DummyBlock:
    name: str
    block_type: str = "attn_q"
    shape: tuple[int, int] = (2, 2)


def test_iter_block_analyses_records_failure_and_continues():
    blocks = [DummyBlock("bad"), DummyBlock("good")]

    def analyze(block):
        if block.name == "bad":
            raise RuntimeError("intentional failure")
        return ([{"block_name": block.name}], [], [], [], [])

    outcomes = list(iter_block_analyses(blocks, analyze))
    assert outcomes[0][1] is None
    assert outcomes[0][2]["block_name"] == "bad"
    assert outcomes[0][2]["error_type"] == "RuntimeError"
    assert "intentional failure" in outcomes[0][2]["error_message"]
    assert outcomes[1][1][0][0]["block_name"] == "good"
    assert outcomes[1][2] is None


def test_validator_fails_closed_on_captured_block_failures(tmp_path: Path):
    (tmp_path / "run_manifest.json").write_text(
        '{"status":"complete","experiment_tier":"debug","scientific_run":false}',
        encoding="utf-8",
    )
    pd.DataFrame([
        {
            "block_name": "good",
            "covariance_moment": "centered",
            "assignment": "observed",
            "alpha": 0.25,
            "delta_g": 0.1,
            "endpoint_numerically_reliable": True,
            "ordering_inferentially_reliable": False,
            "reliable_ordering": "inconclusive",
        }
    ]).to_csv(tmp_path / "block_metrics.csv", index=False)
    for name in ("bootstrap_metrics.csv", "interventions.csv", "alpha_sweep.csv", "damping_sweep.csv"):
        pd.DataFrame().to_csv(tmp_path / name, index=False)
    pd.DataFrame([{"block_name": "bad", "error_type": "RuntimeError", "error_message": "boom"}]).to_csv(
        tmp_path / "block_failures.csv", index=False
    )
    failures, _ = validate_run(tmp_path)
    assert "captured block failures: 1" in failures
