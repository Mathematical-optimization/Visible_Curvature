from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ovc_experiments.io import atomic_write_json


def test_atomic_write_json_emits_strict_json_for_nonfinite_values(tmp_path: Path) -> None:
    path = atomic_write_json(
        {
            "nan": float("nan"),
            "positive_infinity": np.float64(float("inf")),
            "tensor": torch.tensor([1.0, float("nan")]),
        },
        tmp_path / "strict.json",
    )
    text = path.read_text(encoding="utf-8")
    payload = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert payload == {
        "nan": None,
        "positive_infinity": None,
        "tensor": [1.0, None],
    }
