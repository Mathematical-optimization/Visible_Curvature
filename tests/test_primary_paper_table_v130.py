import pandas as pd

from visible_curvature.paper_export import _primary_table_lines


def test_primary_table_reports_block_type_ci_metric_tau_and_tail_state():
    primary = pd.DataFrame(
        [
            {
                "block_name": "model.layers.0.q_proj.weight",
                "block_type": "attn_q",
                "K_adam_median": 123.456,
                "K_shampoo_median": 45.0,
                "delta_g_median": 1.01,
                "bootstrap_ci_low_median": 0.2,
                "bootstrap_ci_high_median": 1.4,
                "condition_metric_consensus": "truncated",
                "fallback_tau_consensus": 1.0e-4,
                "tail_localized_consensus": "yes",
                "n_seeds": 3,
                "reliable_ordering": "positive",
            }
        ]
    )
    text = "\n".join(_primary_table_lines(primary))
    assert "Type" in text
    assert "Bootstrap CI" in text
    assert "Tail" in text
    assert "attn\\_q" in text
    assert "[0.200, 1.400]" in text
    assert "1.0e-04" in text
    assert "yes" in text
    assert " & 3 & " in text
    assert "0.000" not in text
