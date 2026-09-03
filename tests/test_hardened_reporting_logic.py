import pandas as pd
from ovc_experiments.hardened_reporting import evaluate_hypotheses


def test_h2_rejects_negative_correlation():
    geometry = pd.DataFrame({'delta_G': [3., 2., 1., 0.], 'response_shampoo': [0., 1., 2., 3.]})
    summary = evaluate_hypotheses(geometry)
    row = summary[summary.hypothesis == 'H2'].iloc[0]
    assert row.status == 'not_supported'


def test_h3_requires_signed_reversal_per_checkpoint():
    interventions = pd.DataFrame({
        'checkpoint_step': [1, 1, 2, 2],
        'block_name': ['b', 'b', 'b', 'b'],
        'branch': ['aligned', 'reversed', 'aligned', 'reversed'],
        'delta_G': [1.0, -1.0, 0.5, -0.4],
    })
    summary = evaluate_hypotheses(pd.DataFrame({'delta_G': [1., -1.]}), interventions)
    assert summary[summary.hypothesis == 'H3'].iloc[0].status == 'supported'


def test_h5_compares_quarter_to_half_and_requires_both_signs():
    interventions = pd.DataFrame({
        'checkpoint_step': [1,1,1,1],
        'block_name': ['good','good','bad','bad'],
        'alpha': [0.25,0.5,0.25,0.5],
        'G_shampoo': [1.0,2.0,-1.0,-2.0],
    })
    summary = evaluate_hypotheses(pd.DataFrame({'delta_G': [1., -1.]}), interventions)
    assert summary[summary.hypothesis == 'H5'].iloc[0].status == 'supported'
