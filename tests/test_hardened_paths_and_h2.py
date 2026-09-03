from pathlib import Path
import pandas as pd
from ovc_experiments.paths import resolve_output_dir
from ovc_experiments.hardened_reporting import evaluate_hypotheses, evaluate_h2_leave_one_cluster_out


def test_output_dir_is_relative_to_config_file(tmp_path):
    cfg = tmp_path / 'configs' / 'run.yaml'; cfg.parent.mkdir(); cfg.write_text('x: 1')
    output = resolve_output_dir('../outputs', config_path=cfg)
    assert output == (tmp_path / 'outputs').resolve()


def test_positive_in_sample_h2_is_not_confirmatory_support():
    frame = pd.DataFrame({'response_shampoo':[0.,1.,2.,3.], 'delta_G':[0.,1.,2.,3.]})
    row = evaluate_hypotheses(frame).query("hypothesis == 'H2'").iloc[0]
    assert row.status == 'descriptive_only'
