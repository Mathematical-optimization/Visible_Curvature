import json
import math

import pandas as pd

from visible_curvature.synthetic_theory import run_synthetic_theory


def _config(tmp_path):
    return {
        "output_dir": str(tmp_path / "theory"),
        "synthetic": {
            "kappa_values": [8.0],
            "r_values": [2.0],
            "alpha_values": [0.25, 0.5],
            "rho_values": [0.0, 1.0, 4096.0],
            "theorem1": {"chi_values": [8.0], "lambda_values": [0.0, 1.0]},
            "chebyshev": {
                "K_values": [4.0, 16.0],
                "T_values": [2, 4],
                "tolerance": 1.0e-8,
            },
            "tolerance": 1.0e-9,
        },
    }


def test_synthetic_runner_writes_required_files_and_theorem1_sign_reversal(tmp_path):
    out = run_synthetic_theory(_config(tmp_path))
    results_path = out / "theory_results.csv"
    summary_path = out / "theory_summary.json"
    assert results_path.is_file()
    assert summary_path.is_file()
    assert (out / "flat_kronecker_conditioning_results.csv").is_file()
    assert (out / "chebyshev_certificates.csv").is_file()
    assert (out / "integrated_theorem3_witness.csv").is_file()
    assert (out / "resolved_config.yaml").is_file()

    frame = pd.read_csv(results_path)
    t1 = frame[frame["experiment"] == "theorem1"]
    zero = t1[t1["damping"] == 0.0].set_index("assignment")
    assert math.isclose(zero.loc["aligned", "condition_numeric"], 1.0, rel_tol=1e-10)
    assert math.isclose(zero.loc["scalar", "condition_numeric"], 8.0, rel_tol=1e-10)
    assert math.isclose(zero.loc["reversed", "condition_numeric"], 64.0, rel_tol=1e-10)

    certificates = pd.read_csv(out / "chebyshev_certificates.csv")
    assert len(certificates) == 4
    assert certificates["all_checks_passed"].all()
    assert (certificates["quadratic_optimum_relative_error"] < 1.0e-8).all()
    assert (certificates["chebyshev_energy_relative_error"] < 1.0e-8).all()

    summary = json.loads(summary_path.read_text())
    assert summary["chebyshev_all_checks_passed"] is True
    assert summary["chebyshev_certificate_count"] == 4
    assert summary["integrated_theorem3_all_checks_passed"] is True
    assert summary["integrated_theorem3_witness_count"] >= 1
    assert summary["all_checks_passed"] is True


def test_flat_kronecker_invariants_alpha_doubling_and_damping_attenuation(tmp_path):
    out = run_synthetic_theory(_config(tmp_path))
    frame = pd.read_csv(out / "theory_results.csv")
    t3 = frame[frame["experiment"] == "theorem3"].copy()

    assert t3["invariants_passed"].all()
    assert (t3["relative_condition_error"] < 1e-8).all()

    zero = t3[t3["rho"] == 0.0]
    aligned = zero[zero["assignment"] == "aligned"].set_index("alpha")
    reversed_rows = zero[zero["assignment"] == "reversed"].set_index("alpha")
    assert math.isclose(
        aligned.loc[0.5, "delta_g"],
        2.0 * aligned.loc[0.25, "delta_g"],
        rel_tol=1e-9,
    )
    assert math.isclose(
        reversed_rows.loc[0.5, "delta_g"],
        2.0 * reversed_rows.loc[0.25, "delta_g"],
        rel_tol=1e-9,
    )

    practical_aligned = t3[(t3["alpha"] == 0.25) & (t3["assignment"] == "aligned")]
    practical_aligned = practical_aligned.sort_values("rho")
    assert practical_aligned["delta_g"].abs().iloc[-1] < practical_aligned["delta_g"].abs().iloc[0]
