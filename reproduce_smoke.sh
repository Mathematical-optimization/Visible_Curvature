#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${ROOT}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
rm -rf \
  outputs/synthetic_theory \
  outputs/smoke \
  outputs/smoke_aggregate \
  outputs/smoke_figures \
  outputs/smoke_paper

if [[ "${VC_SKIP_TESTS:-0}" != "1" ]]; then
  python -m compileall -q visible_curvature scripts tests
  python -m pytest -q
fi

python scripts/run_synthetic_theory.py --config configs/synthetic_theory.yaml
python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("outputs/synthetic_theory/theory_summary.json").read_text())
if (
    not summary.get("all_checks_passed")
    or not summary.get("chebyshev_all_checks_passed")
    or not summary.get("integrated_theorem3_all_checks_passed")
):
    raise SystemExit(
        "synthetic theory, Chebyshev, or integrated Theorem-3 verification failed"
    )
for name in (
    "theorem1_conditioning_results.csv",
    "flat_kronecker_conditioning_results.csv",
    "chebyshev_certificates.csv",
    "integrated_theorem3_witness.csv",
):
    if not (Path("outputs/synthetic_theory") / name).is_file():
        raise SystemExit(f"missing synthetic output: {name}")
PY
python scripts/run_frozen_analysis.py --config configs/smoke.yaml
python scripts/validate_run.py --output-dir outputs/smoke

python scripts/aggregate_runs.py \
  --run-dir outputs/smoke \
  --output-dir outputs/smoke_aggregate \
  --minimum-seed-count 1
python scripts/make_figures.py \
  --output-dir outputs/smoke_aggregate \
  --figure-dir outputs/smoke_figures
python scripts/validate_run.py --output-dir outputs/smoke_aggregate

mkdir -p outputs/smoke_paper
if python scripts/export_paper_assets.py \
  --output-dir outputs/smoke_aggregate \
  --paper-root outputs/smoke_paper/scientific \
  > outputs/smoke_paper/scientific_export_attempt.log 2>&1; then
  echo "ERROR: synthetic aggregate unexpectedly passed scientific export guard" >&2
  exit 1
else
  echo "expected scientific export rejection"
fi

python scripts/export_paper_assets.py \
  --output-dir outputs/smoke_aggregate \
  --paper-root outputs/smoke_paper/debug \
  --allow-debug-export

grep -q "DEBUG EXPORT -- NOT SCIENTIFIC EVIDENCE" \
  outputs/smoke_paper/debug/network_results_autogen.tex

echo "Canonical Balanced v1.3.0 focused verification complete"
