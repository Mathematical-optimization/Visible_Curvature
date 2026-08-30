#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${ROOT}"
rm -rf outputs/balanced_reliability_smoke
python scripts/run_balanced_reliability.py --policy configs/balanced_reliability_smoke.yaml
python scripts/validate_balanced_run.py --output-root outputs/balanced_reliability_smoke
python scripts/summarize_balanced_results.py --output-root outputs/balanced_reliability_smoke > outputs/balanced_reliability_smoke/summary.txt
printf '%s\n' 'Balanced reliability smoke verification complete'
