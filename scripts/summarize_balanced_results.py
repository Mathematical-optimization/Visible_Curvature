#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a compact canonical balanced reliability report"
    )
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    cert = pd.read_csv(root / "balanced_reliability_certificates.csv")
    metrics = pd.read_csv(root / "final" / "canonical_block_metrics.csv")
    summary = json.loads(
        (root / "balanced_reliability_summary.json").read_text(encoding="utf-8")
    )
    status = json.loads(
        (root / "final" / "scientific_status.json").read_text(encoding="utf-8")
    )
    columns = [
        column
        for column in [
            "block_name",
            "delta_g",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "adaptive_endpoint_certified",
            "adaptive_partial_trace_certified",
            "final_endpoint_numerically_accepted",
            "final_diagnostic_agreement",
            "tau_sign_reason",
            "tail_localized_ordering",
            "balanced_reliable_ordering",
            "balanced_reliability_reasons",
        ]
        if column in metrics.columns
    ]
    print("=== Canonical balanced status ===")
    print(json.dumps(status, indent=2, sort_keys=True))
    print("\n=== Balanced reliability summary ===")
    print(
        json.dumps(
            {
                key: summary.get(key)
                for key in [
                    "num_blocks",
                    "num_endpoint_certified",
                    "num_partial_trace_certified",
                    "num_balanced_primary_reliable",
                    "selected_endpoint_steps",
                    "selected_endpoint_starts",
                    "selected_partial_trace_probes",
                ]
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("\n=== Numerical check table ===")
    print(cert.to_string(index=False))
    print("\n=== Canonical primary rows ===")
    print(metrics[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
