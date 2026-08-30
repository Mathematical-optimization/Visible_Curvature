#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a canonical balanced reliability output"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--require-all-endpoints", action="store_true")
    parser.add_argument("--require-all-partial-traces", action="store_true")
    parser.add_argument("--require-scientific-acceptance", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root)
    required = [
        root / "COMPLETED",
        root / "endpoint_convergence.csv",
        root / "balanced_reliability_certificates.csv",
        root / "balanced_reliability_summary.json",
        root / "final" / "canonical_block_metrics.csv",
        root / "final" / "balanced_reliability_summary.json",
        root / "final" / "scientific_status.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    cert = pd.read_csv(root / "balanced_reliability_certificates.csv")
    summary = json.loads(
        (root / "balanced_reliability_summary.json").read_text(encoding="utf-8")
    )
    status = json.loads(
        (root / "final" / "scientific_status.json").read_text(encoding="utf-8")
    )
    if cert.empty:
        raise SystemExit("empty reliability certificate table")
    if args.require_all_endpoints and not cert[
        "adaptive_endpoint_certified"
    ].astype(bool).all():
        raise SystemExit("one or more endpoint checks failed")
    if args.require_all_partial_traces and not cert[
        "adaptive_partial_trace_certified"
    ].astype(bool).all():
        raise SystemExit("one or more partial-trace checks failed")
    if args.require_scientific_acceptance and status.get("scientific_status") != "accepted":
        raise SystemExit("scientific result is inconclusive")

    payload = {
        "pipeline_status": status.get("pipeline_status", "unknown"),
        "scientific_status": status.get("scientific_status", "unknown"),
        "primary_inference_available": bool(
            status.get("primary_inference_available", False)
        ),
        "num_blocks": int(len(cert)),
        "num_endpoint_checks_passed": int(
            cert["adaptive_endpoint_certified"].astype(bool).sum()
        ),
        "num_partial_trace_checks_passed": int(
            cert["adaptive_partial_trace_certified"].astype(bool).sum()
        ),
        "num_balanced_primary_reliable": int(
            summary.get("num_balanced_primary_reliable", 0)
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
