> **Superseded:** This historical v1.1.0 plan is retained for provenance. The implemented v1.2.0 design and execution plan are `docs/superpowers/specs/2026-08-30-canonical-balanced-pipeline-design.md` and `docs/superpowers/plans/2026-08-30-canonical-balanced-pipeline-v1.2.0.md`.

# Balanced reliability implementation plan

## Goal

Prevent covariance-bootstrap precision from being mistaken for spectral
endpoint accuracy.  The final label must require independent evidence that the
smallest effective-curvature endpoint and partial-trace intervention basis have
stabilised as numerical budgets increase.

## Design

1. Run nested no-bootstrap diagnostic stages with endpoint budgets
   `64, 96, 128, 192, 256` and probe budgets `32, 64, 128, 256`.
2. Require the original Ritz-residual certificate plus cross-budget stability
   of `K_adam`, `K_shampoo`, and `delta_g`.
3. If the endpoint remains uncertified, run a 256-step, six-start multistart
   refinement rather than adding another statistical seed.
4. Certify partial traces separately using PSD negative mass and convergence
   across probe budgets.
5. Run the expensive bootstrap/intervention/alpha/damping controls once at the
   selected maximum certified budget.
6. Preserve the original outputs and write separate `balanced_*` tables.
7. Distinguish coarse truncation saturation from a genuine sign reversal.

## Non-goals

- The pipeline does not convert an uncertified ordinary condition number into a
  confirmed result by choosing a favourable truncation threshold.
- Stability across budgets does not override a failed Ritz residual.
- Partial-trace failure does not invalidate the observed preconditioner result;
  it invalidates only curvature-factor-basis claims.
