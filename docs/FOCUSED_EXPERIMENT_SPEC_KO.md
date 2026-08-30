# Optimizer-Visible Curvature Canonical Balanced 실험 사양

## 목표

ICLR 2027 논문의 frozen-operator 메커니즘을 checkpoint block에서 직접 검사한다. Primary quantity는 empirical GGN curvature와 frozen Adam/Shampoo-form operator로부터 계산한

\[
\Delta G=\log K_{\mathrm{Adam}}-\log K_{\mathrm{Shampoo}}
\]

이다. Online optimizer trajectory, stochastic-risk dominance, generalization, wall-clock superiority는 범위 밖이다.

## 유지하는 실험

1. **Synthetic conditioning**
   - Theorem 1의 aligned/scalar/reversed Adam-form ordering.
   - Flat-Kronecker paired covariance spectrum/diagonal과 Adam operator equality.
   - Shampoo aligned/reversed condition numbers, utilization exponent, damping response.
2. **Weighted-Chebyshev certificate**
   - `T+1` extremal nodes와 positive weights.
   - constrained optimum 및 scaled-Chebyshev equality `C_T(K)`.
   - three-group lower envelope `C_T(K)/3`.
3. **Frozen checkpoint analysis**
   - centered covariance primary, uncentered moment control.
   - empirical GGN curvature와 fixed blockwise stabilization shift.
   - paired Lanczos residual condition metrics와 tau curves.
   - observed/aligned/reversed assignment, `alpha=0.25/0.5`, joint/Shampoo-only damping.
4. **Canonical balanced reliability**
   - nested endpoint/probe budgets.
   - final native endpoint 및 final-versus-diagnostic agreement.
   - partial-trace matrix, clustered-subspace, intervention-factor stability.
   - calibrated grouped bootstrap.
   - canonical table promotion, balanced seed consensus, fail-closed export.

## 지원 backend

- Model: `tiny_causal_lm`, `hf_causal_lm`.
- Data: `synthetic_tokens`, `hf_text`.

## Primary scientific outputs

### Synthetic

- `theorem1_conditioning_results.csv`
- `flat_kronecker_conditioning_results.csv`
- `chebyshev_certificates.csv`
- `theory_summary.json`

### Balanced seed

- `balanced_reliability_certificates.csv`
- `endpoint_convergence.csv`
- `partial_trace_convergence.csv`
- `curvature_shift_overrides.json`
- `final/canonical_block_metrics.csv`
- `final/canonical_interventions.csv`
- `final/canonical_alpha_sweep.csv`
- `final/canonical_damping_sweep.csv`
- `final/canonical_spectral_gain_curve.csv`
- `final/scientific_status.json`

## Scientific invariants

- Model, tokenizer, dataset revision은 immutable commit이다.
- `data.order_seed`는 experiment seed와 분리하여 고정한다.
- Covariance와 curvature batch interval은 겹치지 않는다.
- 첫 diagnostic stage가 정한 curvature shift를 모든 후속 stage에서 재사용한다.
- Direct `delta_g`가 primary이고 elasticity/overlap/commutator는 diagnostics다.
- Numerical acceptance와 inferential ordering을 분리한다.
- Aligned/reversed inference에는 partial-trace geometry acceptance가 필요하다.
- Scientific aggregation은 balanced canonical tables만 사용한다.
- Positive, negative, inconclusive 결과를 모두 보존한다.
