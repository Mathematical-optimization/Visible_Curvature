# Canonical Balanced Frozen-Operator 실험 프로토콜

## 1. 연구 질문과 primary endpoint

Block `b`에 대해 empirical GGN curvature `H_b`와 frozen preconditioner `P_{\Phi,b}`를 사용하여

\[
K_{\Phi,b}=\operatorname{cond}\!\left(P_{\Phi,b}^{1/2}H_bP_{\Phi,b}^{1/2}\right),
\qquad
\Delta G_b=\log K_{\mathrm{Adam},b}-\log K_{\mathrm{Shampoo},b}
\]

를 측정한다.

- `ΔG_b>0`: frozen Shampoo-form residual conditioning이 더 좋다.
- `ΔG_b<0`: frozen Adam-form residual conditioning이 더 좋다.
- 수치 acceptance 또는 통계적 gate가 실패하면 `inconclusive`로 유지한다.

Primary estimand은 다음으로 고정한다.

- centered mini-batch-mean block-gradient covariance;
- observed factor assignment;
- practical factor exponent `α=0.25`;
- fixed empirical GGN operator와 fixed stabilization shift;
- Adam/Shampoo 비교에 동일한 Lanczos start;
- 동일한 ordinary 또는 truncated condition rule;
- balanced canonical reliability gate.

## 2. 실험 단계

### A. Synthetic conditioning

Theorem 1과 flat-Kronecker conditioning mechanism을 dense operator로 확인한다. Analytic/numerical condition number, covariance spectrum/diagonal equality, Adam operator equality, `α` response, damping attenuation을 검사한다.

### B. Weighted-Chebyshev certificate

각 `(K,T)`에 대해 extremal nodes, Lagrange weights, constrained weighted-polynomial optimum, scaled Chebyshev equality, `C_T(K)/3` lower envelope를 독립적으로 확인한다. Dense endpoint conditioning과 full residual-polynomial barrier를 같은 검증으로 취급하지 않는다.

### C. Screening

Confirmatory block을 사전 지정하기 위한 단계다. Bootstrap과 secondary controls를 비활성화한다. Signed outcome을 보고 유리한 block만 선택하지 않는다.

### D. Balanced confirmatory

사전 지정된 최대 두 block과 최소 세 seed에 대해 adaptive diagnostics와 final high-budget run을 수행한다.

## 3. Covariance estimand

Covariance batch `s`에서 mini-batch 평균 loss의 block gradient `G_s`를 계산한다.

\[
\widehat\Sigma_c=\frac1N\sum_s(G_s-\bar G)\odot(G_s-\bar G),
\qquad
\widehat M=\widehat\Sigma_c+\bar G\odot\bar G.
\]

Adam statistic은 coordinate diagonal, Shampoo statistic은 left/right row-column factors다. Centered와 uncentered statistic은 같은 gradient collection에서 생성한다.

Scientific grouped bootstrap에서는 `num_batches`가 `group_size`로 정확히 나누어져야 한다.

## 4. Curvature estimand

Causal-LM cross-entropy의 empirical generalized Gauss–Newton operator를 curvature batch들에 대해 평균한다. 이는 full nonconvex Hessian이 아니다.

Covariance interval과 curvature interval은 겹치지 않게 고정한다. Partial-trace Hutchinson probe는 left/right curvature-factor geometry를 추정한다.

## 5. Stabilization shift

첫 diagnostic stage에서 block별 PSD floor shift를 계산한다. 이후

```text
curvature_shift_overrides.json
```

에 기록된 동일 값을 모든 diagnostic, refinement, final stage에 적용한다. Stage마다 shift를 다시 추정하지 않는다.

각 core run은 raw endpoint estimate, target relative ridge, applied shift, shift source, override digest를 `curvature_shift_records.csv`에 기록한다.

## 6. Frozen operators

Adam-form:

\[
P_{\mathrm{Ad}}=\operatorname{Diag}(d+\lambda)^{-1/2}.
\]

Shampoo-form:

\[
P_{\mathrm{Sh}}^{(\alpha)}=(C_R+\rho_RI)^{-\alpha}\otimes(C_L+\rho_LI)^{-\alpha}.
\]

Effective operator는 `P^{1/2} H P^{1/2}`로 평가한다. Preconditioner의 전체 exponent와 sandwich half-operator exponent를 구분한다.

## 7. Endpoint numerical acceptance

각 stage에서 paired Lanczos로 Adam/Shampoo endpoint를 계산한다. 다음을 모두 요구한다.

1. finite endpoint;
2. core min/max Ritz-residual 기준 통과;
3. shift-ratio 기준 통과;
4. 이전 budget 대비 `K_adam`, `K_shampoo` 상대 변화 허용 범위 내;
5. 이전 budget 대비 `ΔG` 절대 변화 허용 범위 내.

Final run은 자체 endpoint check를 다시 통과하고 selected diagnostic stage와 일치해야 한다. 이 acceptance는 rigorous eigenvalue enclosure가 아니라 preregistered numerical check다.

## 8. Partial-trace geometry acceptance

Aligned/reversed intervention을 inference에 사용하려면 연속 probe budget 사이에서 다음을 검사한다.

1. left/right negative spectral mass;
2. left/right relative Frobenius matrix change;
3. spectral-cluster별 projector distance;
4. aligned/reversed factor relative change.

Adjacent relative eigengap이 설정값보다 작은 eigenvalue들은 하나의 cluster로 묶는다. Degenerate cluster 내부 회전은 허용하되 cluster subspace 자체의 이동은 제한한다.

## 9. Spectral-gain curve

모든 direct comparison에 대해 declared `τ` grid의

\[
K_{\mathrm{Adam}}(\tau),\quad K_{\mathrm{Shampoo}}(\tau),\quad\Delta G(\tau)
\]

를 `spectral_gain_curve.csv`에 저장한다.

Balanced 분류:

- `stable_nonzero`;
- `one_sided_with_coarse_saturation`;
- `sign_flip`;
- `all_saturated`;
- `tau_refinement_unavailable`.

Ordinary endpoint가 accepted되지 않은 상태에서 tail-localized result를 최종 ordering으로 승격하지 않는다.

## 10. Assignment intervention

Observed covariance-factor eigenvalue multiset을 유지한 채 curvature partial-trace eigenspace에 재배치한다.

- `observed`: 실제 factor;
- `aligned`: 큰 factor eigenvalue와 큰 curvature-factor eigenvalue를 대응;
- `reversed`: 큰 factor eigenvalue와 작은 curvature-factor eigenvalue를 대응.

Adam은 observed coordinate diagonal에 고정한다. Aligned/reversed 결과는 partial-trace geometry acceptance를 통과해야 한다.

## 11. Utilization exponent control

같은 factor와 damping에서 `α=0.25`와 `α=0.5`를 비교한다. `α=0.5`는 idealized inverse-square-root control이며 practical Shampoo 교체 제안이 아니다.

## 12. Damping controls

두 sweep을 구분한다.

1. `joint`: Adam과 Shampoo normalized coefficient를 함께 변경한다.
2. `shampoo_only`: Adam primary spectrum을 고정하고 Shampoo coefficient만 변경한다.

논문의 retained-anisotropy attenuation mechanism과 직접 대응하는 것은 `shampoo_only` sweep이다. `joint`는 practical relative-hyperparameter control이다.

## 13. Bootstrap

Covariance batches를 동일 크기의 contiguous group으로 묶어 group 단위로 재표본화한다. Replicate에서는 covariance statistic, damping, frozen operators, low-budget `ΔG`를 다시 계산한다.

보고되는 interval은 high-budget point estimate를 중심으로 low-budget bootstrap deviation을 이동한 calibrated grouped-bootstrap interval이다. Curvature, shift, curvature batches, probe stream에는 조건부다.

## 14. Canonical reliability gate

Primary label이 positive 또는 negative가 되려면 다음을 모두 만족해야 한다.

1. primary row가 centered/observed/`α=0.25`;
2. adaptive endpoint accepted;
3. final native endpoint accepted;
4. final-versus-diagnostic agreement;
5. finite bootstrap replicate 수 기준;
6. bootstrap CI가 0의 한쪽에 위치;
7. point sign과 CI sign 일치;
8. tau classification이 `stable_nonzero` 또는 `one_sided_with_coarse_saturation`.

그 외는 `inconclusive`다. Controls는 endpoint check를 요구하며 aligned/reversed는 partial-trace geometry check도 추가로 요구한다.

## 15. Scientific provenance

- model, tokenizer, dataset을 40자리 immutable commit으로 고정;
- `data.order_seed`를 모든 seed에서 고정;
- experiment seed는 Lanczos/probe/bootstrap randomness에만 사용;
- source order, packed token stream, selected chunks를 hash로 기록;
- covariance/curvature interval 비중첩;
- protocol hash와 runtime identity를 seed 간 비교;
- 가능하면 동일 GPU architecture와 software environment 사용.

## 16. Aggregation 및 reporting

Seed-level primary table은 `final/canonical_block_metrics.csv`다. Aggregator는 balanced root를 전달받아 canonical tables를 자동 선택한다.

Seed consensus는 unanimous reliable sign만 인정한다. Conflict, unresolved seed, insufficient seed count는 모두 `inconclusive`다.

논문에서는 primary centered result와 mechanism controls를 구분한다. GGN curvature, frozen operator, idealized `α=0.5`, intervention realizability 한계, calibrated bootstrap 범위를 명시한다.

## 17. 주장 범위

이 프로토콜은 fixed checkpoint operator의 residual conditioning을 검사한다. Online Adam/Shampoo trajectory, stochastic risk, generalization, whole-network universal dominance, systems cost를 직접 검증하지 않는다.
