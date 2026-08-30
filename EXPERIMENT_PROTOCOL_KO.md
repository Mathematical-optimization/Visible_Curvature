# 실험 프로토콜 — Canonical Balanced v1.3.0

## 1. 목적과 주장 범위

본 프로토콜은 고정 checkpoint에서 second-moment statistic으로 한 번 구성한 Adam-form 및 Shampoo-form operator가 block curvature를 어떻게 변환하는지 측정한다. Online optimizer trajectory, fresh stochastic-noise coupling, generalization, whole-network dominance, wall-clock 우월성을 직접 검증하지 않는다.

Primary quantity는

\[
K(P,H)=\operatorname{cond}(P^{1/2}HP^{1/2}),
\qquad
\Delta G=\log K_{\rm Adam}-\log K_{\rm Shampoo}
\]

이다.

## 2. 데이터와 gradient second moment

1. 모델은 `eval()` 상태로 고정한다.
2. 각 covariance sample은 한 mini-batch의 mean loss로부터 얻은 block gradient다.
3. Primary covariance는 centered mini-batch-mean block-gradient covariance다.
4. Uncentered second moment는 contamination diagnostic으로만 사용한다.
5. Batch size와 sequence length는 covariance estimand 정의의 일부다.
6. Covariance batch interval과 curvature batch interval은 비중첩이어야 한다.
7. 모든 seed에서 동일한 immutable checkpoint, token order, packed stream, selected content를 사용한다.

## 3. Curvature

Primary curvature는 causal-LM mean cross-entropy의 empirical generalized Gauss–Newton operator다.

\[
H_{\rm GGN}=\frac1N J^T(\operatorname{Diag}(p)-pp^T)J.
\]

Full nonconvex Hessian으로 서술하지 않는다. Curvature stream은 covariance stream과 분리한다.

## 4. Stabilization shift

첫 diagnostic stage에서 block별 raw endpoint를 이용해 shift를 한 번 보정한다. 결과는 `curvature_shift_overrides.json`에 기록하며 이후 adaptive stage, refinement, final run에서 같은 값을 사용한다.

보고 항목:

- raw minimum/maximum Ritz estimate;
- target relative ridge;
- applied shift;
- shift source;
- override digest;
- `shift/raw_max` ratio.

Ridge sweep는 별도 secondary control이다. Primary shift를 원하는 결과가 나올 때까지 조정하지 않는다.

## 5. Frozen preconditioners

Adam-form:

\[
P_{\rm Ad}=\operatorname{Diag}(\Sigma+\lambda)^{-1/2}.
\]

Shampoo-form:

\[
P_{\rm Sh}^{(\alpha)}=(C_R+\rho_RI)^{-\alpha}\otimes(C_L+\rho_LI)^{-\alpha}.
\]

Effective operator는 항상 `P^{1/2}HP^{1/2}`로 평가한다. Full preconditioner exponent와 sandwich half-action exponent를 구분한다.

## 6. Primary block set과 control subset

Confirmatory primary set은 exact parameter name으로 사전 등록한다. 배포 template은 OPT-125M의 layer `0,2,4,6,8,11`에서 `q_proj.weight`와 `out_proj.weight`를 선택하여 총 12개 block을 포함한다.

고비용 control subset은 layer `0`과 `11`의 네 block이다. Control subset은 exact primary set의 부분집합이어야 한다. Screening 부호를 보고 block을 교체하지 않는다.

## 7. Compatible condition metrics

Ordinary condition과 truncated condition은 다른 estimand다.

Ordinary:

\[
K=\lambda_{\max}/\lambda_{\min}.
\]

Relative-`τ` truncated:

\[
K_\tau=\frac{\lambda_{\max}}
{\max(\lambda_{\min},\tau\lambda_{\max})}.
\]

한 direct comparison에서 `K_H`, `K_adam`, `K_shampoo`는 같은 metric과 같은 `τ`로 계산한다. 다음을 기록한다.

\[
G_{\rm Adam}=\log K_H-\log K_{\rm Adam},
\qquad
G_{\rm Shampoo}=\log K_H-\log K_{\rm Shampoo}.
\]

`delta_g_from_gains`는 `G_shampoo-G_adam`과 직접 계산한 `ΔG`의 일치 여부를 감사하는 열이다.

기본 policy의 `allow_truncated_primary=false`에서는 ordinary primary만 signed scientific ordering으로 승격한다. Truncated 결과는 secondary tail diagnostic으로 보존한다.

## 8. Endpoint numerical acceptance

각 stage에서 paired common-start Lanczos를 사용한다. 연속 stage 사이에서 다음을 모두 요구한다.

1. finite Ritz endpoint;
2. native min/max Ritz-residual 기준;
3. shift-ratio 기준;
4. `K_adam`, `K_shampoo` 상대 변화 기준;
5. `ΔG` 절대 변화 기준;
6. condition metric 일치;
7. truncated일 경우 `τ` 일치.

Final run도 자체 endpoint check를 통과하고 selected stage와 metric, `τ`, 수치값이 일치해야 한다. 이는 numerical acceptance이며 rigorous enclosure가 아니다.

## 9. Partial-trace geometry acceptance

Aligned/reversed control을 inference에 사용할 때는 연속 probe budget 사이에서 다음을 검사한다.

1. left/right negative spectral mass;
2. relative Frobenius matrix change;
3. spectral cluster별 subspace projector distance;
4. aligned/reversed intervention-factor relative change.

근접 고유값은 설정된 relative gap으로 cluster화한다. Cluster 내부 basis rotation은 허용하지만 cluster subspace 이동은 제한한다.

## 10. Elasticity diagnostic

각 factor 및 Adam coordinate statistic에 대해 `log q`를 `log h`에 회귀한다. Full proxy는

\[
\widehat{\Delta G}_{\rm full}
=
\underbrace{W_A-(W_L+W_R)}_{B}
+
\underbrace{\alpha(r_LW_L+r_RW_R)-\tfrac12r_AW_A}_{R}.
\]

저장 열:

- `baseline_width_mismatch`;
- `delta_g_predicted_consumption`;
- `delta_g_predicted_full_proxy`;
- `delta_g_predicted` (full proxy alias).

Prediction eligibility는 다음을 모두 요구한다.

- left/right/Adam `R²`;
- 최소 mode 수;
- 최소 curvature log-width;
- bounded factor commutator;
- bounded factor eigensolver residual;
- bounded partial-trace negative mass;
- bounded Adam/Shampoo floored fraction;
- finite full proxy.

이는 commuting–Kronecker proxy이며 일반 noncommuting block의 exact formula가 아니다.

## 11. Assignment intervention

Observed covariance-factor eigenvalue multiset을 curvature partial-trace eigenspace에 재배치한다.

- `observed`: 실제 retained factor;
- `aligned`: 큰 factor eigenvalue와 큰 curvature-factor eigenvalue를 대응;
- `reversed`: 큰 factor eigenvalue와 작은 curvature-factor eigenvalue를 대응.

Adam operator는 observed coordinate diagonal에 고정한다. 이 intervention은 retained-factor spectrum을 보존하는 frozen-operator intervention이며, observed covariance와 동일한 full spectrum 및 diagonal을 갖는 실현 가능한 full covariance를 반드시 정의하지는 않는다.

## 12. Alpha control

같은 factor와 damping에서 `α=0.25`와 `α=0.5`를 비교한다. Primary contrast는 block, seed, assignment별

\[
\Delta_\alpha=\Delta G(0.5)-\Delta G(0.25)
\]

이다. `|ΔG|` amplification은 baseline mismatch가 작고 scalar-comparator 근사가 허용되는 subset에서만 secondary 해석으로 사용한다. `α=0.5`는 idealized frozen control이다.

## 13. Damping controls

1. `joint`: Adam과 Shampoo coefficient를 함께 변경한다. Target은 `|ΔG|`다.
2. `shampoo_only`: Adam spectrum을 primary 값에 고정하고 Shampoo coefficient만 변경한다. Target은 `|G_shampoo|` 또는 `|ΔG+G_adam|`다.

일반 block의 Shampoo-only large-damping limit은

\[
\Delta G\to-G_{\rm Adam}
\]

이므로 `ΔG→0`을 일반 예측으로 사용하지 않는다.

## 14. Ridge sweep

사전 등록한 relative coefficient 집합으로 stabilized curvature를 다시 구성하고 primary observed operator pair를 평가한다. Ridge sweep은 결과 선택 도구가 아니라 shift dependence를 공개하는 secondary control이다. `τ` sweep과 혼동하지 않는다.

## 15. Bootstrap

Covariance batches를 contiguous group으로 묶고 group 단위로 재표본화한다. 각 replicate에서 covariance statistic, damping, frozen operators, low-budget `ΔG`를 다시 계산한다.

보고 interval은 high-budget point estimate를 중심으로 low-budget bootstrap deviation을 이동한 calibrated grouped-bootstrap interval이다. Curvature batches, curvature shift, partial-trace probe stream, checkpoint, token stream에는 조건부다.

## 16. Balanced scientific promotion

Primary label이 positive 또는 negative가 되려면 다음을 모두 만족해야 한다.

1. centered/observed/`α=0.25` primary row;
2. adaptive endpoint certificate;
3. final native endpoint acceptance;
4. final-versus-selected-stage metric 및 수치 agreement;
5. ordinary metric 또는 명시적으로 허용된 compatible truncated metric;
6. finite bootstrap replicate 수 기준;
7. bootstrap CI가 0의 한쪽에 위치;
8. point sign과 CI sign 일치;
9. accepted `τ` stability classification.

그 외는 `inconclusive`다. Control row는 endpoint/metric compatibility를 요구하며 aligned/reversed는 partial-trace geometry acceptance도 요구한다.

## 17. Seed aggregation

Seed는 동일 checkpoint와 동일 token stream에서 Lanczos/probe/bootstrap randomness를 바꾸는 numerical replication이다. 독립적인 model-training sample로 해석하지 않는다.

Seed consensus는 unanimous reliable sign만 인정한다. 다음은 모두 `inconclusive`다.

- sign conflict;
- unresolved seed;
- insufficient seed count;
- ordinary/truncated mismatch;
- truncated `τ` mismatch.

Block은 한 model 내부의 상관된 단위이므로 iid block 가정을 사용하는 binomial inference를 적용하지 않는다.

## 18. Paired mechanism reporting

Cross-block pooled median보다 다음 within-block contrasts를 우선한다.

- assignment: aligned minus reversed, observed minus reversed, sign-flip indicator;
- alpha: `ΔG(α)-ΔG(0.25)`;
- damping: coefficient별 declared control value와 최소 사전등록 coefficient 대비 변화.

Aggregator는 `paired_control_contrasts.csv`를 생성한다. Elasticity mechanism은 eligible row만 사용하여 sign accuracy, balanced accuracy, Spearman statistic을 보고한다.

## 19. Provenance

각 run은 다음을 `runtime_provenance.json`과 manifest에 기록한다.

- source-tree SHA-256와 git state;
- Python 및 주요 package versions;
- PyTorch CUDA build와 cuDNN;
- OS, CPU, GPU name, memory, compute capability;
- deterministic-algorithm flags;
- thread 및 CUDA environment variables;
- runtime-environment digest.

Confirmatory seeds는 가능한 한 같은 software environment와 GPU architecture에서 실행한다.

## 20. 논문 보고 규칙

반드시 다음을 구분한다.

- empirical GGN과 full Hessian;
- centered mini-batch-mean covariance와 per-example covariance;
- ordinary condition과 truncated condition;
- observed full covariance와 surgical factor intervention;
- raw `ΔG`와 alpha/damping paired estimand;
- point estimate와 numerical acceptance;
- positive, negative, inconclusive block.
