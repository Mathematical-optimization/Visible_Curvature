# Canonical Balanced v1.3.0 릴리스 노트

## 릴리스 목적

v1.3.0은 frozen Adam/Shampoo 실험을 논문의 정적 이론과 정확히 같은 estimand로 보고하도록 정렬한 scientific-correctness release다. 핵심 변경은 predictor baseline 항, 일반 block에서의 damping/alpha 해석, ordinary/truncated metric compatibility, exact block preregistration, provenance, 그리고 full Theorem-3 witness 검증이다.

## 주요 수정

### 1. Elasticity predictor

기존 consumption-only 식을 세 항으로 분리했다.

```text
baseline_width_mismatch
+ delta_g_predicted_consumption
= delta_g_predicted_full_proxy
```

`delta_g_predicted`는 backward compatibility를 위해 full proxy alias로 남긴다. Adam `R²`, mode 수, curvature width, floor dominance가 reliability gate에 추가되었다.

### 2. Damping과 alpha estimand

- Joint damping: `|ΔG|`.
- Shampoo-only damping: `|G_shampoo|` 또는 scalar-limit distance.
- Alpha: `ΔG(α)-ΔG(0.25)`의 signed paired response.

따라서 일반 block에서 Shampoo-only `ΔG→0`이나 alpha 증가에 따른 `|ΔG|` 증가를 자동 가정하지 않는다.

### 3. Condition metric consistency

Adaptive stage, final run, seed aggregation에서 ordinary/truncated metric을 비교하고 truncated이면 `τ`도 비교한다. 기본 balanced policy는 `allow_truncated_primary=false`다. Truncated result는 secondary diagnostic으로 보존되지만 signed primary ordering으로 승격되지 않는다.

### 4. 계산량 절감

Block-local spectrum cache가 동일한 frozen operator의 Lanczos endpoint를 intervention, alpha, Shampoo-only damping 사이에서 재사용한다. Confirmatory primary는 12개 exact block, expensive control은 4개 preregistered block으로 분리했다.

### 5. Aggregate mechanism analysis

새 산출물:

- `elasticity_prediction_rows.csv`;
- `elasticity_prediction_summary.csv`;
- `paired_control_contrasts.csv`.

Sign accuracy와 correlation은 reliability-gated row에서만 계산한다. Controls는 within-block paired contrast를 먼저 만든다.

### 6. Provenance

`runtime_provenance.json`에 source-tree digest, git state, Python/PyTorch/numerical-library versions, CUDA/cuDNN, CPU/GPU/OS, deterministic flags, thread environment를 기록한다. Aggregate는 runtime environment compatibility를 별도 검사한다.

### 7. Ridge sweep

`ridge_sweep.csv`가 optional secondary control로 추가되었다. 이는 `τ` dependence와 분리된다.

### 8. Integrated Theorem-3 witness

`integrated_theorem3_witness.csv`는 다음을 하나의 budget-dependent construction에서 동시에 검사한다.

- reciprocal closure;
- flat basis covariance spectrum/diagonal invariants;
- scalar Adam operator;
- scalar/aligned/reversed 세 Chebyshev group;
- 공통 초기 에너지 배분;
- 세 simultaneous lower certificates;
- dimension bound.

## Confirmatory 실행 전 주의

배포 환경에서 OPT-125M GPU run을 수행하기 전에 한 block timing/memory pilot을 먼저 수행해야 한다. 12개 primary block 전체에는 primary endpoint를 계산하지만, 고비용 controls는 네 block subset에만 적용한다.

## 호환성

기존 command-line script와 주요 CSV filename은 유지된다. 새 열과 새 summary file은 additive다. 다만 `delta_g_predicted`의 의미는 v1.3.0부터 full proxy이며, scientific promotion은 ordinary-only default를 따른다.
