# Visible Curvature Experiment Code — Canonical Balanced v1.2.1

## v1.2.1 — Partial-trace performance and execution safety

v1.2.1은 v1.2.0의 scientific protocol과 수치 판정 기준을 유지하면서, 실제 OPT-125M 실행에서 확인된 CPU 병목과 concurrent-output 위험을 수정한 patch release다.

### 핵심 수정

1. Clustered eigenspace distance를 `||UU^T-VV^T||_2`의 full projector SVD로 계산하지 않고, 동일한 principal-angle identity `sqrt(1-sigma_min(U^T V)^2)`로 계산한다. Singleton cluster가 많을 때 cluster마다 `768 x 768` SVD를 반복하던 병목을 제거한다.
2. 각 balanced output root에 process-lifetime POSIX lock을 적용한다. 동일 seed/output root를 두 orchestrator가 동시에 쓰면 두 번째 process가 즉시 실패한다.
3. Core child analysis와 CPU-only endpoint/partial-trace certification의 시작·종료를 timestamp와 PID로 출력한다. GPU 메모리가 비는 정상적인 parent-side certification 구간을 명확히 구분한다.
4. `make_balanced_policies.py`에 `--gpus`와 `--cpu-threads`를 추가한다. 생성된 policy의 `runtime_env`가 child core process에 전달된다.
5. Performance regression, projector-distance equivalence, output-lock, GPU/thread policy generation 테스트를 추가한다.

## v1.2.0 — Canonical balanced scientific pipeline

## 목적

v1.2.0은 balanced reliability 계산을 실제 aggregation과 paper export까지 연결하고, intervention이 의존하는 partial-trace geometry와 stage 간 curvature operator 일관성을 직접 검사한다. 또한 dense conditioning verification과 full residual-polynomial lower-bound device의 검증을 분리한다.

## 핵심 변경

### 1. Stage 간 동일 curvature shift

첫 diagnostic stage에서 block별 stabilization shift를 계산하고 `curvature_shift_overrides.json`에 저장한다. 이후 모든 stage는 동일 값을 사용한다. 따라서 endpoint budget 증가가 서로 다른 shifted operator의 비교로 오염되지 않는다.

### 2. Final endpoint 재검증

Adaptive diagnostics가 통과했더라도 final high-budget endpoint가 자체 Ritz-residual check를 통과하고 selected diagnostic estimate와 일치해야 primary result가 승격된다.

### 3. Partial-trace geometry acceptance

Negative spectral mass 외에 다음을 추가했다.

- left/right partial-trace relative Frobenius change;
- spectral-cluster projector distance;
- aligned/reversed intervention-factor relative change;
- near-degenerate eigenspace의 cluster-aware 비교.

Aligned/reversed control은 이 geometry gate를 통과해야 inferentially usable하다.

### 4. Canonical scientific tables

Balanced final output은 다음 canonical tables를 생성한다.

- `canonical_block_metrics.csv`
- `canonical_interventions.csv`
- `canonical_alpha_sweep.csv`
- `canonical_damping_sweep.csv`
- `canonical_spectral_gain_curve.csv`

Aggregation은 이 table들을 자동 선택하고 `balanced_reliable_ordering`으로 seed consensus를 계산한다. Scientific export는 balanced-canonical aggregate만 허용한다.

### 5. Tau curve 연결

Core runner가 모든 declared `τ`의 `K_adam`, `K_shampoo`, `ΔG`, saturation을 `spectral_gain_curve.csv`에 저장한다. Missing curve는 `tau_refinement_unavailable`이며 silent fallback이 없다.

### 6. Damping sweep 분리

- `joint`: Adam/Shampoo damping을 함께 변경
- `shampoo_only`: Adam primary spectrum을 고정하고 Shampoo damping만 변경

`shampoo_only`는 retained factor anisotropy attenuation을 더 직접적으로 검사한다.

### 7. Weighted-Chebyshev certificate

`visible_curvature/chebyshev.py`가 다음을 검증한다.

- `T+1` extremal nodes;
- positive Lagrange weights와 합 1;
- constrained weighted-polynomial optimum `C_T(K)`;
- scaled Chebyshev equality;
- Theorem 3 three-group lower envelope `C_T(K)/3`;
- factor/full dimension strict upper bounds.

Dense flat-Kronecker condition-number check와 별도 CSV로 보고한다.

### 8. Status 의미 분리

Balanced validator는 다음을 구분한다.

- `pipeline_status`: 실행과 산출물 생성 완료 여부
- `scientific_status`: 모든 primary row의 scientific promotion 가능 여부
- `primary_inference_available`: primary sign inference 사용 가능 여부

`pipeline_status=complete`, `scientific_status=inconclusive`는 정상적이고 보존해야 하는 결과다.

## 주요 실행 명령

Synthetic:

```bash
python scripts/run_synthetic_theory.py --config configs/synthetic_theory.yaml
```

Balanced seed:

```bash
python scripts/run_balanced_reliability.py \
  --policy configs/generated_balanced/hf_opt125m_balanced_seed0.yaml
```

Validation:

```bash
python scripts/validate_balanced_run.py \
  --output-root outputs/hf_opt125m_balanced_seed0
```

Aggregation:

```bash
python scripts/aggregate_runs.py \
  --run-dir outputs/hf_opt125m_balanced_seed0 \
  --run-dir outputs/hf_opt125m_balanced_seed1 \
  --run-dir outputs/hf_opt125m_balanced_seed2 \
  --output-dir outputs/hf_opt125m_balanced_aggregate \
  --minimum-seed-count 3 \
  --make-figures
```

## 해석 원칙

- Checkpoint curvature는 empirical GGN이다.
- 결과는 frozen operator와 frozen shift의 residual conditioning에 관한 것이다.
- Numerical acceptance는 rigorous spectral enclosure가 아니다.
- `α=0.5`는 idealized control이다.
- Calibrated bootstrap은 fixed curvature/probe stream에 조건부다.
- Inconclusive result를 제외하거나 threshold를 사후 조정하여 desired sign을 만들지 않는다.
- Online optimizer dominance, stochastic risk, generalization, wall-clock superiority를 주장하지 않는다.
