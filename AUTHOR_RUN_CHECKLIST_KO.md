# Author Run Checklist — Canonical Balanced v1.3.0

## A. Release 검증

- [ ] `python -m pytest -q` 통과
- [ ] `python -m compileall -q visible_curvature scripts tests` 통과
- [ ] `bash reproduce_smoke.sh` 통과
- [ ] `bash reproduce_balanced_smoke.sh`의 `pipeline_status=complete`
- [ ] `sha256sum -c SHA256SUMS.txt` 통과
- [ ] `visible_curvature.__version__`과 `pyproject.toml`이 `1.3.0`
- [ ] synthetic summary의 `all_checks_passed=true`
- [ ] `chebyshev_all_checks_passed=true`
- [ ] `integrated_theorem3_all_checks_passed=true`

## B. GPU pilot

- [ ] 한 block에서 GGN matvec 시간 측정
- [ ] 64/96/128/192/256-step endpoint 시간 측정
- [ ] peak GPU memory 측정
- [ ] partial-trace probe 시간 측정
- [ ] Lanczos spectrum cache hit/miss 확인
- [ ] 12 primary block과 4 control block의 총 runtime 추정
- [ ] output/log/checkpoint 저장 공간 확인

## C. Scientific identity

- [ ] model revision이 40자리 immutable commit
- [ ] tokenizer revision이 40자리 immutable commit
- [ ] dataset revision이 40자리 immutable commit
- [ ] 모든 seed에서 `data.order_seed` 동일
- [ ] 모든 seed에서 source-order/content hash 동일
- [ ] 모든 seed에서 12개 exact block 동일
- [ ] 4개 control subset이 primary 집합의 부분집합
- [ ] covariance/curvature interval 동일하고 비중첩
- [ ] protocol hash 일치
- [ ] runtime identity 일치
- [ ] runtime environment digest 호환

## D. Screening discipline

- [ ] bootstrap 비활성화
- [ ] intervention/alpha/damping/ridge controls 비활성화
- [ ] block 선택 기준을 type/depth/memory/numerical feasibility로 기록
- [ ] screening `ΔG` sign을 선택 기준으로 사용하지 않음
- [ ] confirmatory exact-name YAML을 결과 확인 전에 동결

## E. Balanced policy

- [ ] 최소 세 seed policy 생성
- [ ] seed별 `CUDA_VISIBLE_DEVICES` 충돌 없음
- [ ] CPU certification thread cap 기록
- [ ] `fixed_relative_ridge` 사전 고정
- [ ] endpoint steps/starts schedule 사전 고정
- [ ] partial-trace probe schedule 사전 고정
- [ ] matrix/subspace/intervention tolerance 사전 고정
- [ ] `allow_truncated_primary=false` 유지 또는 변경 근거 사전 기록
- [ ] final bootstrap replicate 수와 minimum finite count 사전 고정
- [ ] covariance batch 수가 group size로 나누어짐

## F. Seed-level artifacts

- [ ] `COMPLETED` 존재
- [ ] `runtime_provenance.json` 존재
- [ ] `curvature_shift_overrides.json` 존재
- [ ] post-calibration stage의 shift digest 일치
- [ ] `endpoint_convergence.csv` 검토
- [ ] `partial_trace_convergence.csv` 검토
- [ ] `balanced_reliability_certificates.csv` 검토
- [ ] `final/scientific_status.json` 검토
- [ ] `final/canonical_block_metrics.csv`에 primary row 존재
- [ ] `final/canonical_spectral_gain_curve.csv` 존재
- [ ] `block_failures.csv` 비어 있음

## G. Numerical audit

- [ ] native min/max Ritz residual 확인
- [ ] consecutive-budget `K_adam`, `K_shampoo`, `ΔG` stability 확인
- [ ] stage 간 condition metric 일치
- [ ] truncated row의 `τ` 일치
- [ ] selected-stage와 final metric 및 수치 agreement 확인
- [ ] ordinary-only promotion 확인
- [ ] factor floored fraction 확인
- [ ] partial-trace negative mass 확인
- [ ] partial-trace matrix change 확인
- [ ] clustered-subspace distance 확인
- [ ] intervention-factor change 확인
- [ ] ridge sweep sign stability 확인
- [ ] 원하는 sign을 얻기 위해 threshold나 shift를 변경하지 않음

## H. Elasticity mechanism

- [ ] `baseline_width_mismatch` 보고
- [ ] consumption-only predictor와 full proxy를 구분
- [ ] left/right/Adam `R²` 확인
- [ ] mode count와 curvature log-width 확인
- [ ] factor commutator 확인
- [ ] factor eigen residual 확인
- [ ] preconditioner floor-dominance 확인
- [ ] eligible row만 sign accuracy와 correlation에 사용
- [ ] proxy를 general noncommuting exact formula로 서술하지 않음

## I. Controls

- [ ] assignment set이 observed/aligned/reversed
- [ ] alpha set이 0.25/0.5
- [ ] `alpha_delta_from_practical`을 primary alpha contrast로 사용
- [ ] `|ΔG|` amplification을 일반 법칙으로 사용하지 않음
- [ ] damping mode가 joint/shampoo_only 모두 포함
- [ ] joint target이 `|ΔG|`
- [ ] Shampoo-only target이 `|G_shampoo|`
- [ ] Shampoo-only에서 Adam spectrum이 coefficient에 따라 변하지 않음
- [ ] aligned/reversed는 geometry gate 통과 row만 사용
- [ ] intervention의 full-covariance realizability 한계를 명시

## J. Aggregate 및 export

- [ ] balanced root 세 개를 aggregator에 전달
- [ ] `reliability_mode=balanced_canonical`
- [ ] `all_sources_balanced=true`
- [ ] `canonical_tables_used=true`
- [ ] `all_primary_metric_compatible=true`
- [ ] `all_primary_rows_numerically_accepted=true`
- [ ] `all_balanced_sources_scientifically_accepted=true`
- [ ] `minimum_seed_count_met=true`
- [ ] `no_block_failures=true`
- [ ] `compatible_protocol_hashes=true`
- [ ] `compatible_runtime_identities=true`
- [ ] `compatible_runtime_environments=true`
- [ ] `elasticity_prediction_summary.csv` 검토
- [ ] `paired_control_contrasts.csv` 검토
- [ ] scientific LaTeX export 통과
- [ ] scientific asset에 debug watermark 없음

## K. 논문 서술

- [ ] empirical mean-CE GGN이라고 명시
- [ ] eval-mode centered mini-batch-mean covariance라고 명시
- [ ] batch size와 sequence length가 estimand 일부임을 명시
- [ ] covariance와 curvature stream이 분리됨을 명시
- [ ] operator와 calibration 후 shift가 frozen임을 명시
- [ ] ordinary/truncated metric을 표와 caption에서 구분
- [ ] numerical acceptance가 rigorous enclosure가 아님을 명시
- [ ] bootstrap의 조건부 범위를 명시
- [ ] `α=0.5`가 idealized control임을 명시
- [ ] positive, negative, inconclusive block을 모두 보고
- [ ] online, stochastic-risk, generalization, systems-speed claim을 하지 않음
