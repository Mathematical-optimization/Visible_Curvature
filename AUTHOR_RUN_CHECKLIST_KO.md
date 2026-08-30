# Author Run Checklist — Canonical Balanced v1.2.0

## A. 코드 및 release

- [ ] `python -m pytest -q` 통과
- [ ] `python -m compileall -q visible_curvature scripts tests` 통과
- [ ] `bash reproduce_smoke.sh` 통과
- [ ] `bash reproduce_balanced_smoke.sh`에서 `pipeline_status=complete`
- [ ] `sha256sum -c SHA256SUMS.txt` 통과
- [ ] `visible_curvature.__version__`과 `pyproject.toml`이 `1.2.0`
- [ ] `outputs/synthetic_theory/theory_summary.json`의 `all_checks_passed=true`
- [ ] `chebyshev_all_checks_passed=true`

## B. 환경 및 계산량

- [ ] Python/PyTorch/CUDA/driver/GPU 정보를 run log에 기록
- [ ] 한 block pilot으로 GGN matvec 시간과 peak memory 측정
- [ ] seed별 동일 GPU architecture 사용 여부 기록
- [ ] output/log/checkpoint 저장 공간 확보

## C. Scientific provenance

- [ ] model revision이 40자리 immutable commit
- [ ] tokenizer revision이 40자리 immutable commit
- [ ] dataset revision이 40자리 immutable commit
- [ ] 모든 seed에서 `data.order_seed` 동일
- [ ] 모든 seed에서 exact block set 동일
- [ ] covariance/curvature batch interval 동일하고 비중첩
- [ ] protocol hash와 runtime identity가 seed 간 일치

## D. Screening

- [ ] bootstrap 비활성화
- [ ] intervention/alpha/damping controls 비활성화
- [ ] block 선택 기준을 type/depth/numerical feasibility로 사전 기록
- [ ] screening `ΔG` sign을 선택 기준으로 사용하지 않음
- [ ] confirmatory exact-block YAML을 결과 확인 전에 동결

## E. Balanced policy

- [ ] 최소 세 seed policy 생성
- [ ] `fixed_relative_ridge` 사전 고정
- [ ] endpoint steps/starts schedule 사전 고정
- [ ] partial-trace probe schedule 사전 고정
- [ ] matrix/subspace/intervention-factor tolerance 사전 고정
- [ ] lower-tail refinement 조건 사전 고정
- [ ] final bootstrap replicate 수 100
- [ ] covariance batch 수가 group size로 나누어짐

## F. Seed-level run audit

- [ ] `COMPLETED` 존재
- [ ] `curvature_shift_overrides.json` 존재
- [ ] 모든 post-calibration stage가 같은 shift override digest 사용
- [ ] `endpoint_convergence.csv` 검토
- [ ] `partial_trace_convergence.csv` 검토
- [ ] `balanced_reliability_certificates.csv` 검토
- [ ] `final/scientific_status.json` 검토
- [ ] `final/canonical_block_metrics.csv`에 primary row 존재
- [ ] `final/canonical_spectral_gain_curve.csv` 존재
- [ ] `block_failures.csv` 비어 있음

## G. Numerical audit

- [ ] raw endpoint와 applied shift 확인
- [ ] min/max Ritz residual 확인
- [ ] selected-stage와 final `K_adam`, `K_shampoo`, `ΔG` agreement 확인
- [ ] ordinary/truncated metric과 saturation 확인
- [ ] tau classification 확인
- [ ] factor floored fraction 확인
- [ ] partial-trace negative mass 확인
- [ ] partial-trace matrix relative change 확인
- [ ] clustered-subspace projector distance 확인
- [ ] aligned/reversed intervention-factor change 확인
- [ ] failed gate를 임의로 완화하거나 재실행하여 sign을 선택하지 않음

## H. Controls

- [ ] assignment set이 observed/aligned/reversed
- [ ] alpha set이 0.25/0.5
- [ ] damping mode가 joint/shampoo_only 모두 포함
- [ ] shampoo_only에서 Adam spectrum이 coefficient에 따라 변하지 않음
- [ ] aligned/reversed inference는 partial-trace geometry gate를 통과한 row만 사용
- [ ] uncentered result를 primary로 사용하지 않음

## I. Aggregation 및 export

- [ ] balanced root 세 개를 aggregator에 전달
- [ ] `reliability_mode=balanced_canonical`
- [ ] `all_sources_balanced=true`
- [ ] `canonical_tables_used=true`
- [ ] `all_primary_rows_numerically_accepted=true`
- [ ] `all_balanced_sources_scientifically_accepted=true`
- [ ] `minimum_seed_count_met=true`
- [ ] `no_block_failures=true`
- [ ] `compatible_protocol_hashes=true`
- [ ] `compatible_runtime_identities=true`
- [ ] `immutable_revisions=true`
- [ ] scientific LaTeX export 통과
- [ ] scientific asset에 debug watermark가 없음

## J. 논문 서술

- [ ] empirical GGN curvature라고 명시
- [ ] centered mini-batch-mean gradient covariance라고 명시
- [ ] operator와 shift가 frozen이라고 명시
- [ ] numerical acceptance가 rigorous eigenvalue enclosure가 아님을 명시
- [ ] bootstrap이 calibrated low-budget grouped bootstrap임을 명시
- [ ] `α=0.5`가 idealized control임을 명시
- [ ] aligned/reversed intervention의 realizability 한계를 명시
- [ ] positive, negative, inconclusive block을 모두 보고
- [ ] online dominance, stochastic-risk dominance, generalization, systems-speed claim을 하지 않음
