# 빠른 시작 — Canonical Balanced v1.2.0

## 1. 설치

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[network,dev]'
```

합성 검증과 단위 테스트만 수행할 때는 다음으로 충분하다.

```bash
pip install -e '.[dev]'
```

## 2. 전체 소프트웨어 검증

기본 smoke workflow:

```bash
bash reproduce_smoke.sh
```

Balanced orchestrator smoke workflow:

```bash
bash reproduce_balanced_smoke.sh
```

Balanced smoke는 실행 경로와 산출물 계약을 검사한다. `pipeline_status=complete`이면 실행이 정상 종료된 것이며, 작은 수치 budget 때문에 `scientific_status=inconclusive`일 수 있다.

## 3. 합성 정리 및 Chebyshev 검증

```bash
python scripts/run_synthetic_theory.py \
  --config configs/synthetic_theory.yaml
```

확인할 파일:

- `outputs/synthetic_theory/theorem1_conditioning_results.csv`
- `outputs/synthetic_theory/flat_kronecker_conditioning_results.csv`
- `outputs/synthetic_theory/chebyshev_certificates.csv`
- `outputs/synthetic_theory/theory_summary.json`

`theory_summary.json`의 `all_checks_passed`와 `chebyshev_all_checks_passed`가 모두 `true`여야 한다.

## 4. Screening

```bash
python scripts/run_frozen_analysis.py \
  --config configs/hf_opt125m_screening.yaml
python scripts/validate_run.py \
  --output-dir outputs/hf_opt125m_screening_seed0
```

Screening은 confirmatory block의 유형과 깊이를 정하는 단계다. Bootstrap과 mechanism control을 수행하지 않으며, `ΔG`의 부호를 보고 유리한 block만 선택해서는 안 된다.

## 5. Exact-block confirmatory config 준비

Screening에서 정한 최대 두 block을 `hf_opt125m_confirmatory.yaml` 형식의 별도 파일에 고정한다. 예:

```text
configs/generated/hf_opt125m_confirmatory_exact_blocks.yaml
```

Model, tokenizer, dataset revision은 40자리 immutable commit으로 고정하고 `data.order_seed`도 고정한다.

## 6. Seed별 balanced policy 생성

```bash
python scripts/make_balanced_policies.py \
  --base-config configs/generated/hf_opt125m_confirmatory_exact_blocks.yaml \
  --seeds 0 1 2
```

## 7. Balanced confirmatory 실행

```bash
for seed in 0 1 2; do
  python scripts/run_balanced_reliability.py \
    --policy configs/generated_balanced/hf_opt125m_balanced_seed${seed}.yaml
  python scripts/validate_balanced_run.py \
    --output-root outputs/hf_opt125m_balanced_seed${seed}
done
```

간단한 요약:

```bash
python scripts/summarize_balanced_results.py \
  --output-root outputs/hf_opt125m_balanced_seed0
```

Primary endpoint는 centered covariance, observed assignment, `α=0.25`에서

\[
\Delta G=\log K_{\mathrm{Adam}}-\log K_{\mathrm{Shampoo}}
\]

이다. `canonical_block_metrics.csv`만 seed-level primary inference에 사용한다.

## 8. 세 seed 집계

```bash
python scripts/aggregate_runs.py \
  --run-dir outputs/hf_opt125m_balanced_seed0 \
  --run-dir outputs/hf_opt125m_balanced_seed1 \
  --run-dir outputs/hf_opt125m_balanced_seed2 \
  --output-dir outputs/hf_opt125m_balanced_aggregate \
  --minimum-seed-count 3 \
  --make-figures
```

`aggregate_manifest.json`에서 다음을 확인한다.

- `reliability_mode = balanced_canonical`
- `all_sources_balanced = true`
- `canonical_tables_used = true`
- `all_primary_rows_numerically_accepted = true`
- `all_balanced_sources_scientifically_accepted = true`
- `minimum_seed_count_met = true`
- `no_block_failures = true`
- `compatible_protocol_hashes = true`
- `compatible_runtime_identities = true`
- `immutable_revisions = true`

하나라도 충족하지 못하면 scientific export가 실패하는 것이 정상이다.

## 9. 논문용 export

```bash
python scripts/export_paper_assets.py \
  --output-dir outputs/hf_opt125m_balanced_aggregate \
  --paper-root /absolute/path/to/paper/experiments
```

Legacy 또는 debug aggregate는 scientific export 대상이 아니다. 형식 점검용 debug export만 `--allow-debug-export`를 사용하며, 결과에는 명시적인 watermark가 들어간다.
