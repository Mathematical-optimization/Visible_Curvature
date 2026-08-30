# 빠른 시작 — Canonical Balanced v1.3.0

## 1. 설치

Python 3.10 이상을 사용한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

Hugging Face checkpoint를 실행할 때는 다음 extras를 사용한다.

```bash
pip install -e '.[network,dev]'
```

CPU 전용 환경에서는 CPU용 PyTorch를 먼저 설치한 뒤 `pip install -e '.[dev]'`를 실행한다.

## 2. 오프라인 완전 검증

```bash
bash reproduce_smoke.sh
```

이 명령은 다음을 순서대로 수행한다.

1. `compileall`과 전체 `pytest`;
2. Theorem 1 및 flat-Kronecker conditioning 검증;
3. weighted-Chebyshev certificate;
4. 공통 초기점을 사용하는 integrated Theorem-3 witness;
5. tiny causal-LM frozen analysis;
6. validation, aggregation, 네 개의 retained figure;
7. debug aggregate의 scientific export 거부;
8. 명시적으로 허용된 debug export의 watermark 확인.

Balanced orchestrator 자체의 smoke는 다음과 같다.

```bash
bash reproduce_balanced_smoke.sh
```

`pipeline_status=complete`가 예상 결과다. 작은 budget 때문에 `scientific_status=inconclusive`일 수 있으며, 이는 실패가 아니다.

## 3. 합성 이론만 실행

```bash
python scripts/run_synthetic_theory.py \
  --config configs/synthetic_theory.yaml
```

다음을 확인한다.

```bash
python - <<'PY'
import json
from pathlib import Path
s = json.loads(Path('outputs/synthetic_theory/theory_summary.json').read_text())
assert s['all_checks_passed']
assert s['chebyshev_all_checks_passed']
assert s['integrated_theorem3_all_checks_passed']
print(s)
PY
```

## 4. Screening

```bash
python scripts/run_frozen_analysis.py \
  --config configs/hf_opt125m_screening.yaml
python scripts/validate_run.py \
  --output-dir outputs/hf_opt125m_screening_seed0
```

Screening에서는 bootstrap과 mechanism control을 실행하지 않는다. Block은 type, depth, memory, endpoint feasibility로만 선택하고 screening `ΔG`의 부호를 선택 기준으로 사용하지 않는다.

## 5. Confirmatory block과 control subset

배포된 `configs/hf_opt125m_confirmatory.yaml`에는 12개의 exact block이 이미 등록되어 있다.

- layer: `0, 2, 4, 6, 8, 11`;
- projection: `q_proj.weight`, `out_proj.weight`.

Intervention, alpha, damping, ridge control은 layer `0`과 `11`의 네 block에만 수행한다. 12개 block 전체에는 primary centered/observed/`α=0.25` 측정을 수행한다.

Block 집합을 바꿀 때는 `blocks.exact_names`와 `analysis.controls.block_names`를 함께 검토한다. Control subset은 primary exact-name 집합의 부분집합이어야 한다.

## 6. Seed별 balanced policy 생성

```bash
python scripts/make_balanced_policies.py \
  --base-config configs/hf_opt125m_confirmatory.yaml \
  --seeds 0 1 2 \
  --gpus 0 1 2 \
  --cpu-threads 8
```

단일 GPU에서 순차 실행할 때는 `--gpus`를 생략한다.

## 7. Balanced seed 실행

```bash
for seed in 0 1 2; do
  python scripts/run_balanced_reliability.py \
    --policy configs/generated_balanced/hf_opt125m_balanced_seed${seed}.yaml
  python scripts/validate_balanced_run.py \
    --output-root outputs/hf_opt125m_balanced_seed${seed}
done
```

각 root에는 `.balanced_run.lock`이 적용된다. 같은 output root를 두 프로세스가 동시에 사용하면 즉시 실패한다.

## 8. Seed aggregate

```bash
python scripts/aggregate_runs.py \
  --run-dir outputs/hf_opt125m_balanced_seed0 \
  --run-dir outputs/hf_opt125m_balanced_seed1 \
  --run-dir outputs/hf_opt125m_balanced_seed2 \
  --output-dir outputs/hf_opt125m_aggregate \
  --minimum-seed-count 3
```

핵심 파일:

```text
paired_seed_summary.csv
elasticity_prediction_rows.csv
elasticity_prediction_summary.csv
paired_control_contrasts.csv
aggregate_manifest.json
```

## 9. Figure와 paper asset

```bash
python scripts/make_figures.py \
  --output-dir outputs/hf_opt125m_aggregate \
  --figure-dir outputs/hf_opt125m_figures

python scripts/export_paper_assets.py \
  --output-dir outputs/hf_opt125m_aggregate \
  --paper-root paper/generated
```

Scientific export는 다음을 모두 요구한다.

- balanced canonical source;
- complete scientific seed runs;
- immutable revisions;
- 같은 protocol hash;
- 같은 runtime identity와 호환되는 runtime environment;
- block failure 없음;
- 최소 seed 수;
- primary metric compatibility;
- final numerical acceptance.

Debug 결과를 확인 목적으로 내보낼 때만 `--allow-debug-export`를 사용한다. 생성된 파일에는 debug watermark가 강제된다.

## 10. 해석 규칙

### Primary gain

\[
\Delta G=\log K_{\rm Adam}-\log K_{\rm Shampoo}.
\]

### Elasticity proxy

```text
baseline_width_mismatch
+ delta_g_predicted_consumption
= delta_g_predicted_full_proxy
```

이는 commuting–Kronecker diagnostic proxy이며 일반 block의 exact predictor가 아니다.

### Damping

- `joint`: `|ΔG|`가 target이다.
- `shampoo_only`: `|G_shampoo|`가 target이다. 일반 block에서 `ΔG`의 극한은 0이 아니라 `-G_adam`이다.

### Alpha

`alpha_delta_from_practical = ΔG(α)-ΔG(0.25)`를 사용한다. `|ΔG|` 증가를 일반 법칙으로 간주하지 않는다.

### Condition metric

- `ordinary`: 이론의 ordinary residual condition에 직접 대응한다.
- `truncated`: 상대 `τ` 위의 secondary lower-tail diagnostic이다.

기본 balanced policy에서는 truncated primary를 scientific ordering으로 승격하지 않는다.

## 11. 실제 실행 전 점검

```bash
python -m pytest -q
python -m compileall -q visible_curvature scripts tests
sha256sum -c SHA256SUMS.txt
```

그리고 한 block GPU pilot으로 다음을 기록한다.

- GGN matvec 시간;
- peak GPU memory;
- 64/96/128/192/256-step endpoint 시간;
- cache hit/miss;
- partial-trace probe 시간;
- ridge 및 condition metric 상태.
