# Optimizer-Visible Curvature 실험 실행서

## 1. 목표와 해석 범위

이 패키지의 primary target은 온라인 AdamW와 Shampoo의 최종 성능 순위가 아니라,
고정 checkpoint에서 측정한 block curvature와 second-moment statistic이 만드는

\[
K_{\Phi,b}
=\operatorname{cond}\!\left(P_{\Phi,b}^{1/2}H_bP_{\Phi,b}^{1/2}\right),
\qquad
\Delta G_b
=\log K_{\mathrm{Ad},b}-\log K_{\mathrm{Sh},b}
\]

입니다. 따라서 결과는 다음 순서로 해석합니다.

1. 실제 checkpoint에서 signed blockwise gain이 존재하는가.
2. matched response와 relative eigenspace geometry가 gain을 설명하는가.
3. assignment, \(\alpha\), damping을 조작하면 gain이 예측대로 변하는가.
4. frozen local dynamics에서 condition ordering이 iteration ordering으로 이어지는가.
5. 마지막으로 짧은 online continuation에서 외적 타당성을 확인한다.

## 2. 환경 설치

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "./experiments[dev]"
pytest experiments/tests -q
```

GPU를 사용할 때에는 설치된 CUDA 버전에 맞는 PyTorch wheel을 먼저 설치한 뒤
`pip install -e "./experiments[dev]"`를 실행합니다.

## 3. Stage 0: 합성 정리 재현

```bash
ovc-experiments synthetic \
  --config experiments/configs/synthetic_theorems.yaml
```

확인할 artifact:

```text
experiments/outputs/synthetic-theorems/synthetic/synthetic_results.csv
experiments/outputs/synthetic-theorems/figures/synthetic_fan.pdf
```

`flat_kron_pair` 행에서 다음 상대 오차가 수치 정밀도 수준인지 확인합니다.

\[
\frac{|K_+K_- - K_H^2|}{K_H^2}.
\]

이 단계는 코드의 수학적 회귀 테스트이며 network evidence가 아닙니다.

## 4. Stage 1: CPU pilot

### Decoder empirical Fisher

```bash
ovc-experiments smoke --config experiments/configs/smoke_decoder.yaml
```

### ViT empirical Fisher

```bash
ovc-experiments smoke --config experiments/configs/smoke_vit.yaml
```

### GGN operator 확인

Legacy dense smoke:

```bash
ovc-experiments smoke --config experiments/configs/smoke_decoder_ggn.yaml
ovc-experiments smoke --config experiments/configs/smoke_vit_ggn.yaml
```

Primary streaming path:

```bash
ovc-experiments streaming-geometry \
  --config experiments/configs/streaming_decoder_ggn.yaml
ovc-experiments streaming-geometry \
  --config experiments/configs/streaming_vit_ggn.yaml
```

Streaming command는 probe example 전체를 curvature microbatch로 재생하고,
per-example gradient를 하나씩 moment accumulator에 넣으며, 한 block의 결과를
기록한 뒤 block-local state를 해제합니다. 이미 `streaming/geometry.csv`가 존재하는
동일 `run.name`에는 덮어쓰지 않으므로 새 run 이름을 사용해야 합니다.

### 실제 Shampoo root-state 확인

```bash
ovc-experiments smoke --config experiments/configs/smoke_decoder_shampoo.yaml
```

Pilot 성공 기준은 예상 부호가 나오는 것이 아니라 다음입니다.

- 모든 prespecified block에 대해 geometry row가 생성된다.
- condition estimator가 censoring과 residual을 일관되게 기록한다.
- centered/uncentered 선택이 config와 artifact에 남는다.
- assignment/alpha/damping/grafting/finite-sample intervention이 실행된다.
- frozen dynamics와 one-block continuation artifact가 생성된다.
- 동일 seed 재실행 시 허용 오차 안에서 동일 결과를 낸다.

## 5. Stage 2: checkpoint sweep

다음 legacy checkpoint sweep은 broad diagnostic surface를 검증하는 small-block
pilot입니다. 큰 모델의 primary geometry에는 checkpoint마다 별도의 streaming config와
고유한 `run.name`을 생성하여 `streaming-geometry`를 실행합니다.

```bash
ovc-experiments checkpoint-sweep \
  --config experiments/configs/checkpoint_sweep_decoder.yaml
```

ViT:

```bash
ovc-experiments checkpoint-sweep \
  --config experiments/configs/checkpoint_sweep_vit.yaml
```

이 명령은 한 번 학습한 뒤 모든 저장 checkpoint에서 동일한 block regex와 probe
설정을 사용합니다. 통합 결과는 다음에 저장됩니다.

```text
checkpoint_sweep/geometry.csv
checkpoint_sweep/interventions.csv
checkpoint_sweep/dynamics.csv
checkpoint_sweep/continuations.csv     # sweep.run_continuations=true일 때
checkpoint_sweep/staleness.csv
figures/checkpoint_delta_gain.pdf
figures/staleness.pdf
```

### 확인 순서

1. `curvature_censored`, `adam_censored`, `shampoo_*_censored`를 먼저 확인합니다.
2. 수치적으로 신뢰할 수 있는 row에서 `delta_G_0.25`의 분포를 봅니다.
3. `response_shampoo`, `response_shampoo_spearman`,
   `projected_commutator_shampoo`, `leading_overlap_affinity`를 함께 봅니다.
4. `optimizer_state_kind`가 존재하면 population moment operator와 actual state
   operator를 구분해 비교합니다.
5. `staleness.csv`에서 `condition_ratio_to_fresh`를 checkpoint lag에 따라 봅니다.

## 6. 실제 모델 연결

### 모델 factory

```yaml
model:
  family: python
  checkpoint: /data/checkpoints/model_step_10000.pt
  kwargs:
    factory: project.experiments.build_model
    model_config: /data/configs/model.json
```

Factory signature 예시:

```python
def build_model(model_config: str) -> torch.nn.Module:
    ...
```

지원 checkpoint 형식:

```python
# OVC package
{"model_state": state_dict, "optimizer_state": ..., "step": ...}

# 일반 wrapper
{"state_dict": state_dict, "epoch": ...}

# bare state_dict
state_dict
```

외부 checkpoint key가 모델과 다르면 factory에서 모델의 parameter naming을
맞추거나 checkpoint를 변환해야 합니다. `blocks.include` regex는
`model.named_parameters()`의 이름에 적용됩니다.

### probe dataset

가장 재현 가능한 방식은 고정 probe sample을 로컬 tensor mapping으로 저장하는
것입니다.

```python
torch.save(
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    },
    "probe.pt",
)
```

```yaml
data:
  family: tensor_file
  path: /data/probes/probe.pt
  num_examples: 1024
  batch_size: 32
```

또는 Python `Dataset` factory를 지정할 수 있습니다.

### task adapter

Decoder LM:

```yaml
task:
  family: causal_lm
  input_key: input_ids
  target_key: labels
  ignore_index: -100
```

Image classification:

```yaml
task:
  family: classification
  input_key: inputs
  target_key: targets
```

모델 호출이나 loss가 다르면 `ovc_experiments.tasks.TaskAdapter` protocol을 구현해
Python factory로 반환합니다.

## 7. Confirmatory design

권장 최소 구성:

- decoder-only Transformer와 ViT;
- architecture당 두 model scale;
- seed 3개 이상;
- initialization/early/middle/late 4--5 checkpoints;
- Q/K/V/O 및 MLP up/down matrix block;
- 고정 probe set;
- centered GGN을 primary panel로 사용;
- uncentered moment, Fisher, true-Hessian sensitivity를 별도 panel로 사용.

각 model/seed/checkpoint/centeredness panel은 별도 `run.name`을 사용합니다.
Streaming geometry 파일은 canonical `delta_G` column을 사용하며 `aggregate`가 이를
legacy `delta_G_0.25`와 함께 인식합니다. 결과를 aggregate할 때 block을 독립 IID
sample로 간주하지 않고 run/checkpoint 단위 cluster를 유지합니다.

```bash
ovc-experiments aggregate \
  --geometry \
    outputs/seed-0/checkpoint_sweep/geometry.csv \
    outputs/seed-1/checkpoint_sweep/geometry.csv \
    outputs/seed-2/checkpoint_sweep/geometry.csv \
  --interventions \
    outputs/seed-0/checkpoint_sweep/interventions.csv \
    outputs/seed-1/checkpoint_sweep/interventions.csv \
    outputs/seed-2/checkpoint_sweep/interventions.csv \
  --output-dir outputs/aggregate \
  --sign-threshold 0.22314355131420976 \
  --bootstrap-replicates 5000 \
  --seed 2027
```

## 8. Curvature 설정

Primary 권장:

```yaml
curvature:
  kind: ggn
  shift: 0.001
  exact_max_dim: 512
  lanczos_steps: 64
  lanczos_starts: 4
  positive_threshold: 1.0e-10
  residual_tolerance: 1.0e-6
  subspace_policy: strict_spd
  slq_probes: 16
  slq_steps: 48
```

`shift`는 block scale에 따라 sweep해야 합니다. 보고 시 한 값만 선택하지 말고

\[
\tau_H\in\{10^{-4},10^{-3},10^{-2}\}\times
\operatorname{tr}(H_b)/d_b
\]

형태의 scale-normalized sensitivity를 권장합니다. 현재 YAML의 `shift`는 절대값이므로
block별 scale normalization이 필요하면 config를 여러 개 생성하거나 model-specific
runner factory에서 값을 설정합니다.

True Hessian은 indefinite할 수 있으므로 `kind: exact_hessian`은 작은 block의
sensitivity analysis에만 사용합니다. 기본 `strict_spd` 정책에서는 resolved null/negative
direction, uncertified minimum, singular preconditioner가 있으면 full-space condition을
censored로 처리합니다. `positive_active`는 명시적인 sensitivity panel에서만 사용하고,
optimizer마다 서로 다른 nullspace를 제거하여 비교해서는 안 됩니다.

## 9. Moment 설정

```yaml
moments:
  centered: true
  backend: loop
  accumulation_dtype: float64
  max_examples: 512
```

- `loop`: 가장 이식성이 높음.
- `vmap`: 지원되는 모델에서 빠르며, vmap rule이 없으면 loop로 fallback.
- `centered: true`: 이론에 직접 대응.
- `centered: false`: mean-gradient contamination을 포함하는 practical panel.

같은 checkpoint에서 centered와 uncentered config를 별도로 실행하고 결과를 섞지
않습니다. Model-scale path에는 다음 제한을 명시합니다.

```yaml
streaming:
  curvature_batch_size: 8
  max_factor_elements: 50000000
  run_interventions: true
  assignment_max_dim: 512
```

`max_factor_elements`는 dense left/right Shampoo factor 저장량을 제한합니다. 한도를
넘은 block은 `factor_storage_exceeds_limit`로 censored됩니다. 이는 tiling을 자동으로
근사하는 것이 아니며, embedding/output처럼 큰 factor는 구현과 동일한 explicit tile
분석을 별도로 구성해야 합니다.

## 10. Assignment와 utilization intervention

`interventions.csv`의 주요 branch:

- `assignment`: aligned, random-i, reversed;
- `alpha`: natural/aligned/reversed factor에서 \(\alpha\) sweep;
- `damping`: left/right factor별 \(\rho_L/m_L\), \(\rho_R/m_R\) sweep;
- `grafting`: positive scalar rescaling;
- `finite_sample`: nested sample-size estimate;
- `tiling`: configured tile geometry.

인과적 핵심 비교는 같은 block과 같은 factor spectrum에서 aligned와 reversed를
paired comparison하는 것입니다. H3 support에는 단순히
\(K_{\rm aligned}<K_{\rm reversed}\)만으로 충분하지 않고, 동일 scalar baseline에
대해 aligned gain이 양수이며 reversed gain이 음수인 signed reversal이 필요합니다.
Natural block에서 expected sign이 안 나왔다는 이유로 block을 제외하지 않습니다.

## 11. Frozen dynamics와 continuation

`dynamics.csv`는 같은 local quadratic과 하나의 original-coordinate error \(e_0\)에서
identity, Adam, Shampoo, actual optimizer state를 비교합니다. SPD operator에서는
\(z_0=P^{-1/2}e_0\)로 변환하여 symmetric effective problem을 실행합니다. 모든 paired
row는 동일한 `initial_error_sha256`와 `initial_objective`를 가져야 합니다. Singular 또는
censored operator에는 transformed CG/Chebyshev 결과를 만들지 않습니다. `method`는
GD, Chebyshev, CG를 구분합니다.

`continuations.csv`는 모델의 다른 parameter를 고정하고 한 block만 fixed-batch로
업데이트합니다. Step size는 measured effective maximum eigenvalue의 역수에
`continuation.step_fraction`을 곱해 정합니다. 이것은 full online training이 아니라
local external-validity bridge입니다.

## 12. 결과 판정

- H1: `delta_G_0.25`가 numerical threshold를 넘는 양·음 부호를 모두 보이는가.
- H2: held-out cluster에서 response+commutator predictor가 gain/sign을 예측하는가.
- H3: 동일 paired unit에서 aligned gain이 양수이고 reversed gain이 음수인가.
- H4: normalized factor damping 증가가 \(|G_{\rm Shampoo}|\)를 0 방향으로
  수축시키는가. Adam을 고정하면 일반적으로 \(\Delta G\to-G_{\rm Adam}\)입니다.
- H5: \(\alpha:1/4\to1/2\)가 favorable와 unfavorable regime 모두에서 같은
  부호를 유지하며 gain magnitude를 확대하는가.
- H6: one-scalar grafting이 block condition을 바꾸지 않는가.

H1/H2가 실패하면 이론의 worst-case mechanism이 해당 empirical regime에서
전형적이지 않다는 결과로 보고합니다. H3--H5가 실패하면 estimator error,
noncommutativity, root staleness, tiling 또는 mechanism mismatch를 분리해 분석합니다.

## 13. 재현성 체크리스트

- [ ] 모든 run에 `resolved_config.yaml`과 `manifest.json`이 존재한다.
- [ ] probe set과 checkpoint checksum을 보존한다.
- [ ] block regex와 제외 규칙을 사전 고정한다.
- [ ] 모든 censoring flag와 Ritz residual을 보존한다.
- [ ] centered/uncentered/actual-state 결과를 분리한다.
- [ ] left/right factor damping을 `rho_left/right_over_min/max`로 각각 보고한다.
- [ ] seed/checkpoint cluster bootstrap을 사용한다.
- [ ] 선정된 예시만이 아니라 prespecified block 전체를 공개한다.
- [ ] wall-clock 결과와 frozen condition result를 별도 panel로 둔다.
