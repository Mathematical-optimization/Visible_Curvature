from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from ovc_experiments.data import SyntheticLanguageDataset, SyntheticVisionDataset
from ovc_experiments.models import TinyDecoderLM, TinyVisionTransformer
from ovc_experiments.optim import MatrixShampoo
from ovc_experiments.tasks import ClassificationTask
from ovc_experiments.training import load_checkpoint, train_and_checkpoint


def test_tiny_models_expose_expected_logits_and_matrix_blocks() -> None:
    decoder = TinyDecoderLM(
        vocab_size=17,
        max_seq_len=8,
        d_model=12,
        n_heads=3,
        n_layers=1,
        mlp_ratio=2,
        dropout=0.0,
    )
    tokens = torch.randint(0, 17, (2, 6))
    logits = decoder(tokens)
    assert logits.shape == (2, 6, 17)
    decoder_names = dict(decoder.named_parameters())
    assert "blocks.0.q_proj.weight" in decoder_names
    assert "blocks.0.up_proj.weight" in decoder_names

    vit = TinyVisionTransformer(
        image_size=8,
        patch_size=4,
        in_channels=1,
        num_classes=3,
        d_model=12,
        n_heads=3,
        n_layers=1,
        mlp_ratio=2,
        dropout=0.0,
    )
    images = torch.randn(2, 1, 8, 8)
    class_logits = vit(images)
    assert class_logits.shape == (2, 3)
    vit_names = dict(vit.named_parameters())
    assert "patch_embed.weight" in vit_names
    assert "blocks.0.v_proj.weight" in vit_names


def test_synthetic_datasets_are_deterministic() -> None:
    language_a = SyntheticLanguageDataset(num_examples=5, seq_len=6, vocab_size=13, seed=7)
    language_b = SyntheticLanguageDataset(num_examples=5, seq_len=6, vocab_size=13, seed=7)
    assert torch.equal(language_a[3]["input_ids"], language_b[3]["input_ids"])
    assert torch.equal(language_a[3]["labels"], language_b[3]["labels"])

    vision_a = SyntheticVisionDataset(
        num_examples=6, image_size=8, in_channels=1, num_classes=3, seed=9
    )
    vision_b = SyntheticVisionDataset(
        num_examples=6, image_size=8, in_channels=1, num_classes=3, seed=9
    )
    assert torch.equal(vision_a[4]["inputs"], vision_b[4]["inputs"])
    assert vision_a[4]["targets"].item() == vision_b[4]["targets"].item()


def test_matrix_shampoo_builds_factor_and_root_state() -> None:
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer = MatrixShampoo(
        model.parameters(),
        lr=0.05,
        beta1=0.0,
        beta2=0.9,
        epsilon=1e-4,
        alpha=0.25,
        root_frequency=1,
        grafting="none",
    )
    inputs = torch.tensor([[1.0, 2.0, -1.0], [0.5, -1.0, 2.0]])
    targets = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    optimizer.step()

    state = optimizer.state[model.weight]
    assert state["left_factor"].shape == (2, 2)
    assert state["right_factor"].shape == (3, 3)
    assert state["left_root"].shape == (2, 2)
    assert state["right_root"].shape == (3, 3)
    assert torch.isfinite(model.weight).all()


class _LinearlySeparableDataset(Dataset):
    def __init__(self) -> None:
        self.inputs = torch.tensor(
            [[2.0, 0.0], [1.0, 0.2], [-2.0, 0.0], [-1.0, -0.2]], dtype=torch.float32
        )
        self.targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"inputs": self.inputs[index], "targets": self.targets[index]}


def test_train_and_checkpoint_saves_loadable_states_and_reduces_loss(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    result = train_and_checkpoint(
        model,
        _LinearlySeparableDataset(),
        ClassificationTask(),
        output_dir=tmp_path,
        optimizer_name="adamw",
        steps=20,
        batch_size=4,
        learning_rate=0.1,
        weight_decay=0.0,
        checkpoint_steps=[0, 10, 20],
        seed=3,
    )
    assert result.losses[-1] < result.losses[0]
    assert sorted(path.name for path in result.checkpoints) == [
        "checkpoint_step_000000.pt",
        "checkpoint_step_000010.pt",
        "checkpoint_step_000020.pt",
    ]

    restored = torch.nn.Linear(2, 2, bias=False)
    metadata = load_checkpoint(restored, result.checkpoints[-1])
    assert metadata["step"] == 20
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(left, right)


def test_train_and_checkpoint_casts_floating_inputs_to_model_dtype(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
    result = train_and_checkpoint(
        model,
        _LinearlySeparableDataset(),
        ClassificationTask(),
        output_dir=tmp_path,
        optimizer_name="adamw",
        steps=1,
        batch_size=4,
        learning_rate=0.05,
        weight_decay=0.0,
        checkpoint_steps=[0, 1],
        seed=5,
    )
    assert len(result.losses) == 1
    assert torch.isfinite(torch.tensor(result.losses)).all()


def test_extract_frozen_optimizer_state_preconditioners(tmp_path: Path) -> None:
    from ovc_experiments.blocks import discover_matrix_blocks
    from ovc_experiments.optimizer_state import extract_frozen_preconditioner

    dataset = _LinearlySeparableDataset()
    task = ClassificationTask()

    adam_model = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
    adam_result = train_and_checkpoint(
        adam_model,
        dataset,
        task,
        output_dir=tmp_path / "adam",
        optimizer_name="adamw",
        steps=1,
        batch_size=4,
        learning_rate=0.05,
        weight_decay=0.0,
        checkpoint_steps=[1],
        seed=7,
    )
    adam_payload = torch.load(adam_result.checkpoints[-1], map_location="cpu", weights_only=False)
    adam_block = discover_matrix_blocks(adam_model, include=[r"weight$"])[0]
    adam_snapshot = extract_frozen_preconditioner(adam_payload, adam_model, adam_block)
    assert adam_snapshot is not None
    adam_state = next(iter(adam_payload["optimizer_state"]["state"].values()))
    group = adam_payload["optimizer_state"]["param_groups"][0]
    step = int(adam_state["step"].item())
    beta2 = float(group["betas"][1])
    expected_weights = 1.0 / (
        (adam_state["exp_avg_sq"] / (1.0 - beta2**step)).sqrt() + float(group["eps"])
    )
    vector = torch.ones(adam_block.numel, dtype=torch.float64)
    assert torch.allclose(
        adam_snapshot.preconditioner.apply(vector), expected_weights.reshape(-1)
    )

    shampoo_model = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
    shampoo_result = train_and_checkpoint(
        shampoo_model,
        dataset,
        task,
        output_dir=tmp_path / "shampoo",
        optimizer_name="shampoo",
        steps=1,
        batch_size=4,
        learning_rate=0.05,
        weight_decay=0.0,
        checkpoint_steps=[1],
        seed=7,
        beta1=0.0,
        beta2=0.9,
        epsilon=1e-4,
        root_frequency=1,
        grafting="none",
    )
    shampoo_payload = torch.load(
        shampoo_result.checkpoints[-1], map_location="cpu", weights_only=False
    )
    shampoo_block = discover_matrix_blocks(shampoo_model, include=[r"weight$"])[0]
    shampoo_snapshot = extract_frozen_preconditioner(
        shampoo_payload, shampoo_model, shampoo_block
    )
    assert shampoo_snapshot is not None
    shampoo_state = next(iter(shampoo_payload["optimizer_state"]["state"].values()))
    expected = (
        shampoo_state["left_root"]
        @ vector.reshape(shampoo_block.layout.matrix_shape)
        @ shampoo_state["right_root"]
    ).reshape(-1)
    assert torch.allclose(shampoo_snapshot.preconditioner.apply(vector), expected)


def test_load_checkpoint_accepts_bare_and_wrapped_state_dicts(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 2, bias=False)
    bare_path = tmp_path / "bare.pt"
    wrapped_path = tmp_path / "wrapped.pt"
    torch.save(source.state_dict(), bare_path)
    torch.save({"state_dict": source.state_dict(), "epoch": 3}, wrapped_path)

    bare_target = torch.nn.Linear(2, 2, bias=False)
    bare_metadata = load_checkpoint(bare_target, bare_path)
    wrapped_target = torch.nn.Linear(2, 2, bias=False)
    wrapped_metadata = load_checkpoint(wrapped_target, wrapped_path)

    assert bare_metadata["checkpoint_format"] == "bare_state_dict"
    assert wrapped_metadata["checkpoint_format"] == "state_dict_wrapper"
    assert wrapped_metadata["epoch"] == 3
    assert torch.allclose(source.weight, bare_target.weight)
    assert torch.allclose(source.weight, wrapped_target.weight)
