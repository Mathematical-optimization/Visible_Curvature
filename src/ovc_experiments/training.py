from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .optim import MatrixShampoo
from .tasks import Batch, TaskAdapter


@dataclass
class TrainingResult:
    losses: list[float]
    checkpoints: list[Path]
    optimizer_name: str


def move_batch(
    batch: Any,
    device: torch.device | str,
    *,
    dtype: torch.dtype | None = None,
) -> Any:
    if isinstance(batch, torch.Tensor):
        target_dtype = dtype if dtype is not None and batch.is_floating_point() else batch.dtype
        return batch.to(device=device, dtype=target_dtype)
    if isinstance(batch, dict):
        return {key: move_batch(value, device, dtype=dtype) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch(value, device, dtype=dtype) for value in batch)
    if isinstance(batch, list):
        return [move_batch(value, device, dtype=dtype) for value in batch]
    return batch


def _build_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    alpha: float,
    root_frequency: int,
    grafting: str,
) -> torch.optim.Optimizer:
    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay,
        )
    if optimizer_name.lower() == "shampoo":
        return MatrixShampoo(
            model.parameters(),
            lr=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            alpha=alpha,
            root_frequency=root_frequency,
            weight_decay=weight_decay,
            grafting=grafting,
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    losses: list[float],
    optimizer_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": int(step),
            "losses": list(losses),
            "optimizer_name": optimizer_name,
            "parameter_names": [name for name, _parameter in model.named_parameters()],
        },
        path,
    )


def train_and_checkpoint(
    model: torch.nn.Module,
    dataset: Dataset,
    task: TaskAdapter,
    *,
    output_dir: str | Path,
    optimizer_name: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_steps: list[int],
    seed: int,
    device: str | torch.device = "cpu",
    beta1: float = 0.9,
    beta2: float = 0.99,
    epsilon: float = 1e-8,
    alpha: float = 0.25,
    root_frequency: int = 5,
    grafting: str = "none",
) -> TrainingResult:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    torch.manual_seed(seed)
    device = torch.device(device)
    model.to(device)
    model.train()
    model_dtype = next(
        (parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()),
        None,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    batches = cycle(loader)
    optimizer = _build_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        alpha=alpha,
        root_frequency=root_frequency,
        grafting=grafting,
    )
    output_path = Path(output_dir)
    requested = sorted(set(int(value) for value in checkpoint_steps if 0 <= value <= steps))
    if 0 not in requested:
        requested.insert(0, 0)
    if steps not in requested:
        requested.append(steps)
    checkpoints: list[Path] = []
    losses: list[float] = []

    def save(step: int) -> None:
        path = output_path / f"checkpoint_step_{step:06d}.pt"
        _save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            step=step,
            losses=losses,
            optimizer_name=optimizer_name,
        )
        checkpoints.append(path)

    save(0)
    for step in range(1, steps + 1):
        batch: Batch = move_batch(next(batches), device, dtype=model_dtype)
        optimizer.zero_grad(set_to_none=True)
        args, kwargs = task.call_spec(batch)
        logits = task.extract_logits(model(*args, **kwargs))
        loss = task.loss_per_example_from_logits(logits, batch).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if step in requested:
            save(step)

    return TrainingResult(losses=losses, checkpoints=checkpoints, optimizer_name=optimizer_name)


def _looks_like_state_dict(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and bool(payload)
        and all(isinstance(key, str) for key in payload)
        and all(isinstance(value, torch.Tensor) for value in payload.values())
    )


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    raw_payload: Any = torch.load(
        Path(path), map_location=map_location, weights_only=False
    )

    if isinstance(raw_payload, dict) and "model_state" in raw_payload:
        state_dict = raw_payload["model_state"]
        metadata = dict(raw_payload)
        metadata.setdefault("checkpoint_format", "ovc")
    elif isinstance(raw_payload, dict) and _looks_like_state_dict(
        raw_payload.get("state_dict")
    ):
        state_dict = raw_payload["state_dict"]
        metadata = dict(raw_payload)
        metadata["checkpoint_format"] = "state_dict_wrapper"
    elif _looks_like_state_dict(raw_payload):
        state_dict = raw_payload
        metadata = {"checkpoint_format": "bare_state_dict"}
    else:
        raise ValueError(
            "Unsupported checkpoint format: expected an OVC payload with "
            "'model_state', a wrapper with 'state_dict', or a bare state_dict."
        )

    model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and "optimizer_state" in metadata:
        optimizer.load_state_dict(metadata["optimizer_state"])
    return metadata
