from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class SyntheticLanguageDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic modular-sequence next-token prediction dataset."""

    def __init__(
        self,
        *,
        num_examples: int = 128,
        seq_len: int = 12,
        vocab_size: int = 32,
        seed: int = 0,
    ) -> None:
        if vocab_size < 4:
            raise ValueError("vocab_size must be at least four")
        generator = torch.Generator().manual_seed(seed)
        starts = torch.randint(0, vocab_size, (num_examples,), generator=generator)
        steps = torch.randint(1, max(2, vocab_size // 4), (num_examples,), generator=generator)
        positions = torch.arange(seq_len + 1)
        sequences = (starts[:, None] + steps[:, None] * positions[None, :]) % vocab_size
        self.input_ids = sequences[:, :-1].to(torch.long)
        self.labels = sequences[:, 1:].to(torch.long)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[index], "labels": self.labels[index]}


class SyntheticVisionDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic class-specific patch dataset for CPU smoke tests."""

    def __init__(
        self,
        *,
        num_examples: int = 128,
        image_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 4,
        noise_std: float = 0.15,
        seed: int = 0,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        generator = torch.Generator().manual_seed(seed)
        targets = torch.arange(num_examples) % num_classes
        images = noise_std * torch.randn(
            num_examples,
            in_channels,
            image_size,
            image_size,
            generator=generator,
        )
        grid = int(torch.ceil(torch.sqrt(torch.tensor(float(num_classes)))).item())
        patch = max(1, image_size // (2 * grid))
        for index, target in enumerate(targets.tolist()):
            row = target // grid
            col = target % grid
            row_start = min(image_size - patch, row * 2 * patch)
            col_start = min(image_size - patch, col * 2 * patch)
            channel = target % in_channels
            images[index, channel, row_start : row_start + patch, col_start : col_start + patch] += 2.0
        self.images = images
        self.targets = targets.to(torch.long)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"inputs": self.images[index], "targets": self.targets[index]}


class TensorFileDataset(Dataset[dict[str, torch.Tensor]]):
    """Load a dictionary of equally sized tensors from a local `.pt` file."""

    def __init__(self, path: str | Path) -> None:
        payload: Any = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("Tensor dataset file must contain a nonempty tensor mapping")
        if not all(isinstance(value, torch.Tensor) for value in payload.values()):
            raise ValueError("Every tensor dataset value must be a torch.Tensor")
        lengths = {int(value.shape[0]) for value in payload.values()}
        if len(lengths) != 1:
            raise ValueError("All tensors must share their leading dimension")
        self.payload = payload
        self.length = lengths.pop()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.payload.items()}
