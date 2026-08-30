from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

SUPPORTED_DATA_BACKENDS = ("synthetic_tokens", "hf_text")


class SyntheticTokenDataset(Dataset):
    def __init__(self, n: int, seq_len: int, vocab_size: int, seed: int):
        generator = torch.Generator().manual_seed(int(seed))
        self.tokens = torch.randint(0, vocab_size, (n, seq_len), generator=generator)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = self.tokens[index]
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "attention_mask": torch.ones_like(tokens),
        }


class PackedTokenDataset(Dataset):
    def __init__(self, chunks: list[torch.Tensor]):
        if not chunks:
            raise ValueError("PackedTokenDataset requires at least one chunk")
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = self.chunks[index]
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "attention_mask": torch.ones_like(tokens),
        }


def deterministic_index_order(length: int, seed: int, shuffle: bool) -> list[int]:
    if length < 0:
        raise ValueError("length must be nonnegative")
    if not shuffle:
        return list(range(length))
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(length, generator=generator).tolist()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _order_hash(order: Sequence[int]) -> str:
    return _sha256_json([int(index) for index in order])


def _integer_stream_hash(values: Sequence[int]) -> str:
    return _sha256_json([int(value) for value in values])


def _text_sequence_hash(values: Sequence[str]) -> str:
    return _sha256_json([str(value) for value in values])


def _tensor_sequence_hash(values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in values:
        material = tensor.detach().cpu().contiguous()
        digest.update(str(material.dtype).encode("utf-8"))
        digest.update(_sha256_json(list(material.shape)).encode("ascii"))
        digest.update(material.numpy().tobytes())
    return digest.hexdigest()


def _ordered_subset(
    dataset: Dataset,
    *,
    order_seed: int,
    shuffle: bool,
    max_examples: int | None = None,
) -> tuple[Subset, list[int]]:
    order = deterministic_index_order(len(dataset), seed=order_seed, shuffle=shuffle)
    if max_examples is not None:
        order = order[: min(int(max_examples), len(order))]
    return Subset(dataset, order), order


def _collate_mapping(batch: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def _loader_factory(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    collate_fn: Callable | None = None,
) -> Callable[[], DataLoader]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")

    def factory() -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    return factory


def _manifest_metadata(name: str, order: Sequence[int], **extra: Any) -> dict[str, Any]:
    return {
        "dataset_name": name,
        "sample_order_sha256": _order_hash(order),
        "sample_count": len(order),
        "_sample_indices": list(order),
        **extra,
    }


def _resolved_tokenizer_commit(tokenizer: Any) -> str | None:
    direct = getattr(tokenizer, "_commit_hash", None)
    if direct:
        return str(direct)
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if isinstance(init_kwargs, Mapping) and init_kwargs.get("_commit_hash"):
        return str(init_kwargs["_commit_hash"])
    return None


def build_dataloader_factory(
    cfg: Mapping[str, Any],
    model_backend: str,
    seed: int,
    model_metadata: Mapping[str, Any] | None = None,
) -> tuple[Callable[[], Iterable], Dict[str, Any]]:
    """Build a deterministic data stream and a content-level provenance manifest.

    ``seed`` is the experiment seed and intentionally does not control data
    order.  Data identity is governed by ``data.order_seed`` and, for the
    synthetic backend, ``data.content_seed``.
    """
    del model_metadata, seed
    if model_backend not in {"tiny_causal_lm", "hf_causal_lm"}:
        raise ValueError("Text data require a causal-LM model backend")
    data_cfg = cfg.get("data", {})
    backend = str(data_cfg.get("backend", "synthetic_tokens"))
    if backend not in SUPPORTED_DATA_BACKENDS:
        supported = ", ".join(SUPPORTED_DATA_BACKENDS)
        raise ValueError(f"Supported data backends are: {supported}; got {backend}")

    batch_size = int(data_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 0))
    shuffle = bool(data_cfg.get("shuffle", False))
    max_examples_cfg = data_cfg.get("max_examples")
    order_seed = int(data_cfg.get("order_seed", 0))

    if backend == "synthetic_tokens":
        content_seed = int(data_cfg.get("content_seed", 0))
        dataset = SyntheticTokenDataset(
            n=int(data_cfg.get("num_examples", 64)),
            seq_len=int(data_cfg.get("seq_len", 16)),
            vocab_size=int(data_cfg.get("vocab_size", cfg.get("model", {}).get("vocab_size", 128))),
            seed=content_seed,
        )
        ordered, order = _ordered_subset(
            dataset,
            order_seed=order_seed,
            shuffle=shuffle,
            max_examples=max_examples_cfg,
        )
        selected = [dataset.tokens[index] for index in order]
        content_hash = _tensor_sequence_hash(selected)
        return (
            _loader_factory(ordered, batch_size, num_workers, _collate_mapping),
            _manifest_metadata(
                "synthetic_tokens",
                order,
                order_seed=order_seed,
                content_seed=content_seed,
                source_order_sha256=_order_hash(order),
                packed_token_stream_sha256=content_hash,
                selected_chunk_content_sha256=content_hash,
                dataset_revision=None,
                tokenizer_revision=None,
                resolved_tokenizer_commit=None,
            ),
        )

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError("Install datasets and transformers for the hf_text backend") from error

    name = str(data_cfg["name"])
    subset = data_cfg.get("subset")
    split = str(data_cfg.get("split", "train"))
    text_field = str(data_cfg.get("text_field", "text"))
    model_name = str(cfg.get("model", {}).get("name"))
    tokenizer_name = str(data_cfg.get("tokenizer", model_name))
    tokenizer_revision = data_cfg.get("tokenizer_revision", cfg.get("model", {}).get("revision"))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        revision=tokenizer_revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_revision = data_cfg.get("revision")
    raw_dataset = load_dataset(name, subset, split=split, revision=dataset_revision)
    dataset_fingerprint = getattr(raw_dataset, "_fingerprint", None)
    max_rows = int(data_cfg.get("max_source_rows", min(4096, len(raw_dataset))))
    source_order = deterministic_index_order(
        len(raw_dataset), seed=order_seed, shuffle=shuffle
    )[:max_rows]
    selected_dataset = raw_dataset.select(source_order)
    sequence_length = int(data_cfg.get("seq_len", 128))
    packing = bool(data_cfg.get("packing", True))
    resolved_tokenizer_commit = _resolved_tokenizer_commit(tokenizer)

    if packing:
        texts = [
            str(row[text_field])
            for row in selected_dataset
            if str(row[text_field]).strip()
        ]
        if not texts:
            raise RuntimeError(f"No non-empty text rows found in {name}/{split}")
        encoded = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
        eos = tokenizer.eos_token_id
        stream: list[int] = []
        for token_ids in encoded["input_ids"]:
            stream.extend(int(value) for value in token_ids)
            if eos is not None:
                stream.append(int(eos))
        usable = (len(stream) // sequence_length) * sequence_length
        if usable < sequence_length:
            raise RuntimeError(f"Tokenized corpus is too short for seq_len={sequence_length}")
        chunks = [
            torch.tensor(stream[offset : offset + sequence_length], dtype=torch.long)
            for offset in range(0, usable, sequence_length)
        ]
        packed = PackedTokenDataset(chunks)
        order = list(range(len(packed)))
        if max_examples_cfg is not None:
            order = order[: min(int(max_examples_cfg), len(order))]
        selected_chunks = [chunks[index] for index in order]
        ordered = Subset(packed, order)
        return (
            _loader_factory(ordered, batch_size, num_workers, _collate_mapping),
            _manifest_metadata(
                name,
                order,
                order_seed=order_seed,
                dataset_subset=subset,
                dataset_split=split,
                text_packing=True,
                source_order_sha256=_order_hash(source_order),
                packed_token_stream_sha256=_integer_stream_hash(stream),
                selected_chunk_content_sha256=_tensor_sequence_hash(selected_chunks),
                source_text_content_sha256=_text_sequence_hash(texts),
                tokenizer_name=tokenizer_name,
                tokenizer_revision=tokenizer_revision,
                resolved_tokenizer_commit=resolved_tokenizer_commit,
                dataset_revision=dataset_revision,
                dataset_fingerprint=dataset_fingerprint,
            ),
        )

    ordered, order = _ordered_subset(
        selected_dataset,
        order_seed=order_seed,
        shuffle=False,
        max_examples=max_examples_cfg,
    )
    selected_texts = [str(selected_dataset[index][text_field]) for index in order]

    def collate(rows: list[Mapping[str, Any]]) -> Mapping[str, torch.Tensor]:
        texts = [str(row[text_field]) for row in rows]
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
            return_tensors="pt",
        )
        labels = tokenized["input_ids"].clone()
        labels[tokenized["attention_mask"] == 0] = -100
        tokenized["labels"] = labels
        return tokenized

    content_hash = _text_sequence_hash(selected_texts)
    return (
        _loader_factory(ordered, batch_size, num_workers, collate),
        _manifest_metadata(
            name,
            order,
            order_seed=order_seed,
            dataset_subset=subset,
            dataset_split=split,
            text_packing=False,
            source_order_sha256=_order_hash(source_order),
            packed_token_stream_sha256=None,
            selected_chunk_content_sha256=content_hash,
            source_text_content_sha256=content_hash,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            resolved_tokenizer_commit=resolved_tokenizer_commit,
            dataset_revision=dataset_revision,
            dataset_fingerprint=dataset_fingerprint,
        ),
    )


def take_batches(factory: Callable[[], Iterable], n: int) -> list[Any]:
    return list(itertools.islice(iter(factory()), n))
