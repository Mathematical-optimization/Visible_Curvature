from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from visible_curvature.data import build_dataloader_factory


class _Dataset:
    _fingerprint = "fingerprint-v1"

    def __init__(self, rows=None):
        self.rows = rows or [
            {"text": "row zero"},
            {"text": "row one"},
            {"text": "row two"},
            {"text": "row three"},
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)

    def select(self, order):
        return _Dataset([self.rows[index] for index in order])


class _Tokenizer:
    pad_token = None
    eos_token = "<eos>"
    eos_token_id = 99
    init_kwargs = {"_commit_hash": "a" * 40}

    def __call__(self, texts, **kwargs):
        values = []
        for text in texts:
            row_id = int(text.split()[-1].replace("zero", "0").replace("one", "1").replace("two", "2").replace("three", "3"))
            values.append([row_id + 1, row_id + 11, row_id + 21])
        if kwargs.get("return_tensors") == "pt":
            length = int(kwargs["max_length"])
            ids = torch.zeros((len(values), length), dtype=torch.long)
            for i, tokens in enumerate(values):
                ids[i, : min(length, len(tokens))] = torch.tensor(tokens[:length])
            return {"input_ids": ids, "attention_mask": (ids != 0).long()}
        return {"input_ids": values}


class _AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return _Tokenizer()


def _install_fake_hf(monkeypatch):
    def load_dataset(*args, **kwargs):
        return _Dataset()

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=_AutoTokenizer))


def _cfg(order_seed: int):
    commit = "b" * 40
    return {
        "model": {"name": "fake/model", "revision": commit},
        "data": {
            "backend": "hf_text",
            "name": "fake/data",
            "revision": "c" * 40,
            "tokenizer_revision": commit,
            "split": "train",
            "text_field": "text",
            "packing": True,
            "seq_len": 4,
            "batch_size": 1,
            "shuffle": True,
            "order_seed": order_seed,
            "max_source_rows": 4,
        },
    }


def test_experiment_seed_does_not_change_hf_data_order_or_content_hash(monkeypatch):
    _install_fake_hf(monkeypatch)
    _, first = build_dataloader_factory(_cfg(order_seed=17), "hf_causal_lm", seed=1)
    _, second = build_dataloader_factory(_cfg(order_seed=17), "hf_causal_lm", seed=999)

    for key in (
        "source_order_sha256",
        "packed_token_stream_sha256",
        "selected_chunk_content_sha256",
        "sample_order_sha256",
    ):
        assert first[key] == second[key]
    assert first["order_seed"] == 17


def test_changing_order_seed_changes_source_and_packed_content_hash(monkeypatch):
    _install_fake_hf(monkeypatch)
    _, first = build_dataloader_factory(_cfg(order_seed=17), "hf_causal_lm", seed=0)
    _, second = build_dataloader_factory(_cfg(order_seed=18), "hf_causal_lm", seed=0)

    assert first["source_order_sha256"] != second["source_order_sha256"]
    assert first["packed_token_stream_sha256"] != second["packed_token_stream_sha256"]
    assert first["selected_chunk_content_sha256"] != second["selected_chunk_content_sha256"]
    assert first["resolved_tokenizer_commit"] == "a" * 40
