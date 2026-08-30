from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_dtype

SUPPORTED_MODEL_BACKENDS = ("tiny_causal_lm", "hf_causal_lm")


class TinySelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = x.shape

        def split(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)

        q = split(self.q_proj(x))
        k = split(self.k_proj(x))
        v = split(self.v_proj(x))
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        causal_mask = torch.triu(
            torch.ones(sequence, sequence, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        output = attention @ v
        output = output.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.o_proj(output)


class TinyMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, hidden: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinySelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = TinyMLP(d_model, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 32,
        n_heads: int = 4,
        hidden: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TinyBlock(d_model, n_heads, hidden) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        del attention_mask
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)
            )
        return {"loss": loss, "logits": logits}


@dataclass
class ModelBundle:
    model: nn.Module
    loss_fn: Callable[[nn.Module, Any], torch.Tensor]
    metadata: Dict[str, Any]
    ggn_spec_fn: Callable[[Any, Any], tuple[torch.Tensor, torch.Tensor]] | None = None


def _hf_causal_loss(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    output = model(**batch)
    loss = output.get("loss") if isinstance(output, Mapping) else getattr(output, "loss", None)
    if loss is None:
        raise RuntimeError("Causal-LM output did not contain loss")
    return loss


def _tiny_loss(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    output = model(**batch)
    loss = output.get("loss") if isinstance(output, Mapping) else getattr(output, "loss", None)
    if loss is None:
        raise RuntimeError("Tiny causal-LM output did not contain loss")
    return loss


def _output_logits(output: Any) -> torch.Tensor:
    logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
    if logits is None:
        raise RuntimeError("Model output did not expose logits required for GGN")
    return logits


def causal_lm_ggn_spec(
    output: Any, batch: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _output_logits(output)[:, :-1, :]
    labels = batch["labels"][:, 1:]
    valid = labels != -100
    return logits[valid], labels[valid]


def load_model_bundle(cfg: Mapping[str, Any], device: torch.device) -> ModelBundle:
    model_cfg = cfg.get("model", {})
    backend = str(model_cfg.get("backend", "tiny_causal_lm"))
    if backend not in SUPPORTED_MODEL_BACKENDS:
        supported = ", ".join(SUPPORTED_MODEL_BACKENDS)
        raise ValueError(f"Supported model backends are: {supported}; got {backend}")

    dtype = get_dtype(str(model_cfg.get("dtype", "float32")))
    if backend == "tiny_causal_lm":
        model = TinyCausalLM(
            vocab_size=int(model_cfg.get("vocab_size", 128)),
            d_model=int(model_cfg.get("d_model", 32)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            hidden=int(model_cfg.get("hidden", 64)),
            n_layers=int(model_cfg.get("n_layers", 2)),
        )
        model.to(device=device, dtype=dtype)
        return ModelBundle(
            model=model,
            loss_fn=_tiny_loss,
            metadata={"backend": backend, "model_name": "tiny_causal_lm"},
            ggn_spec_fn=causal_lm_ggn_spec,
        )

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise ImportError("Install transformers for the hf_causal_lm backend") from error

    name = str(model_cfg["name"])
    kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    attention_impl = model_cfg.get("attn_implementation", "eager")
    if attention_impl is not None:
        kwargs["attn_implementation"] = str(attention_impl)
    revision = model_cfg.get("revision")
    if revision is not None:
        kwargs["revision"] = str(revision)
    if bool(model_cfg.get("trust_remote_code", False)):
        kwargs["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.to(device)
    return ModelBundle(
        model=model,
        loss_fn=_hf_causal_loss,
        metadata={
            "backend": backend,
            "model_name": name,
            "model_revision": revision,
            "resolved_model_commit": getattr(getattr(model, "config", None), "_commit_hash", None),
        },
        ggn_spec_fn=causal_lm_ggn_spec,
    )
