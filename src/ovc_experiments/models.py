from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        causal: bool,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self._split_heads(self.q_proj(hidden))
        key = self._split_heads(self.k_proj(hidden))
        value = self._split_heads(self.v_proj(hidden))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        length = hidden.shape[1]
        if self.causal:
            causal_mask = torch.ones(
                length, length, dtype=torch.bool, device=hidden.device
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            if attention_mask.shape != hidden.shape[:2]:
                raise ValueError(
                    f"attention_mask must have shape {tuple(hidden.shape[:2])}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            key_mask = ~attention_mask.to(dtype=torch.bool)[:, None, None, :]
            scores = scores.masked_fill(key_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = self.attention_dropout(probabilities)
        context = probabilities @ value
        context = context.transpose(1, 2).reshape(hidden.shape)
        return self.output_dropout(self.o_proj(context))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        attention = MultiHeadSelfAttention(
            d_model, n_heads, dropout=dropout, causal=True
        )
        # Register projections directly on the block so names match the paper's
        # W_Q, W_K, W_V, W_O block convention.
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self._attention = attention
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.up_proj = nn.Linear(d_model, hidden)
        self.down_proj = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = hidden + self._attention(self.norm1(hidden), attention_mask)
        mlp = self.down_proj(F.gelu(self.up_proj(self.norm2(hidden))))
        return hidden + self.dropout(mlp)


class TinyDecoderLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 32,
        max_seq_len: int = 16,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, n_heads, mlp_ratio, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        batch, length = input_ids.shape
        if length > self.max_seq_len:
            raise ValueError(f"Sequence length {length} exceeds max_seq_len={self.max_seq_len}")
        if position_ids is None:
            position_ids = torch.arange(length, device=input_ids.device).expand(batch, length)
        hidden = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return self.lm_head(self.final_norm(hidden))


class VisionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        attention = MultiHeadSelfAttention(
            d_model, n_heads, dropout=dropout, causal=False
        )
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self._attention = attention
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.up_proj = nn.Linear(d_model, hidden)
        self.down_proj = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self._attention(self.norm1(hidden))
        hidden = hidden + self.dropout(
            self.down_proj(F.gelu(self.up_proj(self.norm2(hidden))))
        )
        return hidden


class TinyVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 16,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        num_patches = (image_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.blocks = nn.ModuleList(
            [VisionBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.apply(self._initialize)
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(images.shape[0], -1, -1)
        hidden = torch.cat([class_token, patches], dim=1)
        hidden = hidden + self.position_embedding[:, : hidden.shape[1]]
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.final_norm(hidden[:, 0]))
