from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import List, Sequence

import torch


@dataclass
class MatrixBlock:
    name: str
    param_name: str
    param: torch.nn.Parameter
    block_type: str
    layer_idx: int | None
    row_slice: slice | None = None
    col_slice: slice | None = None
    row_indices: torch.Tensor | None = None
    col_indices: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.row_slice is not None and self.row_indices is not None:
            raise ValueError("row_slice and row_indices are mutually exclusive")
        if self.col_slice is not None and self.col_indices is not None:
            raise ValueError("col_slice and col_indices are mutually exclusive")
        if self.row_indices is not None:
            self.row_indices = self.row_indices.detach().long().cpu()
        if self.col_indices is not None:
            self.col_indices = self.col_indices.detach().long().cpu()

    @property
    def shape(self) -> tuple[int, int]:
        m, n = self.param.shape
        if self.row_indices is not None:
            rows = int(self.row_indices.numel())
        else:
            rs = self.row_slice or slice(0, m)
            r0, r1, rstep = rs.indices(m)
            rows = max(0, math.ceil((r1 - r0) / rstep))
        if self.col_indices is not None:
            cols = int(self.col_indices.numel())
        else:
            cs = self.col_slice or slice(0, n)
            c0, c1, cstep = cs.indices(n)
            cols = max(0, math.ceil((c1 - c0) / cstep))
        return rows, cols

    @property
    def numel(self) -> int:
        m, n = self.shape
        return m * n

    def extract(self, full: torch.Tensor) -> torch.Tensor:
        out = full
        if self.row_indices is not None:
            out = out.index_select(0, self.row_indices.to(full.device))
        else:
            out = out[self.row_slice or slice(None), :]
        if self.col_indices is not None:
            out = out.index_select(1, self.col_indices.to(full.device))
        else:
            out = out[:, self.col_slice or slice(None)]
        return out

    def embed(self, block_tensor: torch.Tensor) -> torch.Tensor:
        if tuple(block_tensor.shape) != self.shape:
            raise ValueError(f"block tensor has shape {tuple(block_tensor.shape)}, expected {self.shape}")
        out = torch.zeros_like(self.param)
        row_idx = self.row_indices.to(out.device) if self.row_indices is not None else None
        col_idx = self.col_indices.to(out.device) if self.col_indices is not None else None
        if row_idx is None and col_idx is None:
            out[self.row_slice or slice(None), self.col_slice or slice(None)] = block_tensor
        elif row_idx is not None and col_idx is None:
            cols = self.col_slice or slice(None)
            out[row_idx, cols] = block_tensor
        elif row_idx is None and col_idx is not None:
            rows = self.row_slice or slice(None)
            out[rows, col_idx] = block_tensor
        else:
            assert row_idx is not None and col_idx is not None
            out[row_idx[:, None], col_idx[None, :]] = block_tensor
        return out


_PATTERNS: list[tuple[str, str]] = [
    (r"(?:^|\.)(q_proj|query)(?:\.|$)", "attn_q"),
    (r"(?:^|\.)(k_proj|key)(?:\.|$)", "attn_k"),
    (r"(?:^|\.)(v_proj|value)(?:\.|$)", "attn_v"),
    (r"(?:^|\.)attn\.c_attn(?:\.|$)", "attn_qkv"),
    (r"(?:^|\.)attn\.c_proj(?:\.|$)", "attn_o"),
    (r"(?:^|\.)mlp\.c_fc(?:\.|$)", "mlp_up"),
    (r"(?:^|\.)mlp\.c_proj(?:\.|$)", "mlp_down"),
    # BERT attention output must precede the generic output.dense MLP rule.
    (r"(?:^|\.)attention\.output\.dense(?:\.|$)", "attn_o"),
    (r"(?:^|\.)(o_proj|out_proj)(?:\.|$)", "attn_o"),
    (r"(?:^|\.)attn\.proj(?:\.|$)", "attn_o"),
    (r"(?:^|\.)(attention|self_attn)\.dense(?:\.|$)", "attn_o"),
    (r"(?:^|\.)(qkv|query_key_value|in_proj_weight)(?:\.|$)", "attn_qkv"),
    (r"(?:^|\.)(gate_proj)(?:\.|$)", "mlp_gate"),
    (r"(?:^|\.)(up_proj|fc1|intermediate\.dense|dense_h_to_4h)(?:\.|$)", "mlp_up"),
    (r"(?:^|\.)(down_proj|fc2|output\.dense|dense_4h_to_h)(?:\.|$)", "mlp_down"),
    (r"(?:embed|embedding|wte|patch_embed)", "embedding"),
    (r"(?:lm_head|classifier|head\.weight)", "output_head"),
]

_LAYER_PATTERNS = [
    re.compile(r"(?:layers|layer|blocks|h)\.(\d+)"),
    re.compile(r"encoder\.layer\.(\d+)"),
]


def classify_block(name: str) -> str:
    for pat, label in _PATTERNS:
        if re.search(pat, name, flags=re.IGNORECASE):
            return label
    return "matrix_other"


def infer_layer_idx(name: str) -> int | None:
    for pat in _LAYER_PATTERNS:
        match = pat.search(name)
        if match:
            return int(match.group(1))
    return None


def _parent_modules(model: torch.nn.Module, parameter_name: str) -> list[torch.nn.Module]:
    modules = dict(model.named_modules())
    path = parameter_name.rsplit(".", 1)[0] if "." in parameter_name else ""
    parts = path.split(".") if path else []
    result: list[torch.nn.Module] = []
    for end in range(len(parts), -1, -1):
        key = ".".join(parts[:end])
        module = modules.get(key)
        if module is not None:
            result.append(module)
    return result


def _neox_interleaved_indices(model: torch.nn.Module, name: str, p: torch.Tensor) -> list[torch.Tensor] | None:
    if not re.search(r"(?:^|\.)query_key_value\.weight$", name):
        return None
    hidden = int(p.shape[1])
    if int(p.shape[0]) != 3 * hidden:
        return None
    heads: int | None = None
    head_size: int | None = None
    for module in _parent_modules(model, name):
        for attr in ("num_attention_heads", "num_heads", "n_head"):
            value = getattr(module, attr, None)
            if value is not None:
                heads = int(value)
                break
        for attr in ("head_size", "head_dim"):
            value = getattr(module, attr, None)
            if value is not None:
                head_size = int(value)
                break
        if heads is not None or head_size is not None:
            if heads is None and head_size and hidden % head_size == 0:
                heads = hidden // head_size
            if head_size is None and heads and hidden % heads == 0:
                head_size = hidden // heads
            if heads and head_size and heads * head_size == hidden:
                break
    if not heads or not head_size or heads * head_size != hidden:
        raise RuntimeError(
            f"Cannot safely split GPT-NeoX/Pythia interleaved QKV parameter {name}: "
            "attention head metadata was not found"
        )
    layout = torch.arange(3 * hidden, dtype=torch.long).reshape(heads, 3, head_size)
    return [layout[:, i, :].reshape(-1) for i in range(3)]


def _keep_candidate(block: MatrixBlock, min_numel: int, max_numel: int | None) -> bool:
    return block.numel >= int(min_numel) and (max_numel is None or block.numel <= int(max_numel))


def _select_depth_stratified(blocks: list[MatrixBlock], max_blocks: int) -> list[MatrixBlock]:
    if len(blocks) <= max_blocks:
        return blocks
    by_type: dict[str, list[MatrixBlock]] = {}
    order: list[str] = []
    for block in blocks:
        if block.block_type not in by_type:
            by_type[block.block_type] = []
            order.append(block.block_type)
        by_type[block.block_type].append(block)
    for values in by_type.values():
        values.sort(key=lambda b: (b.layer_idx is None, b.layer_idx if b.layer_idx is not None else 10**9, b.name))

    selected: list[MatrixBlock] = []
    # Round-robin over block types; within each type choose quantiles so early,
    # middle, and late depth are represented before adjacent layers are added.
    cursors = {kind: 0 for kind in order}
    quantile_orders: dict[str, list[int]] = {}
    for kind, values in by_type.items():
        n = len(values)
        candidate = []
        for fraction in (0.0, 0.5, 1.0, 0.25, 0.75):
            candidate.append(int(round(fraction * (n - 1))))
        candidate.extend(range(n))
        quantile_orders[kind] = list(dict.fromkeys(candidate))
    while len(selected) < max_blocks:
        progressed = False
        for kind in order:
            idxs = quantile_orders[kind]
            cursor = cursors[kind]
            if cursor < len(idxs):
                selected.append(by_type[kind][idxs[cursor]])
                cursors[kind] += 1
                progressed = True
                if len(selected) >= max_blocks:
                    break
        if not progressed:
            break
    original_order = {id(block): i for i, block in enumerate(blocks)}
    selected.sort(key=lambda b: original_order[id(b)])
    return selected


def discover_matrix_blocks(
    model: torch.nn.Module,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    block_types: Sequence[str] | None = None,
    max_blocks: int | None = None,
    split_fused_qkv: bool = True,
    min_numel: int = 1,
    max_numel: int | None = None,
    selection_strategy: str = "first",
) -> List[MatrixBlock]:
    include = list(include or [r".*"])
    exclude = list(exclude or [])
    wanted_types = set(block_types or [])
    blocks: list[MatrixBlock] = []
    seen: set[int] = set()

    for name, p in model.named_parameters():
        if p.ndim != 2 or id(p) in seen:
            continue
        if not any(re.search(pattern, name) for pattern in include):
            continue
        if any(re.search(pattern, name) for pattern in exclude):
            continue
        btype = classify_block(name)
        split_labels_requested = bool(wanted_types.intersection({"attn_q", "attn_k", "attn_v"}))
        explicit_fused_only = bool(wanted_types) and "attn_qkv" in wanted_types and not split_labels_requested
        should_split = bool(split_fused_qkv and btype == "attn_qkv" and not explicit_fused_only)
        if wanted_types and btype not in wanted_types and not (should_split and split_labels_requested):
            continue
        layer_idx = infer_layer_idx(name)
        candidates: list[MatrixBlock] = []

        if should_split:
            neox_indices = _neox_interleaved_indices(model, name, p)
            if neox_indices is not None:
                for label, indices in zip(("attn_q", "attn_k", "attn_v"), neox_indices):
                    candidates.append(
                        MatrixBlock(
                            name=f"{name}::{label}",
                            param_name=name,
                            param=p,
                            block_type=label,
                            layer_idx=layer_idx,
                            row_indices=indices,
                        )
                    )
            else:
                gpt2_conv1d = bool(re.search(r"(?:^|\.)attn\.c_attn\.weight$", name))
                split_axis = 1 if gpt2_conv1d else 0
                if p.shape[split_axis] % 3 != 0:
                    if not wanted_types or "attn_qkv" in wanted_types:
                        candidates.append(MatrixBlock(name, name, p, btype, layer_idx))
                else:
                    width = p.shape[split_axis] // 3
                    for i, label in enumerate(("attn_q", "attn_k", "attn_v")):
                        kwargs = {"row_slice": slice(i * width, (i + 1) * width)} if split_axis == 0 else {"col_slice": slice(i * width, (i + 1) * width)}
                        candidates.append(
                            MatrixBlock(
                                name=f"{name}::{label}",
                                param_name=name,
                                param=p,
                                block_type=label,
                                layer_idx=layer_idx,
                                **kwargs,
                            )
                        )
        else:
            candidates.append(MatrixBlock(name, name, p, btype, layer_idx))

        for candidate in candidates:
            if wanted_types and candidate.block_type not in wanted_types:
                continue
            if _keep_candidate(candidate, min_numel, max_numel):
                blocks.append(candidate)
        seen.add(id(p))

    if max_blocks is None:
        return blocks
    max_blocks = int(max_blocks)
    if selection_strategy == "first":
        return blocks[:max_blocks]
    if selection_strategy == "depth_stratified":
        return _select_depth_stratified(blocks, max_blocks)
    raise ValueError(f"Unknown selection_strategy: {selection_strategy}")


def block_metadata(block: MatrixBlock) -> dict:
    return {
        "block_name": block.name,
        "param_name": block.param_name,
        "block_type": block.block_type,
        "layer_idx": block.layer_idx,
        "rows": block.shape[0],
        "cols": block.shape[1],
        "numel": block.numel,
        "indexed_rows": block.row_indices is not None,
        "indexed_cols": block.col_indices is not None,
    }
