import torch
from torch import nn

from visible_curvature.block_registry import classify_block, discover_matrix_blocks


class DummyNeoXAttention(nn.Module):
    def __init__(self, hidden=8, heads=2):
        super().__init__()
        self.num_attention_heads = heads
        self.head_size = hidden // heads
        self.query_key_value = nn.Linear(hidden, 3 * hidden, bias=False)


class DummyNeoX(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = DummyNeoXAttention()


def test_bert_attention_output_is_not_mlp_down():
    assert classify_block("bert.encoder.layer.0.attention.output.dense.weight") == "attn_o"


def test_neox_qkv_split_uses_head_interleaved_rows():
    model = DummyNeoX()
    p = model.attention.query_key_value.weight
    with torch.no_grad():
        p.copy_(torch.arange(p.numel()).reshape_as(p))
    blocks = discover_matrix_blocks(
        model,
        block_types=["attn_q", "attn_k", "attn_v"],
        split_fused_qkv=True,
    )
    assert [b.block_type for b in blocks] == ["attn_q", "attn_k", "attn_v"]
    full = p.detach()
    reshaped = full.reshape(2, 3, 4, 8)
    for i, block in enumerate(blocks):
        expected = reshaped[:, i, :, :].reshape(8, 8)
        assert torch.equal(block.extract(full), expected)
        embedded = block.embed(expected)
        assert torch.equal(block.extract(embedded), expected)
        assert embedded.count_nonzero() == expected.count_nonzero()


def test_max_numel_is_applied_after_fused_split():
    model = DummyNeoX()
    # Full tensor has 192 elements; each Q/K/V block has 64.
    blocks = discover_matrix_blocks(
        model,
        block_types=["attn_q", "attn_k", "attn_v"],
        split_fused_qkv=True,
        max_numel=64,
    )
    assert len(blocks) == 3
    assert all(b.numel == 64 for b in blocks)
