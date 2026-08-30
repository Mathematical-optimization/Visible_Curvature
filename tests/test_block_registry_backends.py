import torch

from visible_curvature.block_registry import classify_block, discover_matrix_blocks


class _GPT2Attn(torch.nn.Module):
    def __init__(self, d=6):
        super().__init__()
        # Mimic transformers.pytorch_utils.Conv1D storage: [in, out].
        self.c_attn = torch.nn.Module()
        self.c_attn.weight = torch.nn.Parameter(torch.randn(d, 3 * d))
        self.c_proj = torch.nn.Module()
        self.c_proj.weight = torch.nn.Parameter(torch.randn(d, d))


class _GPT2MLP(torch.nn.Module):
    def __init__(self, d=6, h=10):
        super().__init__()
        self.c_fc = torch.nn.Module()
        self.c_fc.weight = torch.nn.Parameter(torch.randn(d, h))
        self.c_proj = torch.nn.Module()
        self.c_proj.weight = torch.nn.Parameter(torch.randn(h, d))


class _GPT2Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _GPT2Attn()
        self.mlp = _GPT2MLP()


class _GPT2Like(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([_GPT2Block()])

