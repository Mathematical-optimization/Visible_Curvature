import torch
from torch import nn
import torch.nn.functional as F

from visible_curvature.block_registry import MatrixBlock
from visible_curvature.curvature import BlockGGNOperator, BlockHessianOperator


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


def loss_fn(model, batch):
    x, y = batch
    return F.cross_entropy(model(x), y)


def ggn_spec(output, batch):
    _, y = batch
    return output, y


def test_cross_entropy_ggn_matches_hessian_for_linear_logits():
    torch.manual_seed(3)
    model = LinearClassifier().double()
    block = MatrixBlock("proj.weight", "proj.weight", model.proj.weight, "matrix_other", None)
    batch = (torch.randn(5, 3, dtype=torch.float64), torch.tensor([0, 1, 2, 3, 1]))
    hessian = BlockHessianOperator(model, block, batch, loss_fn, torch.device("cpu"), torch.float64)
    ggn = BlockGGNOperator(model, block, batch, ggn_spec, torch.device("cpu"), torch.float64)
    v = torch.randn_like(model.proj.weight)
    assert torch.allclose(ggn.matvec_matrix(v), hessian.matvec_matrix(v), atol=1e-9, rtol=1e-8)
