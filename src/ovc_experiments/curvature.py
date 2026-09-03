from __future__ import annotations

import torch

from .blocks import MatrixLayout
from .functional import FunctionalBlockModel
from .operators import ShiftedOperator, SymmetricLinearOperator
from .tasks import Batch


class EmpiricalFisherOperator(SymmetricLinearOperator):
    def __init__(self, per_example_gradients: torch.Tensor, layout: MatrixLayout) -> None:
        if tuple(per_example_gradients.shape[1:]) not in {
            layout.original_shape,
            layout.matrix_shape,
        }:
            raise ValueError(
                f"Gradient tail {tuple(per_example_gradients.shape[1:])} does not match block layout"
            )
        self.gradients = per_example_gradients.reshape(per_example_gradients.shape[0], -1)
        self.dimension = layout.numel
        self.dtype = self.gradients.dtype
        self.device = self.gradients.device
        self.name = "empirical-fisher"

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector)
        projections = self.gradients @ vector
        return self.gradients.T @ projections / self.gradients.shape[0]


class CrossEntropyGGNOperator(SymmetricLinearOperator):
    def __init__(self, functional: FunctionalBlockModel, batch: Batch) -> None:
        self.functional = functional
        self.batch = batch
        self.dimension = functional.block.numel
        self.dtype = functional.block_parameter.dtype
        self.device = functional.block_parameter.device
        self.name = "cross-entropy-ggn"
        with torch.no_grad():
            self.fixed_logits = functional.logits(functional.block_parameter, batch).detach()

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector)
        direction = vector.reshape(self.functional.block.shape)
        logit_function = lambda parameter: self.functional.logits(parameter, self.batch)
        _, jvp = torch.func.jvp(
            logit_function,
            (self.functional.block_parameter,),
            (direction,),
        )
        output_curvature_jvp = self.functional.task.output_hessian_action(
            self.fixed_logits, jvp, self.batch
        )
        _, vjp_function = torch.func.vjp(logit_function, self.functional.block_parameter)
        result = vjp_function(output_curvature_jvp)[0]
        return result.reshape(-1).detach()


class ExactHessianOperator(SymmetricLinearOperator):
    def __init__(self, functional: FunctionalBlockModel, batch: Batch) -> None:
        self.functional = functional
        self.batch = batch
        self.dimension = functional.block.numel
        self.dtype = functional.block_parameter.dtype
        self.device = functional.block_parameter.device
        self.name = "exact-hessian"
        self._gradient_function = torch.func.grad(
            lambda parameter: self.functional.mean_loss(parameter, self.batch)
        )

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector)
        direction = vector.reshape(self.functional.block.shape)
        _, product = torch.func.jvp(
            self._gradient_function,
            (self.functional.block_parameter,),
            (direction,),
        )
        return product.reshape(-1).detach()


class RegularizedOperator(ShiftedOperator):
    def __init__(self, base: SymmetricLinearOperator, shift: float) -> None:
        super().__init__(base, shift, name=f"regularized({base.name})")
