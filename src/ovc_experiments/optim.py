from __future__ import annotations

from collections.abc import Iterable

import torch

from .preconditioners import symmetric_matrix_power


class MatrixShampoo(torch.optim.Optimizer):
    """Small, inspectable Shampoo implementation for controlled experiments.

    Matrix parameters use two-sided EMA factors. Vector parameters use an
    AdamW-like fallback. This optimizer is intentionally transparent rather
    than systems-optimized; it is suitable for the small models shipped with
    the experiment package and for short external-validity continuations.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        alpha: float = 0.25,
        root_frequency: int = 10,
        weight_decay: float = 0.0,
        grafting: str = "none",
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must lie in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 <= alpha <= 0.5:
            raise ValueError("alpha must lie in [0, 1/2]")
        if root_frequency < 1:
            raise ValueError("root_frequency must be positive")
        if grafting not in {"none", "sgd", "adam"}:
            raise ValueError("grafting must be one of: none, sgd, adam")
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            alpha=alpha,
            root_frequency=root_frequency,
            weight_decay=weight_decay,
            grafting=grafting,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                if gradient.is_sparse:
                    raise RuntimeError("MatrixShampoo does not support sparse gradients")
                state = self.state[parameter]
                state["step"] = int(state.get("step", 0)) + 1
                step = state["step"]
                if group["weight_decay"] != 0:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                if parameter.ndim >= 2:
                    direction = self._matrix_direction(parameter, gradient, state, group, step)
                else:
                    direction = self._vector_direction(parameter, gradient, state, group, step)
                parameter.add_(direction, alpha=-group["lr"])
        return loss

    def _matrix_direction(self, parameter, gradient, state, group, step):
        rows = parameter.shape[0]
        matrix_gradient = gradient.reshape(rows, -1)
        cols = matrix_gradient.shape[1]
        beta1 = group["beta1"]
        beta2 = group["beta2"]
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(parameter)
            state["left_factor"] = torch.zeros(
                rows, rows, dtype=parameter.dtype, device=parameter.device
            )
            state["right_factor"] = torch.zeros(
                cols, cols, dtype=parameter.dtype, device=parameter.device
            )
            state["left_root"] = torch.eye(rows, dtype=parameter.dtype, device=parameter.device)
            state["right_root"] = torch.eye(cols, dtype=parameter.dtype, device=parameter.device)
            if group["grafting"] == "adam":
                state["diag_second"] = torch.zeros_like(parameter)

        exp_avg = state["exp_avg"]
        exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        left_contribution = matrix_gradient @ matrix_gradient.T / max(cols, 1)
        right_contribution = matrix_gradient.T @ matrix_gradient / max(rows, 1)
        state["left_factor"].mul_(beta2).add_(left_contribution, alpha=1.0 - beta2)
        state["right_factor"].mul_(beta2).add_(right_contribution, alpha=1.0 - beta2)

        if step == 1 or step % group["root_frequency"] == 0:
            correction = max(1.0 - beta2**step, torch.finfo(parameter.dtype).eps)
            left_hat = state["left_factor"] / correction
            right_hat = state["right_factor"] / correction
            left_damped = left_hat + group["epsilon"] * torch.eye(
                rows, dtype=parameter.dtype, device=parameter.device
            )
            right_damped = right_hat + group["epsilon"] * torch.eye(
                cols, dtype=parameter.dtype, device=parameter.device
            )
            state["left_root"] = symmetric_matrix_power(left_damped, -group["alpha"])
            state["right_root"] = symmetric_matrix_power(right_damped, -group["alpha"])

        first_correction = max(1.0 - beta1**step, torch.finfo(parameter.dtype).eps)
        momentum = (exp_avg / first_correction).reshape(rows, cols)
        shampoo = state["left_root"] @ momentum @ state["right_root"]

        grafting = group["grafting"]
        if grafting == "none":
            return shampoo.reshape_as(parameter)
        if grafting == "sgd":
            target = momentum
        else:
            diagonal = state["diag_second"]
            diagonal.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            second_correction = max(1.0 - beta2**step, torch.finfo(parameter.dtype).eps)
            target = (exp_avg / first_correction) / (
                (diagonal / second_correction).sqrt() + group["epsilon"]
            )
            target = target.reshape(rows, cols)
        shampoo_norm = torch.linalg.vector_norm(shampoo)
        target_norm = torch.linalg.vector_norm(target)
        if shampoo_norm > 0 and target_norm > 0:
            shampoo = shampoo * (target_norm / shampoo_norm)
        return shampoo.reshape_as(parameter)

    def _vector_direction(self, parameter, gradient, state, group, step):
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(group["beta1"]).add_(gradient, alpha=1.0 - group["beta1"])
        exp_avg_sq.mul_(group["beta2"]).addcmul_(
            gradient, gradient, value=1.0 - group["beta2"]
        )
        first_correction = max(1.0 - group["beta1"] ** step, torch.finfo(parameter.dtype).eps)
        second_correction = max(1.0 - group["beta2"] ** step, torch.finfo(parameter.dtype).eps)
        return (exp_avg / first_correction) / (
            (exp_avg_sq / second_correction).sqrt() + group["epsilon"]
        )
