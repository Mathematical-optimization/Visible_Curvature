from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from .block_registry import MatrixBlock
from .linear_algebra import multi_start_lanczos
from .utils import eval_mode, move_batch_to_device, only_parameter_requires_grad

Tensor = torch.Tensor


class LinearMatrixOperator:
    def __init__(self, shape: tuple[int, int], matvec_matrix: Callable[[Tensor], Tensor], device: torch.device, dtype: torch.dtype):
        self.shape = shape
        self.dim = shape[0] * shape[1]
        self._matvec_matrix = matvec_matrix
        self.device = device
        self.dtype = dtype

    def matvec_matrix(self, V: Tensor) -> Tensor:
        return self._matvec_matrix(V)

    def matvec(self, v: Tensor) -> Tensor:
        V = v.reshape(self.shape)
        return self._matvec_matrix(V).reshape(-1)

    def shifted(self, shift: float) -> "LinearMatrixOperator":
        if shift == 0.0:
            return self
        return LinearMatrixOperator(
            self.shape,
            lambda V: self._matvec_matrix(V) + float(shift) * V,
            self.device,
            self.dtype,
        )


class BlockHessianOperator(LinearMatrixOperator):
    def __init__(
        self,
        model: torch.nn.Module,
        block: MatrixBlock,
        batch: Any,
        loss_fn: Callable[[torch.nn.Module, Any], Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.block = block
        self.batch = move_batch_to_device(batch, device)
        self.loss_fn = loss_fn
        self.param = block.param

        def hvp(V: Tensor) -> Tensor:
            V = V.to(device=self.param.device, dtype=self.param.dtype)
            v_full = block.embed(V)
            with eval_mode(model), only_parameter_requires_grad(model, self.param):
                loss = loss_fn(model, self.batch)
                g = torch.autograd.grad(loss, self.param, create_graph=True, retain_graph=True)[0]
                dot = torch.sum(g * v_full)
                hv_full = torch.autograd.grad(dot, self.param, retain_graph=False, create_graph=False)[0]
            return block.extract(hv_full).detach().to(device=device, dtype=dtype)

        super().__init__(block.shape, hvp, device=device, dtype=dtype)


class BlockGGNOperator(LinearMatrixOperator):
    """Block empirical generalized Gauss--Newton operator for mean CE loss.

    ``ggn_spec_fn(output, batch)`` must return ``(logits, targets)`` with
    logits shaped ``[num_valid_examples, num_classes]``.  Targets are retained
    for protocol metadata; the cross-entropy logit Hessian depends only on the
    probabilities after valid-example selection.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        block: MatrixBlock,
        batch: Any,
        ggn_spec_fn: Callable[[Any, Any], tuple[Tensor, Tensor]],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.block = block
        self.batch = move_batch_to_device(batch, device)
        self.ggn_spec_fn = ggn_spec_fn
        self.param = block.param
        base_params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        def call_model(params: Mapping[str, Tensor]):
            if isinstance(self.batch, Mapping):
                return torch.func.functional_call(
                    model, (params, buffers), (), dict(self.batch), tie_weights=False, strict=False
                )
            if isinstance(self.batch, (tuple, list)):
                # Classifier datasets conventionally return (inputs, labels).
                return torch.func.functional_call(
                    model, (params, buffers), (self.batch[0],), {}, tie_weights=False, strict=False
                )
            return torch.func.functional_call(
                model, (params, buffers), (self.batch,), {}, tie_weights=False, strict=False
            )

        def logits_from_parameter(parameter_value: Tensor) -> Tensor:
            params = dict(base_params)
            params[block.param_name] = parameter_value
            output = call_model(params)
            logits, _ = ggn_spec_fn(output, self.batch)
            if logits.ndim != 2:
                raise ValueError(f"GGN logits must be rank two [N,C], got {tuple(logits.shape)}")
            if logits.shape[0] == 0:
                raise ValueError("GGN received no valid examples/tokens")
            return logits

        def ggn_vp(V: Tensor) -> Tensor:
            V = V.to(device=self.param.device, dtype=self.param.dtype)
            tangent_full = block.embed(V)
            with eval_mode(model):
                logits, delta_logits = torch.func.jvp(
                    logits_from_parameter, (self.param,), (tangent_full,)
                )
                probabilities = torch.softmax(logits, dim=-1)
                mean_delta = torch.sum(probabilities * delta_logits, dim=-1, keepdim=True)
                cotangent = probabilities * (delta_logits - mean_delta) / logits.shape[0]
                _, vjp_fn = torch.func.vjp(logits_from_parameter, self.param)
                hv_full = vjp_fn(cotangent)[0]
            return block.extract(hv_full).detach().to(device=device, dtype=dtype)

        super().__init__(block.shape, ggn_vp, device=device, dtype=dtype)



@dataclass
class CurvatureBuildResult:
    operator: LinearMatrixOperator
    raw_min_ritz: float
    raw_max_ritz: float
    shift: float
    psd_mode: str
    target_ridge: float
    ridge_mode: str
    shift_source: str = "estimated"
    raw_min_residual: float = float("nan")
    raw_max_residual: float = float("nan")


def stabilize_curvature(
    raw: LinearMatrixOperator,
    psd_mode: str,
    ridge: float,
    lanczos_steps: int,
    lanczos_starts: int,
    seed: int,
    max_rounds: int = 3,
    ridge_mode: str = "absolute",
    shift_override: float | None = None,
) -> CurvatureBuildResult:
    """Build the declared curvature operator and empirically verify its PSD shift.

    A smallest Ritz value is an *upper* approximation to the true minimum
    eigenvalue, so a single shift based on one Lanczos pass does not certify
    positive definiteness.  For ``psd_mode=shift`` we therefore re-run Lanczos
    after each correction and add another shift if a negative Ritz value is
    still observed.  This remains a numerical stabilization, not a formal PSD
    certificate; the raw and applied shifts are logged for audit.
    """
    spec = multi_start_lanczos(
        raw.matvec, raw.dim, steps=lanczos_steps, device=raw.device, dtype=raw.dtype,
        starts=lanczos_starts, seed=seed,
    )
    min_ritz = float(spec["min_ritz"])
    max_ritz = float(spec["max_ritz"])
    mode = str(psd_mode)
    shift = 0.0
    ridge_mode = str(ridge_mode)
    if ridge_mode == "absolute":
        target = max(float(ridge), 0.0)
    elif ridge_mode == "relative_max":
        target = max(float(ridge), 0.0) * max(abs(max_ritz), torch.finfo(raw.dtype).tiny)
    else:
        raise ValueError(f"Unknown ridge_mode: {ridge_mode}")
    raw_min_residual = float(spec.get("min_ritz_residual", float("nan")))
    raw_max_residual = float(spec.get("max_ritz_residual", float("nan")))

    if shift_override is not None:
        shift = float(shift_override)
        if not math.isfinite(shift) or shift < 0.0:
            raise ValueError("shift_override must be a finite nonnegative value")
        return CurvatureBuildResult(
            raw.shifted(shift),
            min_ritz,
            max_ritz,
            shift,
            mode,
            target,
            ridge_mode,
            "override",
            raw_min_residual,
            raw_max_residual,
        )

    if mode == "shift":
        shift = max(0.0, -min_ritz + target)
        op = raw.shifted(shift)
        for rr in range(max(1, int(max_rounds))):
            chk = multi_start_lanczos(
                op.matvec, op.dim, steps=lanczos_steps, device=op.device, dtype=op.dtype,
                starts=lanczos_starts, seed=seed + 15485863 * (rr + 1),
            )
            chk_min = float(chk["min_ritz"])
            if chk_min > 0.5 * max(target, 1e-12):
                break
            correction = max(0.0, -chk_min + target)
            if correction <= 0:
                break
            shift += correction
            op = raw.shifted(shift)
        return CurvatureBuildResult(
            op,
            min_ritz,
            max_ritz,
            shift,
            mode,
            target,
            ridge_mode,
            "estimated",
            raw_min_residual,
            raw_max_residual,
        )
    if mode == "require":
        if min_ritz <= 0:
            raise RuntimeError(f"Curvature is not positive definite (min Ritz={min_ritz:.3e}); choose psd_mode=shift or fisher")
        shift = target
    elif mode == "none":
        shift = target
    else:
        raise ValueError(f"Unknown psd_mode: {mode}")
    return CurvatureBuildResult(
        raw.shifted(shift),
        min_ritz,
        max_ritz,
        shift,
        mode,
        target,
        ridge_mode,
        "estimated",
        raw_min_residual,
        raw_max_residual,
    )


def estimate_partial_traces_and_diagonal(
    op: LinearMatrixOperator,
    num_probes: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Estimate both Kronecker partial traces and ``diag(H)``.

    The same Rademacher HVPs provide all three quantities:
    ``E[(HV)V^T]``, ``E[V^T(HV)]``, and ``E[V * HV]``.
    Reusing probes both reduces cost and makes Adam/Shampoo visibility
    diagnostics refer to the same randomized curvature sample.
    """
    if int(num_probes) <= 0:
        raise ValueError("num_probes must be positive")
    m, n = op.shape
    gen = torch.Generator(device=op.device).manual_seed(int(seed))
    L = torch.zeros(m, m, device=op.device, dtype=op.dtype)
    R = torch.zeros(n, n, device=op.device, dtype=op.dtype)
    D = torch.zeros(m, n, device=op.device, dtype=op.dtype)
    for _ in range(int(num_probes)):
        V = torch.empty(m, n, device=op.device, dtype=op.dtype)
        V.bernoulli_(0.5, generator=gen).mul_(2).sub_(1)
        HV = op.matvec_matrix(V)
        L.add_(HV @ V.T)
        R.add_(V.T @ HV)
        D.add_(V * HV)
    L.div_(num_probes)
    R.div_(num_probes)
    D.div_(num_probes)
    L = 0.5 * (L + L.T)
    R = 0.5 * (R + R.T)
    return L, R, D


def estimate_partial_traces(
    op: LinearMatrixOperator,
    num_probes: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    L, R, _ = estimate_partial_traces_and_diagonal(op, num_probes, seed)
    return L, R


def effective_operator(op: LinearMatrixOperator, apply_half: Callable[[Tensor], Tensor]) -> LinearMatrixOperator:
    def mv(V: Tensor) -> Tensor:
        X = apply_half(V)
        HX = op.matvec_matrix(X)
        return apply_half(HX)
    return LinearMatrixOperator(op.shape, mv, device=op.device, dtype=op.dtype)


def estimate_partial_trace_convergence(
    op: LinearMatrixOperator,
    probe_budgets: list[int] | tuple[int, ...],
    seed: int,
) -> dict[int, tuple[Tensor, Tensor, Tensor]]:
    """Return cumulative partial-trace/diagonal estimates at several budgets."""
    budgets = sorted({int(b) for b in probe_budgets})
    if not budgets or budgets[0] <= 0:
        raise ValueError("probe budgets must be positive")
    m, n = op.shape
    gen = torch.Generator(device=op.device).manual_seed(int(seed))
    L = torch.zeros(m, m, device=op.device, dtype=op.dtype)
    R = torch.zeros(n, n, device=op.device, dtype=op.dtype)
    D = torch.zeros(m, n, device=op.device, dtype=op.dtype)
    out: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
    targets = set(budgets)
    for k in range(1, budgets[-1] + 1):
        V = torch.empty(m, n, device=op.device, dtype=op.dtype)
        V.bernoulli_(0.5, generator=gen).mul_(2).sub_(1)
        HV = op.matvec_matrix(V)
        L.add_(HV @ V.T)
        R.add_(V.T @ HV)
        D.add_(V * HV)
        if k in targets:
            Lk = 0.5 * ((L / k) + (L / k).T)
            Rk = 0.5 * ((R / k) + (R / k).T)
            out[k] = (Lk.clone(), Rk.clone(), (D / k).clone())
    return out
