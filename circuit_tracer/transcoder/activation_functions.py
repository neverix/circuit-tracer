from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class jumprelu(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        grad_output = grad_outputs[0]
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


class JumpReLU(torch.nn.Module):
    def __init__(self, threshold: float | torch.Tensor, bandwidth: float = 2) -> None:
        super().__init__()
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold)

        self.threshold = nn.Parameter(threshold)
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return jumprelu.apply(x, self.threshold, self.bandwidth)  # type: ignore

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, bandwidth={self.bandwidth}"


class TopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor):
        _, indices = torch.topk(x, k=self.k, dim=-1)
        gate = torch.zeros_like(x)
        gate.scatter_(dim=-1, index=indices, value=1)
        return x * gate.to(x.dtype)

class FrozenSiLU(torch.autograd.Function):
    """SiLU with frozen sigmoid in backward: grad * sigmoid(x)."""

    @staticmethod
    def forward(ctx, x):
        sig = torch.sigmoid(x)
        ctx.save_for_backward(sig)
        return F.silu(x)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> torch.Tensor:
        (sig,) = ctx.saved_tensors
        return sig * grad_outputs[0]


class FrozenGELUTanh(torch.autograd.Function):
    """GELU(tanh approx) with frozen gate in backward: grad * gate."""

    @staticmethod
    def forward(ctx, x):
        gate = 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))
        ctx.save_for_backward(gate)
        return x * gate

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> torch.Tensor:
        (gate,) = ctx.saved_tensors
        return gate * grad_outputs[0]


class HalfRule(torch.autograd.Function):
    """Forward: identity. Backward: grad / 2. Corrects bilinear double-counting."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> torch.Tensor:
        return grad_outputs[0] / 2


def _unwrap(act_fn):
    """Unwrap nnsight Envoy wrapper if present."""
    return act_fn._module if hasattr(act_fn, '_module') else act_fn


def _is_gelu_tanh(act_fn) -> bool:
    """Detect GELU with tanh approximation (Gemma 3, etc.)."""
    act_fn = _unwrap(act_fn)
    if hasattr(act_fn, 'approximate') and act_fn.approximate == 'tanh':
        return True
    if isinstance(act_fn, nn.GELU) and getattr(act_fn, 'approximate', 'none') == 'tanh':
        return True
    # HuggingFace PytorchGELUTanh
    if 'GELUTanh' in type(act_fn).__name__:
        return True
    return False


def _is_silu(act_fn) -> bool:
    """Detect SiLU / Swish (Llama, Gemma 2, etc.)."""
    act_fn = _unwrap(act_fn)
    if isinstance(act_fn, nn.SiLU):
        return True
    if 'SiLU' in type(act_fn).__name__:
        return True
    return False


class GatedActivationFunction(nn.Module):
    """Gated MLP activation: act_fn(gate_proj @ x) * (up_proj @ x).

    With relp=True, uses frozen activation derivatives in backward
    (RelP, https://arxiv.org/abs/2508.21258). With half_rule=True,
    applies HalfRule to correct bilinear double-counting.
    """

    def __init__(self, W_gate, act_fn, relp: bool = False, half_rule: bool | None = None):
        super().__init__()
        self.W_gate: torch.Tensor = W_gate
        self.act_fn = act_fn
        self.relp = relp
        self.half_rule = relp if half_rule is None else half_rule
        self.original_inputs: torch.Tensor | None = None

        # Determine frozen activation variant
        if _is_gelu_tanh(act_fn):
            self._frozen_act = FrozenGELUTanh.apply
        elif _is_silu(act_fn):
            self._frozen_act = FrozenSiLU.apply
        elif relp:
            raise ValueError(
                f"RelP requires GELU(tanh) or SiLU activation, got {type(act_fn).__name__}"
            )
        else:
            self._frozen_act = None

    def set_original_inputs(self, inputs: torch.Tensor):
        self.original_inputs = inputs

    def forward(self, x: torch.Tensor):
        assert self.original_inputs is not None
        gate_input = F.linear(self.original_inputs, self.W_gate)
        if self.relp:
            assert self._frozen_act is not None
            gate = self._frozen_act(gate_input)
        else:
            gate = self.act_fn(gate_input)
        if self.half_rule:
            return HalfRule.apply(gate * x)
        return gate * x
