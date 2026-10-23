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

    def _gate_fn(self, z: torch.Tensor) -> torch.Tensor:
        """The gate g(z) with act(z) = z * g(z); also the RelP frozen derivative."""
        if _is_gelu_tanh(self.act_fn):
            return 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (z + 0.044715 * z**3)))
        assert _is_silu(self.act_fn), (
            f"No gate form for {type(_unwrap(self.act_fn)).__name__}"
        )
        return torch.sigmoid(z)

    def _act_and_deriv(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Elementwise act(z) and d act/dz.

        With relp, the derivative is frozen at the gate value g(z)
        (RelP, https://arxiv.org/abs/2508.21258). Otherwise autograd through
        act_fn gives the true derivative, for any elementwise activation.
        """
        if self.relp:
            gate = self._gate_fn(z)
            return z * gate, gate
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            act = self.act_fn(z)
            (deriv,) = torch.autograd.grad(act.sum(), z)
        return act.detach(), deriv

    def encoder_rows(self, input_acts, pos_idx, feat_idx, W_up, b_up=None):
        """Jacobian rows of act(W_gate x) * (W_up x + b_up) w.r.t. x.

        For each (position p, neuron s) pair:
            row = act_val[p,s] * W_up[s] + up_val[p,s] * act_deriv[p,s] * W_gate[s]
        With relp, act_deriv is frozen (see _act_and_deriv) and the rows are the
        RelP encoder vectors; otherwise they are the exact local Jacobian.
        With half_rule, rows are multiplied by 0.5.
        """
        gate_pre = F.linear(input_acts, self.W_gate)[pos_idx, feat_idx]
        up_val = F.linear(input_acts, W_up, b_up)[pos_idx, feat_idx]
        act_val, act_deriv = self._act_and_deriv(gate_pre)
        rows = (
            act_val.unsqueeze(-1) * W_up[feat_idx]
            + (up_val * act_deriv).unsqueeze(-1) * self.W_gate[feat_idx]
        )
        if self.half_rule:
            rows = rows * 0.5
        return rows

    def forward(self, x: torch.Tensor):
        assert self.original_inputs is not None
        gate_input = F.linear(self.original_inputs, self.W_gate)
        if self.relp:
            assert self._frozen_act is not None
            gate: torch.Tensor = self._frozen_act(gate_input)  # type: ignore
        else:
            gate = self.act_fn(gate_input)
        if self.half_rule:
            return HalfRule.apply(gate * x)
        return gate * x
