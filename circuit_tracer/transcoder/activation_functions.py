from typing import Any, Literal

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


def gelu_tanh_gate(x: torch.Tensor) -> torch.Tensor:
    """g(x) with GELU_tanh(x) = x * g(x)."""
    return 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


class FrozenSiLU(torch.autograd.Function):
    """SiLU whose backward treats the sigmoid gate as a constant: grad * sigmoid(x)."""

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
    """GELU (tanh approximation) whose backward treats the gate as a constant."""

    @staticmethod
    def forward(ctx, x):
        gate = gelu_tanh_gate(x)
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
        # a copy, so that callers may modify the output in place
        return x.clone()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> torch.Tensor:
        return grad_outputs[0] / 2


def _activation_kind(act_fn) -> Literal["gelu_tanh", "silu"] | None:
    """Recognize the activations that RelP has a frozen form for.

    HuggingFace passes modules (nn.SiLU, GELUTanh, ...); TransformerLens passes
    functions (silu, gelu_pytorch_tanh, gelu_new).
    """
    name = getattr(act_fn, "__name__", type(act_fn).__name__)
    if getattr(act_fn, "approximate", None) == "tanh" or "GELUTanh" in name:
        return "gelu_tanh"
    if name in ("gelu_pytorch_tanh", "gelu_new"):
        return "gelu_tanh"
    if isinstance(act_fn, nn.SiLU) or name in ("silu", "SiLUActivation"):
        return "silu"
    return None


class GatedActivationFunction(nn.Module):
    """Gated MLP activation: act_fn(W_gate x) * pre_acts, with pre_acts = W_up x + b_up.

    The gate reads the MLP input x, not pre_acts, so set_original_inputs(x) must be
    called before forward(pre_acts).

    With relp=True the activation derivative is frozen at the gate value
    (RelP, https://arxiv.org/abs/2508.21258). half_rule halves the backward of
    the product (and the encoder rows); None follows relp.
    """

    def __init__(self, W_gate, act_fn, relp: bool = False, half_rule: bool | None = None):
        super().__init__()
        self.W_gate: torch.Tensor = W_gate
        self.act_fn = act_fn
        self.relp = relp
        self.half_rule = relp if half_rule is None else half_rule
        self.original_inputs: torch.Tensor | None = None
        self.kind = _activation_kind(act_fn)
        if relp and self.kind is None:
            raise ValueError(
                f"RelP requires a GELU(tanh) or SiLU activation, got {type(act_fn).__name__}"
            )

    def set_original_inputs(self, inputs: torch.Tensor):
        self.original_inputs = inputs

    def _frozen_act(self, z: torch.Tensor) -> torch.Tensor:
        frozen = FrozenGELUTanh if self.kind == "gelu_tanh" else FrozenSiLU
        return frozen.apply(z)  # type: ignore

    def _act_and_deriv(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Elementwise act(z) and d act/dz.

        With relp the derivative is frozen at the gate value g(z), where
        act(z) = z * g(z). Otherwise autograd through act_fn gives the true
        derivative of any elementwise activation.
        """
        if self.relp:
            gate = gelu_tanh_gate(z) if self.kind == "gelu_tanh" else torch.sigmoid(z)
            return z * gate, gate
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            act = self.act_fn(z)
            (deriv,) = torch.autograd.grad(act.sum(), z)
        return act.detach(), deriv

    def encoder_rows(self, input_acts, pos_idx, feat_idx, W_up, b_up=None):
        """Jacobian rows of act(W_gate x) * (W_up x + b_up) with respect to x.

        For each (position p, neuron s) pair:
            row = act(gate_pre) * W_up[s] + up_val * act'(gate_pre) * W_gate[s]
        act' is frozen with relp (the RelP encoder vectors) and exact otherwise.
        half_rule halves the rows.
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
        assert self.original_inputs is not None, "call set_original_inputs first"
        gate_pre = F.linear(self.original_inputs, self.W_gate)
        gate = self._frozen_act(gate_pre) if self.relp else self.act_fn(gate_pre)
        if self.half_rule:
            return HalfRule.apply(gate * x)
        return gate * x
