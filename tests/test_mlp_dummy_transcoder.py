"""MLPDummyTranscoder without a model: thresholding and encoder vectors."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from circuit_tracer.transcoder import MLPDummyTranscoder
from circuit_tracer.transcoder.activation_functions import GatedActivationFunction

D_MODEL, D_MLP, N_POS = 8, 16, 6

ACTIVATIONS = {"gelu_tanh": lambda: nn.GELU(approximate="tanh"), "silu": nn.SiLU}


def make_transcoder(act, relp, half_rule=None, jacobian_encoders=False, bias=False, **kwargs):
    torch.manual_seed(123)
    W_up, W_gate, W_down = (torch.randn(D_MLP, D_MODEL) for _ in range(3))
    b_up = torch.randn(D_MLP) if bias else None
    act_fn = GatedActivationFunction(W_gate, ACTIVATIONS[act](), relp=relp, half_rule=half_rule)
    transcoder = MLPDummyTranscoder(
        W_up, b_up, W_down, None, act_fn, layer_idx=0, jacobian_encoders=jacobian_encoders, **kwargs
    )
    return transcoder, torch.randn(N_POS, D_MODEL)


def autograd_encoder_rows(transcoder, input_acts, pos_idx, feat_idx):
    """d act[p, s] / d input[p] via autograd through the activation function's forward.

    The forward uses the frozen derivative with relp and the true one otherwise,
    and applies half_rule, so this is the reference for both encoder modes.
    """
    x = input_acts.detach().requires_grad_(True)
    transcoder.activation_function.set_original_inputs(x)
    acts = transcoder.activation_function(F.linear(x, transcoder.W_enc, transcoder.b_enc))
    rows = [
        torch.autograd.grad(acts[p, s], x, retain_graph=True)[0][p]
        for p, s in zip(pos_idx.tolist(), feat_idx.tolist())
    ]
    return torch.stack(rows)


@pytest.mark.parametrize("act", ACTIVATIONS)
@pytest.mark.parametrize(
    "relp,half_rule,bias",
    [(True, False, False), (True, True, False), (True, True, True), (False, False, False)],
    ids=["relp", "relp-half_rule", "relp-half_rule-bias", "exact"],
)
def test_encoder_rows_match_autograd(act, relp, half_rule, bias):
    transcoder, input_acts = make_transcoder(
        act, relp, half_rule, jacobian_encoders=not relp, bias=bias
    )
    sparse_acts, encoders = transcoder.encode_sparse(input_acts, zero_positions=slice(0, 0))
    pos_idx, feat_idx = sparse_acts.indices()
    assert len(pos_idx) > N_POS * D_MLP // 2  # no threshold: only exact zeros drop out

    expected = autograd_encoder_rows(transcoder, input_acts, pos_idx, feat_idx)
    torch.testing.assert_close(encoders, expected, atol=1e-5, rtol=1e-5)


def test_default_encoders_are_up_rows():
    transcoder, input_acts = make_transcoder("gelu_tanh", relp=False)
    sparse_acts, encoders = transcoder.encode_sparse(input_acts)
    _, feat_idx = sparse_acts.indices()
    assert torch.equal(encoders, transcoder.W_enc[feat_idx])


def test_no_active_neurons():
    transcoder, input_acts = make_transcoder("gelu_tanh", relp=True, activation_threshold=1e6)
    sparse_acts, encoders = transcoder.encode_sparse(input_acts)
    assert sparse_acts._nnz() == 0
    assert encoders.shape == (0, D_MODEL)


def test_relp_accepts_activation_functions():
    """TransformerLens passes activation functions, not modules."""
    assert GatedActivationFunction(torch.randn(D_MLP, D_MODEL), F.silu, relp=True).kind == "silu"


@pytest.mark.parametrize("act_fn", [nn.ReLU(), F.gelu], ids=["relu", "gelu_exact"])
def test_relp_rejects_activation_without_frozen_form(act_fn):
    with pytest.raises(ValueError, match="RelP requires"):
        GatedActivationFunction(torch.randn(D_MLP, D_MODEL), act_fn, relp=True)


@pytest.mark.parametrize("act", ACTIVATIONS)
def test_relp_half_rule_conserves_activation(act):
    """With relp and half_rule (and no up bias), encoder row . input = activation.

    row . x = 0.5 * (act(g) * up + up * gate(g) * g) = act(g) * up, since act(g) = g * gate(g).
    """
    transcoder, input_acts = make_transcoder(act, relp=True, half_rule=True)
    sparse_acts, encoders = transcoder.encode_sparse(input_acts, zero_positions=slice(0, 0))
    pos_idx, _ = sparse_acts.indices()
    torch.testing.assert_close(
        (encoders * input_acts[pos_idx]).sum(-1), sparse_acts.values(), atol=1e-5, rtol=1e-5
    )


def test_threshold_and_topk():
    transcoder = MLPDummyTranscoder(
        torch.eye(6), None, torch.zeros(6, 6), None, nn.Identity(), layer_idx=0, activation_topk=3
    )
    x = torch.tensor([[3.0, -1.0, 0.5, 2.0, -4.0, 0.1]])
    assert torch.equal(transcoder.encode(x), torch.tensor([[3.0, 0.0, 0.0, 2.0, -4.0, 0.0]]))

    # the threshold drops |a| <= 1.5 first; top-2 then selects among {3, 2, -4}
    transcoder.activation_threshold = 1.5
    transcoder.activation_topk = 2
    assert torch.equal(transcoder.encode(x), torch.tensor([[3.0, 0.0, 0.0, 0.0, -4.0, 0.0]]))

    # k >= d_transcoder keeps everything
    transcoder.activation_threshold = None
    transcoder.activation_topk = 6
    assert torch.equal(transcoder.encode(x), x)
