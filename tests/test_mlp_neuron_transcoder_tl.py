"""MLP neurons as transcoder features, TransformerLens backend, on tiny models of each family."""

import gc

import pytest
import torch
from transformer_lens import HookedTransformerConfig

from circuit_tracer import ReplacementModel
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.transcoder import MLPNeuronTranscoderSet, MLPNeuronConfig
from circuit_tracer.utils import get_default_device

TINY_DIMS = dict(
    n_layers=2,
    d_model=8,
    d_head=4,
    n_heads=2,
    n_key_value_heads=2,
    d_mlp=16,
    d_vocab=16,
    n_ctx=64,
    normalization_type="RMSPre",
    positional_embedding_type="rotary",
    rotary_dim=4,
    gated_mlp=True,
    tokenizer_name="gpt2",  # any ungated tokenizer
)

CONFIGS = {
    # gated GELU(tanh), norms before and after each sublayer, soft-capped attention
    "gemma2": dict(
        model_name="gemma-2-2b",
        original_architecture="Gemma2ForCausalLM",
        act_fn="gelu_pytorch_tanh",
        use_normalization_before_and_after=True,
        attn_scores_soft_cap=50.0,
    ),
    "llama": dict(
        model_name="llama-3.1-8b", original_architecture="LlamaForCausalLM", act_fn="silu"
    ),
    "qwen3": dict(
        model_name="qwen3-0.6b",
        original_architecture="Qwen3ForCausalLM",
        act_fn="silu",
        use_qk_norm=True,
    ),
}


def load_model(family: str, **mlp_config) -> TransformerLensReplacementModel:
    cfg = HookedTransformerConfig(**TINY_DIMS, **CONFIGS[family], device=str(get_default_device()))
    mlp_config.setdefault("activation_topk", None)
    mlp_config.setdefault("activation_threshold", None)
    model = ReplacementModel.from_config(cfg, MLPNeuronConfig(**mlp_config))
    assert isinstance(model, TransformerLensReplacementModel)
    return model


def transcoder_set(model: TransformerLensReplacementModel) -> MLPNeuronTranscoderSet:
    assert isinstance(model.transcoders, MLPNeuronTranscoderSet)
    return model.transcoders


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


TOKENS = torch.tensor([0, 3, 4, 3, 2])


def mlp_cache(model, tokens):
    """Post- and pre-activations of the original MLPs, BOS position zeroed."""
    _, cache = model.run_with_cache(tokens)
    layers = range(model.cfg.n_layers)
    post = torch.stack([cache[f"blocks.{i}.mlp.old_mlp.hook_post"][0] for i in layers])
    pre = torch.stack([cache[f"blocks.{i}.mlp.old_mlp.hook_pre_linear"][0] for i in layers])
    post[:, 0] = 0
    pre[:, 0] = 0
    return post, pre


@pytest.mark.parametrize("family", CONFIGS)
def test_transcoder_reproduces_mlp(family):
    model = load_model(family)
    _, cache = model.run_with_cache(TOKENS)
    mlp_input = cache["blocks.0.mlp.hook_in"][0]
    transcoder = transcoder_set(model)[0]
    with torch.no_grad():
        acts = transcoder.encode(mlp_input)
    torch.testing.assert_close(acts, cache["blocks.0.mlp.old_mlp.hook_post"][0])
    torch.testing.assert_close(transcoder.decode(acts), cache["blocks.0.mlp.hook_out"][0])


@pytest.mark.parametrize("family", CONFIGS)
def test_sparse_matches_dense(family):
    model = load_model(family)
    transcoder = transcoder_set(model)[0]
    torch.manual_seed(42)
    x = torch.randn(5, transcoder.d_model, device=model.cfg.device)
    with torch.no_grad():
        dense = transcoder.encode(x)
        dense[0] = 0
        sparse_acts, _ = transcoder.encode_sparse(x, zero_positions=slice(0, 1))
        reconstruction, _ = transcoder.decode_sparse(sparse_acts)
    torch.testing.assert_close(sparse_acts.to_dense(), dense)
    torch.testing.assert_close(reconstruction, transcoder.decode(dense))


@pytest.mark.parametrize("family", CONFIGS)
def test_get_activations(family):
    model = load_model(family)
    type(model.tokenizer).all_special_ids = property(lambda self: [0])  # type: ignore
    _, acts = model.get_activations(TOKENS)
    _, pre_acts = model.get_activations(TOKENS, apply_activation_function=False)
    expected_acts, expected_pre = mlp_cache(model, TOKENS)
    torch.testing.assert_close(acts, expected_acts)
    torch.testing.assert_close(pre_acts, expected_pre)


@pytest.mark.parametrize("family", CONFIGS)
def test_activation_paths_agree_with_topk(family):
    """encode, encode_sparse and the set-level activation function apply the same top-k."""
    model = load_model(family, activation_topk=4)
    transcoder = transcoder_set(model)[0]
    torch.manual_seed(0)
    x = torch.randn(5, transcoder.d_model, device=model.cfg.device)
    with torch.no_grad():
        acts = transcoder.encode(x)
        assert ((acts != 0).sum(-1) == 4).all()

        pre_acts = transcoder.encode(x, apply_activation_function=False)
        torch.testing.assert_close(
            transcoder_set(model).apply_activation_function(0, pre_acts), acts
        )

        sparse_acts, _ = transcoder.encode_sparse(x, zero_positions=slice(0, 0))
        torch.testing.assert_close(sparse_acts.to_dense(), acts)


@pytest.mark.parametrize("family", CONFIGS)
def test_relp(family):
    """RelP on TransformerLens (plain activation functions): rows conserve activations."""
    model = load_model(family, encoders="relp", half_rule=True)
    transcoder = transcoder_set(model)[0]
    assert transcoder.b_enc is not None
    transcoder.b_enc.zero_()  # conservation needs no up bias
    x = torch.randn(5, transcoder.d_model, device=model.cfg.device)
    sparse_acts, encoders = transcoder.encode_sparse(x, zero_positions=slice(0, 0))
    pos_idx, _ = sparse_acts.indices()
    assert not encoders.requires_grad
    torch.testing.assert_close((encoders * x[pos_idx]).sum(-1), sparse_acts.values())
