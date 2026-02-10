"""Tests for MLPDummyTranscoder and MLP attribution mode (TransformerLens backend)."""

import gc

import numpy as np
import pytest
import torch
from transformer_lens import HookedTransformerConfig

from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer import ReplacementModel
from circuit_tracer.utils import get_default_device

# Use 'mlp-0' to disable the activation threshold for tests, matching the
# nnsight tests and avoiding any Envoy wrapping issues.
MLP_NO_THRESHOLD = 'mlp-0'


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def create_small_tl_config():
    """Create a small TransformerLens config for testing."""
    return HookedTransformerConfig.from_dict({
        "n_layers": 2,
        "d_model": 8,
        "n_ctx": 64,
        "d_head": 4,
        "model_name": "gemma-2-2b",
        "n_heads": 2,
        "d_mlp": 16,
        "act_fn": "gelu_pytorch_tanh",
        "d_vocab": 16,
        "eps": 1e-06,
        "use_attn_result": False,
        "use_attn_scale": True,
        "attn_scale": np.float64(16.0),
        "use_local_attn": True,
        "original_architecture": "Gemma2ForCausalLM",
        "tokenizer_name": "gpt2",
        "window_size": 4096,
        "attn_types": ["global", "local"],
        "normalization_type": "RMSPre",
        "device": get_default_device(),
        "attention_dir": "causal",
        "positional_embedding_type": "rotary",
        "final_rms": True,
        "d_vocab_out": 16,
        "rotary_dim": 4,
        "use_hook_tokens": False,
        "gated_mlp": True,
        "default_prepend_bos": True,
        "dtype": torch.float32,
        "tokenizer_prepends_bos": True,
        "n_key_value_heads": 2,
        "rotary_base": 10000.0,
        "use_normalization_before_and_after": True,
        "attn_scores_soft_cap": 50.0,
        "output_logits_soft_cap": 0.0,
    })


def test_tl_mlp_dummy_transcoder_output_matches_mlp():
    """Test that MLPDummyTranscoder produces the same output as the original MLP."""
    cfg = create_small_tl_config()
    model = ReplacementModel.from_config(cfg, MLP_NO_THRESHOLD, backend="transformerlens")
    assert isinstance(model, TransformerLensReplacementModel)

    mlp = model.blocks[0].mlp.old_mlp
    transcoder = model.transcoders[0]

    torch.manual_seed(42)
    test_input = torch.randn(3, cfg.d_model, dtype=torch.float32, device=cfg.device)

    with torch.no_grad():
        if hasattr(mlp, 'W_gate'):
            gate_out = mlp.act_fn(test_input @ mlp.W_gate)
            hidden = gate_out * (test_input @ mlp.W_in)
        else:
            hidden = mlp.act_fn(test_input @ mlp.W_in)
        mlp_output = hidden @ mlp.W_out + mlp.b_out

        transcoder_encoded = transcoder.encode(test_input)
        transcoder_output = transcoder.decode(transcoder_encoded)

    assert torch.allclose(transcoder_encoded, hidden, atol=1e-5)
    assert torch.allclose(transcoder_output, mlp_output, atol=1e-5)


def test_tl_mlp_dummy_transcoder_encode_sparse():
    """Test that encode_sparse produces correct sparse activations."""
    cfg = create_small_tl_config()
    model = ReplacementModel.from_config(cfg, MLP_NO_THRESHOLD, backend="transformerlens")
    assert isinstance(model, TransformerLensReplacementModel)

    transcoder = model.transcoders[0]

    torch.manual_seed(42)
    test_input = torch.randn(5, cfg.d_model, dtype=torch.float32, device=cfg.device)

    with torch.no_grad():
        dense_acts = transcoder.encode(test_input)
        sparse_acts, _ = transcoder.encode_sparse(test_input, zero_positions=slice(0, 1))

        dense_acts_zeroed = dense_acts.clone()
        dense_acts_zeroed[0] = 0
        sparse_dense = sparse_acts.to_dense()

    assert torch.allclose(sparse_dense, dense_acts_zeroed, atol=1e-5)


def test_tl_mlp_dummy_transcoder_decode_sparse():
    """Test that decode_sparse produces correct reconstruction."""
    cfg = create_small_tl_config()
    model = ReplacementModel.from_config(cfg, MLP_NO_THRESHOLD, backend="transformerlens")
    assert isinstance(model, TransformerLensReplacementModel)

    transcoder = model.transcoders[0]

    torch.manual_seed(42)
    test_input = torch.randn(5, cfg.d_model, dtype=torch.float32, device=cfg.device)

    with torch.no_grad():
        sparse_acts, _ = transcoder.encode_sparse(test_input, zero_positions=slice(0, 1))
        reconstruction, _ = transcoder.decode_sparse(sparse_acts)

        dense_acts = transcoder.encode(test_input)
        dense_acts[0] = 0
        direct_decode = transcoder.decode(dense_acts)

    assert torch.allclose(reconstruction, direct_decode, atol=1e-5)


def test_tl_mlp_get_activations():
    """Test that get_activations works correctly with MLP transcoders."""
    cfg = create_small_tl_config()
    model = ReplacementModel.from_config(cfg, MLP_NO_THRESHOLD, backend="transformerlens")
    assert isinstance(model, TransformerLensReplacementModel)

    type(model.tokenizer).all_special_ids = property(lambda self: [0])

    s = torch.tensor([0, 3, 4, 3, 2])

    logits, activation_cache = model.get_activations(s)

    assert logits.shape[0] == 1
    assert logits.shape[1] == len(s)
    assert logits.shape[2] == cfg.d_vocab

    assert activation_cache.shape[0] == cfg.n_layers
    assert activation_cache.shape[1] == len(s)
    assert activation_cache.shape[2] == cfg.d_mlp

    assert not torch.isnan(activation_cache).any()
    assert not torch.isinf(activation_cache).any()
    assert (activation_cache[:, 0] == 0).all()

    _, cache = model.run_with_cache(s)

    tl_activation_cache = torch.stack([cache[f'blocks.{layer}.mlp.old_mlp.hook_post'].squeeze(0) for layer in range(model.cfg.n_layers)])
    pre_act_hook = 'blocks.{layer}.mlp.old_mlp.hook_pre_linear' if 'blocks.0.mlp.old_mlp.hook_pre_linear' in cache else 'blocks.{layer}.mlp.old_mlp.hook_pre'
    tl_pre_activation_cache = torch.stack([cache[pre_act_hook.format(layer=layer)].squeeze(0) for layer in range(model.cfg.n_layers)])
    _, pre_activation_cache = model.get_activations(s, apply_activation_function=False)

    tl_activation_cache[:, 0] = 0
    tl_pre_activation_cache[:, 0] = 0

    assert torch.allclose(activation_cache, tl_activation_cache)
    assert torch.allclose(pre_activation_cache, tl_pre_activation_cache)
