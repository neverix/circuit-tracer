"""Tests for MLPDummyTranscoder and MLP attribution mode (NNSight backend)."""

import gc

import pytest
import torch
from transformers import Gemma2Config
from nnsight import save

from circuit_tracer.replacement_model.replacement_model_nnsight import (
    NNSightReplacementModel,
)
from circuit_tracer import ReplacementModel
from circuit_tracer.replacement_model.replacement_model import MLPTranscoderConfig
from circuit_tracer.transcoder import MLPDummyTranscoderSet


gemma_2_config_dict = {
    "architectures": ["Gemma2ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_logit_softcapping": 50.0,
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "eos_token_id": 1,
    "final_logit_softcapping": None,
    "head_dim": 256,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 2304,
    "initializer_range": 0.02,
    "intermediate_size": 9216,
    "max_position_embeddings": 8192,
    "model_type": "gemma2",
    "num_attention_heads": 8,
    "num_hidden_layers": 26,
    "num_key_value_heads": 4,
    "pad_token_id": 0,
    "query_pre_attn_scalar": 256,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "torch_dtype": "float32",
    "transformers_version": "4.42.4",
    "use_cache": True,
    "vocab_size": 256000,
    "_name_or_path": "openai-community/gpt2",
}

# Disable the activation threshold for tests.
MLP_NO_THRESHOLD = MLPTranscoderConfig(activation_threshold=None)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def create_small_gemma_config():
    """Create a small Gemma config for NNSight testing."""
    config = Gemma2Config.from_dict(gemma_2_config_dict)
    config.num_hidden_layers = 2
    config.hidden_size = 8
    config.intermediate_size = 16
    config.head_dim = 4
    config.vocab_size = 16
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.final_logit_softcapping = None  # type: ignore
    config.torch_dtype = "float32"
    return config


def _unwrap_transcoders(model):
    """Unwrap NNSight Envoy to get the actual MLPDummyTranscoderSet."""
    return model.transcoders._module if hasattr(model.transcoders, '_module') else model.transcoders


def test_nnsight_mlp_dummy_transcoder_output_matches_mlp():
    """Test that MLPDummyTranscoder produces the same output as the original MLP."""
    config = create_small_gemma_config()

    model = ReplacementModel.from_config(config, MLP_NO_THRESHOLD, backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    from circuit_tracer.utils.tl_nnsight_mapping import get_mapping
    nnsight_config = get_mapping(model.config.architectures[0])  # type: ignore
    mlp = model._resolve_attr(model, nnsight_config.mlp.format(layer=0))
    transcoders = _unwrap_transcoders(model)
    assert isinstance(transcoders, MLPDummyTranscoderSet)
    transcoder = transcoders[0]

    torch.manual_seed(42)
    test_input = torch.randn(3, config.hidden_size, dtype=torch.float32)

    with torch.no_grad():
        gate_out = mlp.act_fn(mlp.gate_proj(test_input))  # type: ignore
        up_out = mlp.up_proj(test_input)  # type: ignore
        hidden = gate_out * up_out
        mlp_output = mlp.down_proj(hidden)  # type: ignore

        transcoder_encoded = transcoder.encode(test_input)
        transcoder_output = transcoder.decode(transcoder_encoded)

    assert torch.allclose(transcoder_encoded, hidden, atol=1e-5)
    assert torch.allclose(transcoder_output, mlp_output, atol=1e-5)

def test_nnsight_mlp_dummy_transcoder_output_matches_mlp_trace():
    """Test that MLPDummyTranscoder produces the same output as the original MLP."""
    config = create_small_gemma_config()

    model = ReplacementModel.from_config(config, MLP_NO_THRESHOLD, backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    transcoders = _unwrap_transcoders(model)
    assert isinstance(transcoders, MLPDummyTranscoderSet)
    transcoder = transcoders[0]

    torch.manual_seed(42)

    with torch.no_grad():
        with model.trace([0, 3, 6, 2, 6, 7]):
            mlp_input = save(model.model.layers[0].mlp.input)
            hidden = save(model.model.layers[0].mlp.down_proj.input)
            mlp_output = save(model.model.layers[0].mlp.output)

        transcoder_encoded = transcoder.encode(mlp_input)
        transcoder_output = transcoder.decode(transcoder_encoded)

    assert torch.allclose(transcoder_encoded, hidden, atol=1e-5)
    assert torch.allclose(transcoder_output, mlp_output, atol=1e-5)


def test_nnsight_mlp_dummy_transcoder_encode_sparse():
    """Test that encode_sparse produces correct sparse activations."""
    config = create_small_gemma_config()

    model = ReplacementModel.from_config(config, MLP_NO_THRESHOLD, backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    transcoders = _unwrap_transcoders(model)
    assert isinstance(transcoders, MLPDummyTranscoderSet)
    transcoder = transcoders[0]

    torch.manual_seed(42)
    test_input = torch.randn(5, config.hidden_size, dtype=torch.float32)

    with torch.no_grad():
        dense_acts = transcoder.encode(test_input)
        sparse_acts, _ = transcoder.encode_sparse(test_input, zero_positions=slice(0, 1))

        dense_acts_zeroed = dense_acts.clone()
        dense_acts_zeroed[0] = 0
        sparse_dense = sparse_acts.to_dense()

    assert torch.allclose(sparse_dense, dense_acts_zeroed, atol=1e-5)


def test_nnsight_mlp_dummy_transcoder_decode_sparse():
    """Test that decode_sparse produces correct reconstruction."""
    config = create_small_gemma_config()

    model = ReplacementModel.from_config(config, MLP_NO_THRESHOLD, backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    transcoders = _unwrap_transcoders(model)
    assert isinstance(transcoders, MLPDummyTranscoderSet)
    transcoder = transcoders[0]

    torch.manual_seed(42)
    test_input = torch.randn(5, config.hidden_size, dtype=torch.float32)

    with torch.no_grad():
        sparse_acts, _ = transcoder.encode_sparse(test_input, zero_positions=slice(0, 1))
        reconstruction, _ = transcoder.decode_sparse(sparse_acts)

        dense_acts = transcoder.encode(test_input)
        dense_acts[0] = 0
        direct_decode = transcoder.decode(dense_acts)

    assert torch.allclose(reconstruction, direct_decode, atol=1e-5)


def test_nnsight_mlp_get_activations():
    """Test that get_activations works correctly with MLP transcoders."""
    config = create_small_gemma_config()
    model = ReplacementModel.from_config(config, MLP_NO_THRESHOLD, backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    s = torch.tensor([0, 3, 4, 3, 2])

    tokenizer_class = type(model.tokenizer)
    original_all_special_ids = tokenizer_class.all_special_ids
    try:
        tokenizer_class.all_special_ids = property(lambda self: [0])  # type: ignore

        logits, activation_cache = model.get_activations(s)

        assert logits.shape[0] == 1
        assert logits.shape[1] == len(s)
        assert logits.shape[2] == config.vocab_size

        assert activation_cache.shape[0] == config.num_hidden_layers
        assert activation_cache.shape[1] == len(s)
        assert activation_cache.shape[2] == config.intermediate_size

        assert not torch.isnan(activation_cache).any()
        assert not torch.isinf(activation_cache).any()
        assert (activation_cache[:, 0] == 0).all()

        traced_cache, pre_act_traced_cache = [], []
        with model.trace(s):
            for layer in range(model.cfg.n_layers):
                pre_act_traced_cache.append(save(model.model.layers[layer].mlp.up_proj.output.squeeze(0)))
                traced_cache.append(save(model.model.layers[layer].mlp.down_proj.input.squeeze(0)))

        traced_cache = torch.stack(traced_cache)
        pre_act_traced_cache = torch.stack(pre_act_traced_cache)

        traced_cache[:, 0] = 0
        pre_act_traced_cache[:, 0] = 0
        assert torch.allclose(activation_cache, traced_cache)

        _, pre_activation_cache = model.get_activations(s, apply_activation_function=False)
        assert torch.allclose(pre_activation_cache, pre_act_traced_cache)

    finally:
        tokenizer_class.all_special_ids = original_all_special_ids  # type: ignore
