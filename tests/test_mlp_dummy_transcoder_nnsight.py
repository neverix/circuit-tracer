"""Tests for MLPDummyTranscoder and MLP attribution mode (NNSight backend)."""

import gc
from typing import Any

import pytest
import torch
from transformers import Gemma2Config, Gemma3Config, Gemma3TextConfig, LlamaConfig, Qwen3Config
from nnsight import save

from circuit_tracer.replacement_model.replacement_model_nnsight import (
    NNSightReplacementModel,
)
from circuit_tracer import ReplacementModel
from circuit_tracer.replacement_model.replacement_model import MLPTranscoderConfig
from circuit_tracer.transcoder import MLPDummyTranscoderSet
from circuit_tracer.transcoder.mlp_dummy_transcoder import MLPDummyTranscoder
from circuit_tracer.utils.tl_nnsight_mapping import get_mapping


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

# Shared tiny dimensions for all model families.
TINY_DIMS: dict[str, Any] = {
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "vocab_size": 16,
}


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def create_small_gemma2_config():
    """Create a small Gemma 2 config for NNSight testing."""
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


def create_small_gemma3_config():
    """Create a small Gemma 3 (text-only) config for NNSight testing."""
    config = Gemma3TextConfig(**TINY_DIMS)
    config._name_or_path = "openai-community/gpt2"
    config.architectures = ["Gemma3ForCausalLM"]
    return config


def create_small_gemma3_conditional_config():
    """Create a small Gemma 3 multimodal config for NNSight testing."""
    vision_config = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "image_size": 14,
        "patch_size": 14,
        "num_channels": 3,
    }
    config = Gemma3Config(
        text_config=dict(TINY_DIMS), vision_config=vision_config, mm_tokens_per_image=1
    )
    config._name_or_path = "openai-community/gpt2"
    config.architectures = ["Gemma3ForConditionalGeneration"]
    return config


def create_small_llama_config():
    """Create a small Llama config for NNSight testing."""
    config = LlamaConfig(**TINY_DIMS)
    config._name_or_path = "openai-community/gpt2"
    config.architectures = ["LlamaForCausalLM"]
    return config


def create_small_qwen3_config():
    """Create a small Qwen 3 config for NNSight testing."""
    config = Qwen3Config(**TINY_DIMS)
    config._name_or_path = "openai-community/gpt2"
    config.architectures = ["Qwen3ForCausalLM"]
    return config


CONFIG_FACTORIES = {
    "gemma2": create_small_gemma2_config,
    "gemma3": create_small_gemma3_config,
    "gemma3-vlm": create_small_gemma3_conditional_config,
    "llama": create_small_llama_config,
    "qwen3": create_small_qwen3_config,
}

MODEL_FAMILIES = list(CONFIG_FACTORIES)


def _unwrap_transcoders(model) -> MLPDummyTranscoderSet:
    """Unwrap NNSight Envoy to get the actual MLPDummyTranscoderSet."""
    transcoders = (
        model.transcoders._module if hasattr(model.transcoders, "_module") else model.transcoders
    )
    assert isinstance(transcoders, MLPDummyTranscoderSet)
    return transcoders


def _get_mlp(model, layer: int) -> Any:
    """Resolve the layer's MLP module via the architecture mapping.

    Works both outside and inside a trace context (Envoy attribute chains),
    and handles the nested language tower of Gemma3ForConditionalGeneration.
    """
    mapping = get_mapping(model.config.architectures[0])
    return NNSightReplacementModel._resolve_attr(model, mapping.mlp.format(layer=layer))


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_nnsight_mlp_dummy_transcoder_output_matches_mlp(family):
    """Test that MLPDummyTranscoder produces the same output as the original MLP."""
    config = CONFIG_FACTORIES[family]()
    text_config = config.get_text_config()

    model = ReplacementModel.from_config(config, MLP_NO_THRESHOLD, backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    mlp = _get_mlp(model, 0)
    transcoders = _unwrap_transcoders(model)
    assert isinstance(transcoders, MLPDummyTranscoderSet)
    transcoder = transcoders[0]

    torch.manual_seed(42)
    test_input = torch.randn(3, text_config.hidden_size, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        gate_out = mlp.act_fn(mlp.gate_proj(test_input))  # type: ignore
        up_out = mlp.up_proj(test_input)  # type: ignore
        hidden = gate_out * up_out
        mlp_output = mlp.down_proj(hidden)  # type: ignore

        transcoder_encoded = transcoder.encode(test_input)
        transcoder_output = transcoder.decode(transcoder_encoded)

    assert torch.allclose(transcoder_encoded, hidden, atol=1e-5)
    assert torch.allclose(transcoder_output, mlp_output, atol=1e-5)


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_nnsight_mlp_dummy_transcoder_output_matches_mlp_trace(family):
    """Test that MLPDummyTranscoder produces the same output as the original MLP."""
    config = CONFIG_FACTORIES[family]()

    model = ReplacementModel.from_config(config, MLP_NO_THRESHOLD, backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    transcoders = _unwrap_transcoders(model)
    assert isinstance(transcoders, MLPDummyTranscoderSet)
    transcoder = transcoders[0]

    torch.manual_seed(42)

    with torch.no_grad():
        with model.trace([0, 3, 6, 2, 6, 7]):
            mlp = _get_mlp(model, 0)
            mlp_input = save(mlp.input)
            hidden = save(mlp.down_proj.input)
            mlp_output = save(mlp.output)

        transcoder_encoded = transcoder.encode(mlp_input)
        transcoder_output = transcoder.decode(transcoder_encoded)

    assert torch.allclose(transcoder_encoded, hidden, atol=1e-5)
    assert torch.allclose(transcoder_output, mlp_output, atol=1e-5)


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_nnsight_mlp_dummy_transcoder_encode_sparse(family):
    """Test that encode_sparse produces correct sparse activations."""
    config = CONFIG_FACTORIES[family]()
    text_config = config.get_text_config()

    model = ReplacementModel.from_config(config, MLP_NO_THRESHOLD, backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    transcoders = _unwrap_transcoders(model)
    assert isinstance(transcoders, MLPDummyTranscoderSet)
    transcoder = transcoders[0]

    torch.manual_seed(42)
    test_input = torch.randn(5, text_config.hidden_size, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        dense_acts = transcoder.encode(test_input)
        sparse_acts, _ = transcoder.encode_sparse(test_input, zero_positions=slice(0, 1))

        dense_acts_zeroed = dense_acts.clone()
        dense_acts_zeroed[0] = 0
        sparse_dense = sparse_acts.to_dense()

    assert torch.allclose(sparse_dense, dense_acts_zeroed, atol=1e-5)


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_nnsight_mlp_dummy_transcoder_decode_sparse(family):
    """Test that decode_sparse produces correct reconstruction."""
    config = CONFIG_FACTORIES[family]()
    text_config = config.get_text_config()

    model = ReplacementModel.from_config(config, MLP_NO_THRESHOLD, backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    transcoders = _unwrap_transcoders(model)
    assert isinstance(transcoders, MLPDummyTranscoderSet)
    transcoder = transcoders[0]

    torch.manual_seed(42)
    test_input = torch.randn(5, text_config.hidden_size, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        sparse_acts, _ = transcoder.encode_sparse(test_input, zero_positions=slice(0, 1))
        reconstruction, _ = transcoder.decode_sparse(sparse_acts)

        dense_acts = transcoder.encode(test_input)
        dense_acts[0] = 0
        direct_decode = transcoder.decode(dense_acts)

    assert torch.allclose(reconstruction, direct_decode, atol=1e-5)


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_nnsight_mlp_get_activations(family):
    """Test that get_activations works correctly with MLP transcoders."""
    config = CONFIG_FACTORIES[family]()
    text_config = config.get_text_config()

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
        assert logits.shape[2] == text_config.vocab_size

        assert activation_cache.shape[0] == text_config.num_hidden_layers
        assert activation_cache.shape[1] == len(s)
        assert activation_cache.shape[2] == text_config.intermediate_size

        assert not torch.isnan(activation_cache).any()
        assert not torch.isinf(activation_cache).any()
        assert (activation_cache[:, 0] == 0).all()

        traced_cache, pre_act_traced_cache = [], []
        with model.trace(s):
            for layer in range(model.cfg.n_layers):
                mlp = _get_mlp(model, layer)
                pre_act_traced_cache.append(save(mlp.up_proj.output.squeeze(0)))
                traced_cache.append(save(mlp.down_proj.input.squeeze(0)))

        traced_cache = torch.stack(traced_cache)
        pre_act_traced_cache = torch.stack(pre_act_traced_cache)

        traced_cache[:, 0] = 0
        pre_act_traced_cache[:, 0] = 0
        assert torch.allclose(activation_cache, traced_cache)

        _, pre_activation_cache = model.get_activations(s, apply_activation_function=False)
        assert torch.allclose(pre_activation_cache, pre_act_traced_cache)

    finally:
        tokenizer_class.all_special_ids = original_all_special_ids  # type: ignore


def test_nnsight_mlp_dummy_transcoder_gpt_oss_unsupported():
    """GPT-OSS has MoE MLPs (no up_proj); the dummy-MLP path must fail clearly."""
    from transformers import GptOssConfig

    config = GptOssConfig(
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=512,
        layer_types=["sliding_attention", "full_attention"],
    )
    config._name_or_path = "openai-community/gpt2"
    config.architectures = ["GptOssForCausalLM"]

    with pytest.raises(NotImplementedError, match="MoE"):
        ReplacementModel.from_config(config, MLP_NO_THRESHOLD, backend="nnsight")


def test_activation_topk_hand_computed():
    """activation_topk on hand-picked activations, expected output as literals."""
    from torch import nn

    transcoder = MLPDummyTranscoder(
        W_enc=torch.eye(6),
        b_enc=None,
        W_dec=torch.zeros(6, 6),
        b_dec=None,
        activation_function=nn.Identity(),
        layer_idx=0,
        activation_topk=3,
    )
    x = torch.tensor([[3.0, -1.0, 0.5, 2.0, -4.0, 0.1]])
    assert torch.equal(transcoder.encode(x), torch.tensor([[3.0, 0.0, 0.0, 2.0, -4.0, 0.0]]))

    # threshold kills |a| <= 1.5 -> {-1.0, 0.5, 0.1}; top-2 of the survivors {3, 2, -4}
    transcoder.activation_threshold = 1.5
    transcoder.activation_topk = 2
    assert torch.equal(transcoder.encode(x), torch.tensor([[3.0, 0.0, 0.0, 0.0, -4.0, 0.0]]))

    # k >= d_transcoder is a no-op
    transcoder.activation_threshold = None
    transcoder.activation_topk = 6
    assert torch.equal(transcoder.encode(x), x)


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_nnsight_mlp_dummy_transcoder_activation_topk(family):
    """activation_topk keeps the k largest-|a| neurons per position, values
    untouched; encode, encode_sparse and the set-level gate agree."""
    k = 4
    config = CONFIG_FACTORIES[family]()
    text_config = config.get_text_config()

    model = ReplacementModel.from_config(
        config,
        MLPTranscoderConfig(activation_threshold=None, activation_topk=k),
        backend="nnsight",
    )
    assert isinstance(model, NNSightReplacementModel)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    transcoders = _unwrap_transcoders(model)
    assert isinstance(transcoders, MLPDummyTranscoderSet)
    transcoder = transcoders[0]
    assert transcoder.activation_topk == k

    torch.manual_seed(0)
    test_input = torch.randn(5, text_config.hidden_size, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        gated = transcoder.encode(test_input)
        transcoder.activation_topk = None
        ungated = transcoder.encode(test_input)
        transcoder.activation_topk = k

        # exactly k survivors per position (random floats: no |a| ties)
        assert ((gated != 0).sum(dim=-1) == k).all()
        # survivors keep their original values
        survivor_mask = gated != 0
        assert torch.equal(gated[survivor_mask], ungated[survivor_mask])
        # every survivor is at least as large in |a| as every casualty
        inf = torch.tensor(torch.inf, device=gated.device)
        min_kept = torch.where(survivor_mask, gated.abs(), inf).amin(dim=-1)
        max_dropped = torch.where(survivor_mask, -inf, ungated.abs()).amax(dim=-1)
        assert (max_dropped <= min_kept).all()

        # encode_sparse applies the same gate
        sparse_acts, _ = transcoder.encode_sparse(test_input, zero_positions=slice(0, 1))
        gated_zeroed = gated.clone()
        gated_zeroed[0] = 0
        assert torch.allclose(sparse_acts.to_dense(), gated_zeroed, atol=1e-6)

        # set-level apply_activation_function applies the same gate
        pre_acts = transcoder.encode(test_input, apply_activation_function=False)
        set_gated = transcoders.apply_activation_function(0, pre_acts)
        assert torch.equal(set_gated, gated)


def test_nnsight_mlp_dummy_transcoder_topk_composes_with_threshold():
    """With both set, the threshold applies first and top-k selects among survivors."""
    k = 8
    threshold = 0.25
    config = create_small_llama_config()
    text_config = config.get_text_config()

    model = ReplacementModel.from_config(
        config,
        MLPTranscoderConfig(activation_threshold=threshold, activation_topk=k),
        backend="nnsight",
    )
    assert isinstance(model, NNSightReplacementModel)
    transcoder = _unwrap_transcoders(model)[0]
    assert isinstance(transcoder, MLPDummyTranscoder)

    torch.manual_seed(0)
    test_input = torch.randn(5, text_config.hidden_size, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        gated = transcoder.encode(test_input)
        transcoder.activation_threshold = None
        transcoder.activation_topk = None
        raw = transcoder.encode(test_input)
        transcoder.activation_threshold = threshold
        transcoder.activation_topk = k

    # every survivor cleared the threshold and kept its value
    survivor_mask = gated != 0
    assert ((gated != 0).sum(dim=-1) <= k).all()
    assert (gated[survivor_mask].abs() > threshold).all()
    assert torch.equal(gated[survivor_mask], raw[survivor_mask])
    # no dropped above-threshold value out-ranks a survivor
    inf = torch.tensor(torch.inf, device=gated.device)
    min_kept = torch.where(survivor_mask, gated.abs(), inf).amin(dim=-1)
    dropped_abs = torch.where(survivor_mask, -inf, raw.abs())
    dropped_abs = torch.where(dropped_abs > threshold, dropped_abs, -inf)
    assert (dropped_abs.amax(dim=-1) <= min_kept).all()
