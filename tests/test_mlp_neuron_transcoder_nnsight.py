"""MLP neurons as transcoder features, nnsight backend, on tiny models of each family."""

import gc
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from nnsight import save
from transformers import (
    Gemma2Config,
    Gemma3Config,
    Gemma3TextConfig,
    GptOssConfig,
    LlamaConfig,
    Qwen3Config,
)

from circuit_tracer import ReplacementModel
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.transcoder import MLPNeuronTranscoderSet, MLPNeuronConfig
from circuit_tracer.utils.tl_nnsight_mapping import get_mapping

TINY_DIMS: dict[str, Any] = dict(
    hidden_size=8,
    intermediate_size=16,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=4,
    vocab_size=16,
)


def gemma3_vlm_config():
    vision_config = dict(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=14,
        patch_size=14,
    )
    return Gemma3Config(text_config=TINY_DIMS, vision_config=vision_config, mm_tokens_per_image=1)


CONFIGS = {
    "gemma2": (lambda: Gemma2Config(**TINY_DIMS), "Gemma2ForCausalLM"),
    "gemma3": (lambda: Gemma3TextConfig(**TINY_DIMS), "Gemma3ForCausalLM"),
    "gemma3-vlm": (gemma3_vlm_config, "Gemma3ForConditionalGeneration"),
    "llama": (lambda: LlamaConfig(**TINY_DIMS), "LlamaForCausalLM"),
    "qwen3": (lambda: Qwen3Config(**TINY_DIMS), "Qwen3ForCausalLM"),
}


def load_model(family: str, **mlp_config) -> NNSightReplacementModel:
    make_config, architecture = CONFIGS[family]
    config = make_config()
    config._name_or_path = "openai-community/gpt2"  # any ungated tokenizer
    config.architectures = [architecture]
    mlp_config.setdefault("activation_topk", None)
    mlp_config.setdefault("activation_threshold", None)
    model = ReplacementModel.from_config(config, MLPNeuronConfig(**mlp_config), backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)
    # the default init (std 0.02) makes neuron values ~1e-5, below assert_close's tolerance
    for param in model.parameters():
        torch.nn.init.uniform_(param, -1, 1)
    return model


def transcoder_set(model: NNSightReplacementModel) -> MLPNeuronTranscoderSet:
    transcoders = model.transcoders._module  # type: ignore
    assert isinstance(transcoders, MLPNeuronTranscoderSet)
    return transcoders


def get_mlp(model: NNSightReplacementModel, layer: int) -> Any:
    mapping = get_mapping(model.config.architectures[0])  # type: ignore
    return model._resolve_attr(model, mapping.mlp.format(layer=layer))


@contextmanager
def special_ids(model, ids: list[int]):
    tokenizer_class = type(model.tokenizer)
    original = tokenizer_class.all_special_ids
    tokenizer_class.all_special_ids = property(lambda self: ids)  # type: ignore
    try:
        yield
    finally:
        tokenizer_class.all_special_ids = original  # type: ignore


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


TOKENS = torch.tensor([0, 3, 6, 2, 6, 7])


@pytest.mark.parametrize("family", CONFIGS)
def test_transcoder_reproduces_mlp(family):
    model = load_model(family)
    with torch.no_grad():
        with model.trace(TOKENS):
            mlp = get_mlp(model, 0)
            mlp_input = save(mlp.input)
            hidden = save(mlp.down_proj.input)
            mlp_output = save(mlp.output)

        transcoder = transcoder_set(model)[0]
        acts = transcoder.encode(mlp_input)
        torch.testing.assert_close(acts, hidden)
        torch.testing.assert_close(transcoder.decode(acts), mlp_output)


@pytest.mark.parametrize("family", CONFIGS)
def test_sparse_matches_dense(family):
    model = load_model(family)
    transcoder = transcoder_set(model)[0]
    torch.manual_seed(42)
    x = torch.randn(5, transcoder.d_model, device=model.device)
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
    with special_ids(model, [0]), torch.no_grad():
        _, acts = model.get_activations(TOKENS)
        _, pre_acts = model.get_activations(TOKENS, apply_activation_function=False)
        expected_pre, expected_acts = [], []
        with model.trace(TOKENS):
            for layer in range(model.cfg.n_layers):  # nnsight needs execution order
                mlp = get_mlp(model, layer)
                expected_pre.append(save(mlp.up_proj.output[0]))
                expected_acts.append(save(mlp.down_proj.input[0]))

    expected_pre, expected_acts = torch.stack(expected_pre), torch.stack(expected_acts)
    expected_acts[:, 0] = 0  # the BOS position is zeroed
    expected_pre[:, 0] = 0
    torch.testing.assert_close(acts, expected_acts)
    torch.testing.assert_close(pre_acts, expected_pre)


@pytest.mark.parametrize("family", CONFIGS)
def test_activation_paths_agree_with_topk(family):
    """encode, encode_sparse and the set-level activation function apply the same top-k."""
    model = load_model(family, activation_topk=4)
    transcoders = transcoder_set(model)
    transcoder = transcoders[0]
    torch.manual_seed(0)
    x = torch.randn(5, transcoder.d_model, device=model.device)
    with torch.no_grad():
        acts = transcoder.encode(x)
        assert ((acts != 0).sum(-1) == 4).all()

        pre_acts = transcoder.encode(x, apply_activation_function=False)
        torch.testing.assert_close(transcoders.apply_activation_function(0, pre_acts), acts)

        sparse_acts, _ = transcoder.encode_sparse(x, zero_positions=slice(0, 0))
        torch.testing.assert_close(sparse_acts.to_dense(), acts)


def test_moe_mlp_unsupported():
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
        ReplacementModel.from_config(
            config,
            MLPNeuronConfig(activation_topk=None, activation_threshold=None),
            backend="nnsight",
        )
