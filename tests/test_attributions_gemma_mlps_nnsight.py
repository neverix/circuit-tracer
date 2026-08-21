import gc
from functools import partial

import pytest
import torch
import torch.nn as nn
from nnsight import save
from tqdm import tqdm
from transformers import AutoConfig, Gemma2Config

from circuit_tracer import Graph, ReplacementModel
from circuit_tracer.attribution.attribute_nnsight import attribute
from circuit_tracer.replacement_model.replacement_model import MLPTranscoderConfig
from circuit_tracer.replacement_model.replacement_model_nnsight import (
    NNSightReplacementModel,
)

gemma_2_config_dict = {
    "architectures": ["Gemma2ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_logit_softcapping": 50.0,
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "eos_token_id": 1,
    "final_logit_softcapping": 30.0,
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

gemma_2_config = Gemma2Config.from_dict(gemma_2_config_dict)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def verify_token_and_error_edges(
    model: NNSightReplacementModel,
    graph: Graph,
    act_atol=1e-3,
    act_rtol=1e-3,
    logit_atol=1e-5,
    logit_rtol=1e-3,
    apply_activation_function: bool = False,
):
    s = graph.input_tokens
    adjacency_matrix = graph.adjacency_matrix.to(device=model.device, dtype=model.dtype)
    active_features = graph.active_features.to(device=model.device)
    logit_tokens = graph.logit_tokens.to(device=model.device)
    total_active_features = active_features.size(0)
    pos_start = 1  # skip first position (BOS token)

    ctx = model.setup_attribution(s)

    error_vectors = ctx.error_vectors
    token_vectors = ctx.token_vectors

    logits, activation_cache = model.get_activations(s, apply_activation_function=apply_activation_function)
    logits = logits.squeeze(0)

    relevant_activations = activation_cache[
        active_features[:, 0], active_features[:, 1], active_features[:, 2]
    ]
    relevant_logits = logits[-1, logit_tokens]
    demeaned_relevant_logits = relevant_logits - logits[-1].mean()

    def verify_intervention(expected_effects, intervention, target_layer):
        _, freeze_fns = model.setup_intervention_with_freeze(
            s,
            constrained_layers=range(model.cfg.n_layers),  # type: ignore
        )
        _, activation_fn = model.get_activation_fn(apply_activation_function=apply_activation_function)

        with model.trace() as tracer:
            with tracer.invoke(s):
                pass

            direct_effects_barrier = tracer.barrier(2)

            with tracer.invoke():
                _, new_activation_cache = activation_fn()  # type:ignore
                new_activation_cache = save(new_activation_cache)  # type: ignore

            for i, freeze_fn in enumerate(freeze_fns):
                with tracer.invoke():
                    freeze_fn(direct_effects_barrier=direct_effects_barrier)

            with tracer.invoke():
                if "embed" == target_layer:
                    intervention(model.embed_location)

                for layer, feature_output_loc in enumerate(model.feature_output_locs):
                    direct_effects_barrier()
                    if layer == target_layer:
                        intervention(feature_output_loc)

                new_logits = save(model.output.logits.squeeze(0))  # type: ignore

        new_relevant_activations = new_activation_cache[
            active_features[:, 0], active_features[:, 1], active_features[:, 2]
        ]
        new_relevant_logits = new_logits[-1, logit_tokens]
        new_demeaned_relevant_logits = new_relevant_logits - new_logits[-1].mean()

        expected_activation_difference = expected_effects[:total_active_features]
        expected_logit_difference = expected_effects[-len(logit_tokens) :]

        assert torch.allclose(
            new_relevant_activations,
            relevant_activations + expected_activation_difference,
            atol=act_atol,
            rtol=act_rtol,
        )
        assert torch.allclose(
            new_demeaned_relevant_logits,
            demeaned_relevant_logits + expected_logit_difference,
            atol=logit_atol,
            rtol=logit_rtol,
        )

    for error_node_layer in range(error_vectors.size(0)):
        for error_node_pos in range(pos_start, error_vectors.size(1)):
            error_node_index = error_node_layer * error_vectors.size(1) + error_node_pos
            expected_effects = adjacency_matrix[:, total_active_features + error_node_index]

            def error_intervention(feature_output_loc, error_node_layer, error_node_pos):
                activations = feature_output_loc.output
                steering_vector = torch.zeros_like(activations)
                steering_vector[:, error_node_pos] += error_vectors[
                    error_node_layer, error_node_pos
                ]
                feature_output_loc.output = activations + steering_vector

            intervention = partial(
                error_intervention,
                error_node_layer=error_node_layer,
                error_node_pos=error_node_pos,
            )
            verify_intervention(expected_effects, intervention, error_node_layer)

    total_error_nodes = error_vectors.size(0) * error_vectors.size(1)
    for token_pos in range(pos_start, token_vectors.size(0)):
        expected_effects = adjacency_matrix[
            :, total_active_features + total_error_nodes + token_pos
        ]

        def token_intervention(token_loc, token_pos):
            activations = token_loc.output
            steering_vector = torch.zeros_like(activations)
            steering_vector[:, token_pos] += token_vectors[token_pos]
            token_loc.output = activations + steering_vector

        intervention = partial(token_intervention, token_pos=token_pos)  # type: ignore
        verify_intervention(expected_effects, intervention, "embed")


def verify_feature_edges(
    model: NNSightReplacementModel,
    graph: Graph,
    n_samples: int = 100,
    act_atol=5e-4,
    act_rtol=1e-5,
    logit_atol=1e-5,
    logit_rtol=1e-3,
    apply_activation_function: bool = False,
):
    """Verify feature-to-feature and feature-to-logit edges via intervention.

    Args:
        apply_activation_function: If False (default), measure pre-activation changes.
            If True, measure post-activation changes (for RelP encoder vectors).
    """
    s = graph.input_tokens
    adjacency_matrix = graph.adjacency_matrix.to(device=model.device, dtype=model.dtype)  # type:ignore
    active_features = graph.active_features.to(device=model.device)  # type:ignore
    logit_tokens = graph.logit_tokens.to(device=model.device)  # type:ignore
    total_active_features = active_features.size(0)

    logits, activation_cache = model.get_activations(s, apply_activation_function=apply_activation_function)
    logits = logits.squeeze(0)
    # Post-activation cache for computing intervention values (delta must be in post-act space)
    _, post_activation_cache = model.get_activations(s, apply_activation_function=True)

    relevant_activations = activation_cache[
        active_features[:, 0], active_features[:, 1], active_features[:, 2]
    ]
    relevant_logits = logits[-1, logit_tokens]
    demeaned_relevant_logits = relevant_logits - logits[-1].mean()

    def verify_intervention(
        expected_effects, layer: int, pos: int, feature_idx: int, new_activation
    ):
        new_logits, new_activation_cache = model.feature_intervention(
            s,
            [(layer, pos, feature_idx, new_activation)],
            constrained_layers=range(model.cfg.n_layers),  # type: ignore
            apply_activation_function=apply_activation_function,
        )
        new_logits = new_logits.squeeze(0)

        new_relevant_activations = new_activation_cache[  # type:ignore
            active_features[:, 0], active_features[:, 1], active_features[:, 2]
        ]  # type:ignore
        new_relevant_logits = new_logits[-1, logit_tokens]
        new_demeaned_relevant_logits = new_relevant_logits - new_logits[-1].mean()

        expected_activation_difference = expected_effects[:total_active_features]
        expected_logit_difference = expected_effects[-len(logit_tokens) :]

        assert torch.allclose(
            new_relevant_activations,
            relevant_activations + expected_activation_difference,
            atol=act_atol,
            rtol=act_rtol,
        )
        assert torch.allclose(
            new_demeaned_relevant_logits,
            demeaned_relevant_logits + expected_logit_difference,
            atol=logit_atol,
            rtol=logit_rtol,
        )

    random_order = torch.randperm(active_features.size(0))
    chosen_nodes = random_order[:n_samples]
    for chosen_node in tqdm(chosen_nodes):
        layer, pos, feature_idx = active_features[chosen_node]
        old_activation = post_activation_cache[layer, pos, feature_idx]
        new_activation = old_activation * 2
        expected_effects = adjacency_matrix[:, chosen_node]
        verify_intervention(expected_effects, layer, pos, feature_idx, new_activation)  # type: ignore


def load_dummy_gemma_model(cfg: AutoConfig) -> NNSightReplacementModel:
    model = ReplacementModel.from_config(cfg, MLPTranscoderConfig(activation_threshold=None), backend="nnsight")

    for _, param in model.named_parameters():
        nn.init.uniform_(param, a=-1, b=1)

    model.tokenizer.pad_token = model.tokenizer.eos_token  # type:ignore

    assert isinstance(model, NNSightReplacementModel)
    return model


def test_small_gemma_model():
    s = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])
    gemma_small_config = gemma_2_config
    gemma_small_config.num_hidden_layers = 2
    gemma_small_config.hidden_size = 8
    gemma_small_config.intermediate_size = 16
    gemma_small_config.head_dim = 4
    gemma_small_config.vocab_size = 16
    gemma_small_config.num_attention_heads = 2
    gemma_small_config.num_key_value_heads = 2
    gemma_small_config.final_logit_softcapping = None  # type: ignore
    gemma_small_config.torch_dtype = "float32"
    model = load_dummy_gemma_model(gemma_small_config)  # type: ignore

    # Save original property to restore later
    tokenizer_class = type(model.tokenizer)
    original_all_special_ids = tokenizer_class.all_special_ids  # type:ignore
    try:
        tokenizer_class.all_special_ids = property(lambda self: [0])  # type:ignore
        graph = attribute(s, model)

        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)
    finally:
        # Restore original property
        tokenizer_class.all_special_ids = original_all_special_ids  # type:ignore


def test_large_gemma_model():
    s = torch.tensor([0, 113, 24, 53, 27])
    gemma_large_cfg = gemma_2_config
    gemma_large_cfg.num_hidden_layers = 16
    gemma_large_cfg.hidden_size = 64
    gemma_large_cfg.intermediate_size = 128
    gemma_large_cfg.head_dim = 32
    gemma_large_cfg.vocab_size = 128
    gemma_large_cfg.num_attention_heads = 16
    gemma_large_cfg.num_key_value_heads = 16
    gemma_large_cfg.final_logit_softcapping = None  # type:ignore
    gemma_large_cfg.torch_dtype = "float64"
    model = load_dummy_gemma_model(gemma_large_cfg)  # type: ignore

    # Save original property to restore later
    tokenizer_class = type(model.tokenizer)
    original_all_special_ids = tokenizer_class.all_special_ids  # type:ignore
    try:
        tokenizer_class.all_special_ids = property(lambda self: [0])  # type:ignore
        graph = attribute(s, model)

        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)
    finally:
        # Restore original property
        tokenizer_class.all_special_ids = original_all_special_ids  # type:ignore


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gemma_2_2b():
    s = "The National Digital Analytics Group (ND"
    model = ReplacementModel.from_pretrained_and_transcoders(
        "google/gemma-2-2b", MLPTranscoderConfig(activation_threshold=0.25), backend="nnsight"
    )
    assert isinstance(model, NNSightReplacementModel)

    graph = attribute(s, model)

    print("Changing logit softcap to 0, as the logits will otherwise be off.")
    with model.zero_softcap():
        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)


def load_dummy_gemma_model_relp(cfg: AutoConfig, half_rule: bool | None = None) -> NNSightReplacementModel:
    model = ReplacementModel.from_config(
        cfg, MLPTranscoderConfig(activation_threshold=None, relp=True, half_rule=half_rule),
        backend="nnsight",
    )

    for _, param in model.named_parameters():
        nn.init.uniform_(param, a=-1, b=1)

    model.tokenizer.pad_token = model.tokenizer.eos_token  # type:ignore

    assert isinstance(model, NNSightReplacementModel)
    return model


def test_analytic_vs_autograd_encoder_vectors():
    """Verify analytic encoder rows match the per-feature autograd reference.

    Covers RelP (frozen derivative) and exact-Jacobian (jacobian_encoders
    without relp), both GELU tanh (Gemma) and SiLU (Llama), with and without
    half_rule, and with/without encoder bias.
    """
    from circuit_tracer.transcoder.activation_functions import GatedActivationFunction
    from circuit_tracer.transcoder.mlp_dummy_transcoder import MLPDummyTranscoder

    torch.manual_seed(123)

    d_model, d_mlp = 8, 16
    n_pos = 6

    W_up = torch.randn(d_mlp, d_model)
    W_gate = torch.randn(d_mlp, d_model)
    W_dec = torch.randn(d_mlp, d_model)
    input_acts = torch.randn(n_pos, d_model)

    configs = [
        ("gelu_tanh, no half_rule", nn.GELU(approximate='tanh'), True, False),
        ("gelu_tanh, half_rule", nn.GELU(approximate='tanh'), True, True),
        ("silu, no half_rule", nn.SiLU(), True, False),
        ("silu, half_rule", nn.SiLU(), True, True),
        # exact (non-frozen) Jacobian encoders, no relp
        ("gelu_tanh, exact", nn.GELU(approximate='tanh'), False, False),
        ("silu, exact", nn.SiLU(), False, False),
    ]

    for label, act_fn_module, relp, half_rule in configs:
        act_fn = GatedActivationFunction(W_gate, act_fn_module, relp=relp, half_rule=half_rule)
        transcoder = MLPDummyTranscoder(
            W_enc=W_up, b_enc=None, W_dec=W_dec,
            b_dec=None, activation_function=act_fn, layer_idx=0,
            jacobian_encoders=not relp,
        )

        sparse_acts, analytic_encoders = transcoder.encode_sparse(input_acts)
        pos_idx, feat_idx = sparse_acts.indices()

        if len(pos_idx) == 0:
            continue

        autograd_encoders = transcoder._compute_encoder_rows_autograd(
            input_acts, pos_idx, feat_idx
        )

        assert torch.allclose(analytic_encoders, autograd_encoders, atol=1e-5), (
            f"{label}: max diff = {(analytic_encoders - autograd_encoders).abs().max().item()}"
        )

    # Test with encoder bias (b_enc != None)
    b_enc = torch.randn(d_mlp)
    act_fn_bias = GatedActivationFunction(
        W_gate, nn.GELU(approximate='tanh'), relp=True, half_rule=True,
    )
    transcoder_bias = MLPDummyTranscoder(
        W_enc=W_up, b_enc=b_enc, W_dec=W_dec,
        b_dec=None, activation_function=act_fn_bias, layer_idx=0,
    )

    sparse_acts_b, analytic_b = transcoder_bias.encode_sparse(input_acts)
    pos_b, feat_b = sparse_acts_b.indices()
    if len(pos_b) > 0:
        autograd_b = transcoder_bias._compute_encoder_rows_autograd(
            input_acts, pos_b, feat_b
        )
        assert torch.allclose(analytic_b, autograd_b, atol=1e-5), (
            f"b_enc: max diff = {(analytic_b - autograd_b).abs().max().item()}"
        )

    # Test zero active features (high threshold → nothing active)
    act_fn_empty = GatedActivationFunction(
        W_gate, nn.GELU(approximate='tanh'), relp=True, half_rule=True,
    )
    transcoder_empty = MLPDummyTranscoder(
        W_enc=W_up, b_enc=None, W_dec=W_dec,
        b_dec=None, activation_function=act_fn_empty, layer_idx=0,
        activation_threshold=1e6,
    )
    sparse_empty, enc_empty = transcoder_empty.encode_sparse(input_acts)
    assert enc_empty.shape == (0, d_model)

    # Without relp and without jacobian_encoders, encoder vectors are just the
    # W_enc (= up_proj) rows
    act_fn_plain = GatedActivationFunction(W_gate, nn.GELU(approximate='tanh'), relp=False)
    transcoder_plain = MLPDummyTranscoder(
        W_enc=W_up, b_enc=None, W_dec=W_dec,
        b_dec=None, activation_function=act_fn_plain, layer_idx=0,
    )
    sparse_plain, plain_encoders = transcoder_plain.encode_sparse(input_acts)
    _, feat_plain = sparse_plain.indices()
    assert torch.equal(plain_encoders, W_up[feat_plain])


def test_relp_attribution():
    """End-to-end test: RelP attribution produces a valid graph.

    With relp=True and half_rule=False, the adjacency matrix gives the frozen
    Jacobian inner products. These predict post-activation changes as a linear
    approximation (exact for pre-activations, approximate for post-activations
    due to gate nonlinearity).

    We verify the graph is structurally valid and the pre-activation edges
    (token/error → feature) still work, since those only involve the linear
    up_proj pathway.
    """
    torch.manual_seed(42)
    s = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])
    gemma_small_config = gemma_2_config
    gemma_small_config.num_hidden_layers = 2
    gemma_small_config.hidden_size = 8
    gemma_small_config.intermediate_size = 16
    gemma_small_config.head_dim = 4
    gemma_small_config.vocab_size = 16
    gemma_small_config.num_attention_heads = 2
    gemma_small_config.num_key_value_heads = 2
    gemma_small_config.final_logit_softcapping = None  # type: ignore
    gemma_small_config.torch_dtype = "float32"
    model = load_dummy_gemma_model_relp(gemma_small_config, half_rule=False)  # type: ignore

    tokenizer_class = type(model.tokenizer)
    original_all_special_ids = tokenizer_class.all_special_ids  # type:ignore
    try:
        tokenizer_class.all_special_ids = property(lambda self: [0])  # type:ignore
        graph = attribute(s, model)

        # Graph should have the expected structure
        assert graph.adjacency_matrix is not None
        assert graph.active_features.size(0) > 0
        assert graph.logit_tokens.size(0) > 0

    finally:
        tokenizer_class.all_special_ids = original_all_special_ids  # type:ignore


if __name__ == "__main__":
    torch.manual_seed(42)
    test_small_gemma_model()
    test_large_gemma_model()
    test_gemma_2_2b()
