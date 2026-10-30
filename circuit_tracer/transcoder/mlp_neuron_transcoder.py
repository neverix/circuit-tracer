"""MLP neurons as transcoder features.

MLPNeuronTranscoder exposes one layer of a model's own MLP through the transcoder
interface, so attribution graphs can use MLP neurons as nodes without a trained
dictionary. The weights are the model's own tensors (not copies), and the
reconstruction is exact: without thresholding, decode(encode(x)) is the MLP output.
"""

from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from circuit_tracer.transcoder.activation_functions import GatedActivationFunction
from circuit_tracer.transcoder.single_layer_transcoder import TranscoderSet


Encoders = Literal["up", "jacobian", "relp"]


class MLPNeuronConfig(NamedTuple):
    """Use the model's MLP neurons as features instead of a trained transcoder.

    Pass in place of a TranscoderSet to ReplacementModel.from_pretrained_and_transcoders
    or ReplacementModel.from_config.

    activation_topk and activation_threshold select the graph-node candidates among
    the active neurons. Both are required; None disables either. Prefer
    activation_topk: activation scales differ between models and between layers, so
    one absolute threshold keeps a different fraction of the neurons in each.
    """

    # Keep the top-k |activations| per position.
    activation_topk: int | None
    # Drop |activations| <= threshold (before activation_topk). Not recommended
    # over activation_topk.
    activation_threshold: float | None
    scan_name: str | None = None
    # The encoder vector of each active neuron of a gated MLP:
    #   "up": the up-projection row;
    #   "jacobian": the exact local Jacobian row of the activation;
    #   "relp": the Jacobian row with the activation derivative frozen at the gate
    #     value, also used in the activation's backward
    #     (RelP, https://arxiv.org/abs/2508.21258).
    encoders: Encoders = "up"
    # Halve the Jacobian rows (corrects bilinear double-counting).
    # None: on for "relp", off otherwise.
    half_rule: bool | None = None


class MLPWeights(NamedTuple):
    """One layer's MLP weights, oriented as (d_mlp, d_model)."""

    W_up: torch.Tensor
    b_up: torch.Tensor | None
    W_down: torch.Tensor
    b_down: torch.Tensor | None
    # None for an ungated MLP: act_fn(W_up x + b_up)
    W_gate: torch.Tensor | None
    act_fn: Callable[[torch.Tensor], torch.Tensor]


class MLPNeuronTranscoder(nn.Module):
    """One layer's MLP neurons in the transcoder interface.

    Neuron s computes act(W_gate[s] x) * (W_up[s] x + b_up[s]) for a gated MLP and
    act(W_up[s] x + b_up[s]) for an ungated one. W_enc is W_up and W_dec is W_down.

    Pre-activations (encode with apply_activation_function=False) are the up
    projection. For a gated MLP the gate reads the MLP input, which encode() records
    on the GatedActivationFunction; activation_function(pre_acts) is only valid for
    the input of the latest encode() call on this layer.
    """

    W_skip = None  # read by TranscoderSet

    def __init__(
        self,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor | None,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor | None,
        activation_function: Callable[[torch.Tensor], torch.Tensor],
        layer_idx: int,
        activation_threshold: float | None = None,
        activation_topk: int | None = None,
        encoders: Encoders = "up",
    ):
        super().__init__()
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.W_dec = W_dec
        self.b_dec = b_dec
        self.activation_function = activation_function
        self.layer_idx = layer_idx
        self.d_model = W_enc.size(1)
        self.d_transcoder = W_enc.size(0)
        self.activation_threshold = activation_threshold
        self.activation_topk = activation_topk
        if encoders != "up" and not (
            isinstance(activation_function, GatedActivationFunction)
            and activation_function.relp == (encoders == "relp")
        ):
            raise ValueError(
                f'encoders="{encoders}" needs a GatedActivationFunction with relp={encoders == "relp"}'
            )
        self.encoders = encoders

    def _threshold_activations(self, acts: torch.Tensor) -> torch.Tensor:
        """Zero the activations that are not graph-node candidates.

        activation_threshold zeroes |acts| <= threshold. activation_topk keeps
        |acts| >= the k-th largest |acts| per position (ties kept), among the
        values that survive the threshold.
        """
        if self.activation_threshold is not None:
            acts = acts * (acts.abs() > self.activation_threshold)
        if self.activation_topk is not None and self.activation_topk < acts.shape[-1]:
            kth_largest = acts.abs().topk(self.activation_topk, dim=-1).values[..., -1:]
            acts = acts * (acts.abs() >= kth_largest)
        return acts

    def encode(self, input_acts: torch.Tensor, apply_activation_function: bool = True):
        input_acts = input_acts.to(self.W_enc.dtype)
        pre_acts = F.linear(input_acts, self.W_enc, self.b_enc)
        if isinstance(self.activation_function, GatedActivationFunction):
            self.activation_function.set_original_inputs(input_acts)
        if not apply_activation_function:
            return pre_acts
        return self._threshold_activations(self.activation_function(pre_acts))

    def decode(self, acts: torch.Tensor, input_acts: torch.Tensor | None = None):
        reconstruction = acts @ self.W_dec
        if self.b_dec is not None:
            reconstruction = reconstruction + self.b_dec
        return reconstruction

    def compute_skip(self, input_acts):
        raise ValueError("MLPNeuronTranscoder has no skip connection")

    def _get_decoder_vectors(self, feat_ids=None):
        return self.W_dec if feat_ids is None else self.W_dec[feat_ids]

    def encode_sparse(self, input_acts: torch.Tensor, zero_positions: slice = slice(0, 1)):
        """Sparse activations and the encoder vector of each active (position, neuron).

        The encoder vector is the W_enc row with encoders="up", and otherwise a
        Jacobian row; see MLPNeuronConfig and GatedActivationFunction.encoder_rows.
        """
        input_acts = input_acts.to(self.W_enc.dtype)
        acts = self.encode(input_acts)
        acts[zero_positions] = 0
        sparse_acts = acts.to_sparse()
        pos_idx, feat_idx = sparse_acts.indices()

        if self.encoders == "up":
            encoders = self.W_enc[feat_idx]
        else:
            act_fn: GatedActivationFunction = self.activation_function  # type: ignore
            encoders = act_fn.encoder_rows(input_acts, pos_idx, feat_idx, self.W_enc, self.b_enc)
        return sparse_acts, encoders

    def decode_sparse(self, sparse_acts: torch.Tensor, input_acts: torch.Tensor | None = None):
        """Reconstruction and the activation-scaled decoder vector of each active neuron."""
        pos_idx, feat_idx = sparse_acts.indices()
        scaled_decoders = self.W_dec[feat_idx] * sparse_acts.values()[:, None]
        reconstruction = torch.zeros(
            sparse_acts.shape[0],
            self.d_model,
            device=scaled_decoders.device,
            dtype=scaled_decoders.dtype,
        )
        reconstruction.index_add_(0, pos_idx, scaled_decoders)
        if self.b_dec is not None:
            reconstruction = reconstruction + self.b_dec
        return reconstruction, scaled_decoders

    def to_safetensors(self, save_path: str):
        raise NotImplementedError(
            "MLPNeuronTranscoder holds the model's own weights; save the model"
        )


class MLPNeuronTranscoderSet(TranscoderSet):
    """MLPNeuronTranscoders for every layer, in the TranscoderSet interface."""

    def __init__(
        self,
        transcoders: dict[int, MLPNeuronTranscoder],
        feature_input_hook: str,
        feature_output_hook: str,
        scan_name: str | list[str] | None = None,
    ):
        super().__init__(transcoders, feature_input_hook, feature_output_hook, scan_name)  # type: ignore[arg-type]

    def apply_activation_function(self, layer_id, features):
        transcoder: MLPNeuronTranscoder = self.transcoders[layer_id]  # type: ignore
        return transcoder._threshold_activations(transcoder.activation_function(features))


def _detach(t: torch.Tensor | None) -> torch.Tensor | None:
    return None if t is None else t.detach()


def build_mlp_transcoder_set(
    mlps: Sequence[MLPWeights], config: MLPNeuronConfig
) -> MLPNeuronTranscoderSet:
    """Wrap each layer's MLP weights in an MLPNeuronTranscoder.

    The features read the MLP input (mlp.hook_in) and write the MLP output
    (mlp.hook_out). The weights are detached: they share storage with the model's
    parameters but carry no autograd history (a transposed view taken before the
    model is frozen would otherwise make every encode build a graph).
    """
    transcoders = {}
    for layer, mlp in enumerate(mlps):
        mlp = mlp._replace(
            W_up=mlp.W_up.detach(),
            b_up=_detach(mlp.b_up),
            W_down=mlp.W_down.detach(),
            b_down=_detach(mlp.b_down),
            W_gate=_detach(mlp.W_gate),
        )
        if mlp.W_gate is not None:
            act_fn = GatedActivationFunction(
                mlp.W_gate, mlp.act_fn, relp=config.encoders == "relp", half_rule=config.half_rule
            )
        elif config.encoders != "up":
            raise NotImplementedError(
                f'encoders="{config.encoders}" is only implemented for gated MLPs'
            )
        else:
            act_fn = mlp.act_fn
        transcoders[layer] = MLPNeuronTranscoder(
            mlp.W_up,
            mlp.b_up,
            mlp.W_down,
            mlp.b_down,
            act_fn,
            layer,
            activation_threshold=config.activation_threshold,
            activation_topk=config.activation_topk,
            encoders=config.encoders,
        )
    return MLPNeuronTranscoderSet(
        transcoders,
        feature_input_hook="mlp.hook_in",
        feature_output_hook="mlp.hook_out",
        scan_name=config.scan_name,
    )
