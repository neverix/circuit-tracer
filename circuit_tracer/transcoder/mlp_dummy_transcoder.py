"""MLP neurons as transcoder features.

MLPDummyTranscoder exposes one layer of a model's own MLP through the transcoder
interface, so attribution graphs can use MLP neurons as nodes without a trained
dictionary. The weights are the model's own tensors (not copies), and the
reconstruction is exact: without thresholding, decode(encode(x)) is the MLP output.
"""

from collections.abc import Callable, Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from circuit_tracer.transcoder.activation_functions import GatedActivationFunction
from circuit_tracer.transcoder.single_layer_transcoder import TranscoderSet


class MLPTranscoderConfig(NamedTuple):
    """Use the model's MLP neurons as features instead of a trained transcoder.

    Pass in place of a TranscoderSet to ReplacementModel.from_pretrained_and_transcoders
    or ReplacementModel.from_config.
    """

    # Neurons with |activation| <= threshold are not graph-node candidates.
    activation_threshold: float | None
    scan_name: str | None = None
    # RelP: frozen-derivative encoder vectors for gated MLPs.
    # https://arxiv.org/abs/2508.21258
    relp: bool = False
    # Halve the encoder vectors of gated MLPs (corrects bilinear double-counting).
    # None follows relp.
    half_rule: bool | None = None
    # Keep only the top-k |activations| per position as graph-node candidates.
    # Composes with activation_threshold (the threshold applies first).
    activation_topk: int | None = None
    # Use exact local-Jacobian encoder vectors for gated MLPs instead of the
    # up-projection rows. relp implies the frozen-derivative form.
    jacobian_encoders: bool = False


class MLPWeights(NamedTuple):
    """One layer's MLP weights, oriented as (d_mlp, d_model)."""

    W_up: torch.Tensor
    b_up: torch.Tensor | None
    W_down: torch.Tensor
    b_down: torch.Tensor | None
    # None for an ungated MLP: act_fn(W_up x + b_up)
    W_gate: torch.Tensor | None
    act_fn: Callable[[torch.Tensor], torch.Tensor]


class MLPDummyTranscoder(nn.Module):
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
        jacobian_encoders: bool = False,
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
        self.jacobian_encoders = jacobian_encoders

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
        raise ValueError("MLPDummyTranscoder has no skip connection")

    def _get_decoder_vectors(self, feat_ids=None):
        return self.W_dec if feat_ids is None else self.W_dec[feat_ids]

    def encode_sparse(self, input_acts: torch.Tensor, zero_positions: slice = slice(0, 1)):
        """Sparse activations and the encoder vector of each active (position, neuron).

        The encoder vector is the W_enc row by default. With jacobian_encoders it is
        the exact local Jacobian row of a gated neuron, and with relp the RelP
        frozen-derivative row (https://arxiv.org/abs/2508.21258); see
        GatedActivationFunction.encoder_rows.
        """
        input_acts = input_acts.to(self.W_enc.dtype)
        acts = self.encode(input_acts)
        acts[zero_positions] = 0
        sparse_acts = acts.to_sparse()
        pos_idx, feat_idx = sparse_acts.indices()

        act_fn = self.activation_function
        if isinstance(act_fn, GatedActivationFunction) and (act_fn.relp or self.jacobian_encoders):
            encoders = act_fn.encoder_rows(input_acts, pos_idx, feat_idx, self.W_enc, self.b_enc)
        else:
            encoders = self.W_enc[feat_idx]
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
            "MLPDummyTranscoder holds the model's own weights; save the model"
        )


class MLPDummyTranscoderSet(TranscoderSet):
    """MLPDummyTranscoders for every layer, in the TranscoderSet interface."""

    def __init__(
        self,
        transcoders: dict[int, MLPDummyTranscoder],
        feature_input_hook: str,
        feature_output_hook: str,
        scan_name: str | list[str] | None = None,
    ):
        super().__init__(transcoders, feature_input_hook, feature_output_hook, scan_name)  # type: ignore[arg-type]

    def apply_activation_function(self, layer_id, features):
        transcoder: MLPDummyTranscoder = self.transcoders[layer_id]  # type: ignore
        return transcoder._threshold_activations(transcoder.activation_function(features))


def _detach(t: torch.Tensor | None) -> torch.Tensor | None:
    return None if t is None else t.detach()


def build_mlp_transcoder_set(
    mlps: Sequence[MLPWeights], config: MLPTranscoderConfig
) -> MLPDummyTranscoderSet:
    """Wrap each layer's MLP weights in an MLPDummyTranscoder.

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
                mlp.W_gate, mlp.act_fn, relp=config.relp, half_rule=config.half_rule
            )
        elif config.relp or config.jacobian_encoders:
            raise NotImplementedError(
                "relp and jacobian_encoders are only implemented for gated MLPs"
            )
        else:
            act_fn = mlp.act_fn
        transcoders[layer] = MLPDummyTranscoder(
            mlp.W_up,
            mlp.b_up,
            mlp.W_down,
            mlp.b_down,
            act_fn,
            layer,
            activation_threshold=config.activation_threshold,
            activation_topk=config.activation_topk,
            jacobian_encoders=config.jacobian_encoders,
        )
    return MLPDummyTranscoderSet(
        transcoders,
        feature_input_hook="mlp.hook_in",
        feature_output_hook="mlp.hook_out",
        scan_name=config.scan_name,
    )
