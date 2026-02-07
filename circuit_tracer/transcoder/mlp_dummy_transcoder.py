from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from circuit_tracer.utils import get_default_device
from .activation_functions import GatedActivationFunction


class MLPDummyTranscoder(nn.Module):
    """
    A per-layer transcoder (PLT) that replaces MLP computation with interpretable features.

    Per-layer transcoders decompose the output of a single MLP layer into sparsely active
    features that often correspond to interpretable concepts. Unlike cross-layer transcoders,
    each PLT operates independently on its assigned layer, which can result in longer paths
    through attribution graphs when features amplify across multiple layers.

    Attributes:
        d_model: Dimension of the transformer's residual stream
        d_transcoder: Number of learned features (typically >> d_model for superposition)
        layer_idx: Which transformer layer this transcoder replaces
        W_enc: Encoder weights mapping residual stream to feature space
        W_dec: Decoder weights mapping features back to residual stream
        b_enc: Encoder bias terms
        b_dec: Decoder bias terms (reconstruction baseline)
        W_skip: Optional skip connection weights (https://arxiv.org/abs/2501.18823)
        activation_function: Sparsity-inducing nonlinearity (e.g., ReLU, JumpReLU)
    """

    def __init__(
        self,
        W_enc: torch.Tensor, 
        b_enc: torch.Tensor | None,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor | None,
        activation_function: nn.Module,
        layer_idx: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        if device is None:
            device = get_default_device()

        self.W_enc = W_enc
        self.b_enc = b_enc
        self.W_dec = W_dec
        self.b_dec = b_dec
        self.d_model = self.W_enc.size(1)
        self.d_transcoder = self.W_enc.size(0)
        self.layer_idx = layer_idx
        self.activation_function = activation_function

    @property
    def device(self):
        """Get the device of the module's parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self):
        """Get the dtype of the module's parameters."""
        return self.W_enc.dtype

    def _get_decoder_vectors(self, feat_ids=None):
        to_read = feat_ids if feat_ids is not None else np.s_[:]
        return self.W_dec[to_read].to(self.dtype)

    def encode(self, input_acts, apply_activation_function: bool = True):
        W_enc = self.W_enc
        pre_acts = F.linear(input_acts.to(W_enc.dtype), W_enc, self.b_enc)
        if isinstance(self.activation_function, GatedActivationFunction):
            self.activation_function.set_original_inputs(input_acts)

        if not apply_activation_function:
            return pre_acts
        return self.activation_function(pre_acts)

    def decode(self, acts):
        return F.linear(acts, self.W_dec.T, self.b_dec)

    def compute_skip(self, input_acts):
        raise ValueError("MLPDummyTranscoder has no skip connection")

    def forward(self, input_acts):
        transcoder_acts = self.encode(input_acts)
        decoded = self.decode(transcoder_acts)
        decoded = decoded.detach()
        decoded.requires_grad = True

        return decoded

    def encode_sparse(self, input_acts, zero_positions: slice = slice(0, 1)):
        """Encode and return sparse activations with active encoder vectors.

        Args:
            input_acts: Input activations
            zero_positions: slice representing the positions to zero out

        Returns:
            sparse_acts: Sparse tensor of activations
            active_encoders: Encoder vectors for active features only
        """
        W_enc = self.W_enc
        pre_acts = F.linear(input_acts.to(W_enc.dtype), W_enc, self.b_enc)
        acts = self.activation_function(input_acts, pre_acts)

        acts[zero_positions] = 0

        sparse_acts = acts.to_sparse()
        _, feat_idx = sparse_acts.indices()
        active_encoders = W_enc[feat_idx]

        return sparse_acts, active_encoders

    def decode_sparse(self, sparse_acts):
        """Decode sparse activations and return reconstruction with scaled decoder vectors.

        Returns:
            reconstruction: Decoded output
            scaled_decoders: Decoder vectors scaled by activation values
        """
        pos_idx, feat_idx = sparse_acts.indices()
        values = sparse_acts.values()

        # Get decoder vectors for active features only
        W_dec = self._get_decoder_vectors(feat_idx.cpu())
        scaled_decoders = W_dec * values[:, None]

        # Reconstruct using index_add
        n_pos = sparse_acts.shape[0]
        reconstruction = torch.zeros(
            n_pos, self.d_model, device=sparse_acts.device, dtype=sparse_acts.dtype
        )
        reconstruction = reconstruction.index_add_(0, pos_idx, scaled_decoders)
        if self.b_dec is not None:
            reconstruction = reconstruction + self.b_dec

        return reconstruction, scaled_decoders

    def to_safetensors(self, save_path: str):
        raise NotImplementedError("Can't convert MLPDummyTranscoder to safetensors")


class MLPDummyTranscoderSet(nn.Module):
    """
    A collection of per-layer transcoders that enable construction of a replacement model.

    TranscoderSet manages the collection of MLPDummyTranscoders needed for this substitution,
    where each transcoder replaces the MLP computation at its corresponding layer.

    Attributes:
        transcoders: ModuleList of MLPDummyTranscoder instances, one per layer
        n_layers: Total number of layers covered
        d_transcoder: Common feature dimension across all transcoders
        feature_input_hook: Hook point where features read from (e.g., "hook_resid_mid")
        feature_output_hook: Hook point where features write to (e.g., "hook_mlp_out")
        scan: Optional identifier to identify corresponding feature visualization
        skip_connection: Whether transcoders include learned skip connections
    """
    transcoders: nn.ModuleList
    n_layers: int
    d_transcoder: int

    def __init__(
        self,
        transcoders: dict[int, MLPDummyTranscoder],
        feature_input_hook: str,
        feature_output_hook: str,
        scan: str | list[str] | None = None,
    ):
        super().__init__()
        # Validate that we have continuous layers from 0 to max
        assert set(transcoders.keys()) == set(range(max(transcoders.keys()) + 1)), (
            f"Each layer should have a transcoder, but got transcoders for layers "
            f"{set(transcoders.keys())}"
        )

        self.transcoders = nn.ModuleList([transcoders[i] for i in range(len(transcoders))])
        self.n_layers = len(self.transcoders)
        self.d_transcoder = self.transcoders[0].d_transcoder  #type: ignore

        # Verify all transcoders have the same d_transcoder
        for transcoder in self.transcoders:
            assert transcoder.d_transcoder == self.d_transcoder, (
                f"All transcoders must have the same d_transcoder, but got "
                f"{transcoder.d_transcoder} != {self.d_transcoder}"
            )

        # Store hook configuration
        self.feature_input_hook = feature_input_hook
        self.feature_output_hook = feature_output_hook
        self.scan = scan
        self.skip_connection = False

    def __len__(self):
        return self.n_layers

    def __getitem__(self, idx: int) -> MLPDummyTranscoder:
        return self.transcoders[idx]  # type: ignore

    def __iter__(self) -> Iterator[MLPDummyTranscoder]:
        return iter(self.transcoders)  # type: ignore

    def apply_activation_function(self, layer_id, features):
        return self.transcoders[layer_id].activation_function(features)  # type: ignore

    def compute_skip(self, layer_id: int, inputs):
        return self.transcoders[layer_id].compute_skip(inputs)  # type: ignore

    def encode(self, input_acts):
        return torch.stack(
            [transcoder.encode(input_acts[i]) for i, transcoder in enumerate(self.transcoders)],  # type: ignore
            dim=0,
        )

    def _get_decoder_vectors(self, layer_id, features):
        return self.transcoders[layer_id]._get_decoder_vectors(features)  # type: ignore

    def select_decoder_vectors(self, features):
        if not features.is_sparse:
            features = features.to_sparse()

        all_layer_idx, all_pos_idx, all_feat_idx = features.indices()
        all_activations = features.values()
        all_scaled_decoder_vectors = []
        for unique_layer in all_layer_idx.unique():
            layer_mask = all_layer_idx == unique_layer
            feat_idx = all_feat_idx[layer_mask]
            activations = all_activations[layer_mask]

            decoder_vectors = self._get_decoder_vectors(unique_layer.item(), feat_idx)

            # Multiply each activation by its corresponding decoder vector
            scaled_decoder_vectors = activations.unsqueeze(-1) * decoder_vectors
            all_scaled_decoder_vectors.append(scaled_decoder_vectors)

        all_scaled_decoder_vectors = torch.cat(all_scaled_decoder_vectors)
        encoder_mapping = torch.arange(features._nnz(), device=features.device)

        return (
            all_pos_idx,
            all_layer_idx,
            all_feat_idx,
            all_scaled_decoder_vectors,
            encoder_mapping,
        )

    def decode(self, acts):
        return torch.stack(
            [transcoder.decode(acts[i]) for i, transcoder in enumerate(self.transcoders)],  # type: ignore
            dim=0,
        )

    def compute_attribution_components(
        self, mlp_inputs: torch.Tensor, zero_positions: slice = slice(0, 1)
    ) -> dict[str, torch.Tensor]:
        """Extract active features and their encoder/decoder vectors for attribution.

        Args:
            mlp_inputs: (n_layers, n_pos, d_model) tensor of MLP inputs
            zero_positions: (slice) slice indicating which positions to zero out

        Returns:
            Dict containing all components needed for AttributionContext:
                - activation_matrix: Sparse (n_layers, n_pos, d_transcoder) activations
                - reconstruction: (n_layers, n_pos, d_model) reconstructed outputs
                - encoder_vecs: Concatenated encoder vectors for active features
                - decoder_vecs: Concatenated decoder vectors (scaled by activations)
                - decoder_biases: Decoder biases for each layer
                - encoder_to_decoder_map: Mapping from encoder to decoder indices
                - decoder_locations: Layer and position indices for decoders
        """
        device = mlp_inputs.device

        reconstruction = torch.zeros_like(mlp_inputs)
        encoder_vectors = []
        decoder_vectors = []
        decoder_biases = []
        sparse_acts_list = []

        for layer, transcoder in enumerate(self.transcoders):
            sparse_acts, active_encoders = transcoder.encode_sparse(  # type: ignore
                mlp_inputs[layer], zero_positions=zero_positions
            )
            reconstruction[layer], active_decoders = transcoder.decode_sparse(sparse_acts)  # type: ignore
            encoder_vectors.append(active_encoders)
            decoder_vectors.append(active_decoders)
            decoder_biases.append(transcoder.b_dec.clone())  # type: ignore
            sparse_acts_list.append(sparse_acts)

        activation_matrix = torch.stack(sparse_acts_list).coalesce()
        encoder_to_decoder_map = torch.arange(activation_matrix._nnz(), device=device)

        return {
            "activation_matrix": activation_matrix,
            "reconstruction": reconstruction,
            "encoder_vecs": torch.cat(encoder_vectors, dim=0),
            "decoder_vecs": torch.cat(decoder_vectors, dim=0),
            "decoder_biases": torch.stack(decoder_biases),
            "encoder_to_decoder_map": encoder_to_decoder_map,
            "decoder_locations": activation_matrix.indices()[:2],
        }

    def encode_layer(self, x, layer_id, apply_activation_function=True):
        return self.transcoders[layer_id].encode(
            x, apply_activation_function=apply_activation_function
        )  # type: ignore

    def to_safetensors(self, save_dir: str):
        raise NotImplementedError("Can't convert MLPDummyTranscoder to safetensors")

