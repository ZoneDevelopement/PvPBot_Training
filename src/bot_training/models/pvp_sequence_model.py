from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class PvPSequenceModel(nn.Module):
    """Transformer-based sequence model for PvP action prediction."""

    def __init__(
        self,
        input_feature_count: int,
        boolean_action_count: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        item_slot_count: int = 38,
        item_vocabulary_size: int = 2048,
        item_embedding_dim: int = 16,
    ) -> None:
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for sinusoidal positional encoding")
        if item_slot_count <= 0:
            raise ValueError("item_slot_count must be greater than zero")
        if item_vocabulary_size <= 0:
            raise ValueError("item_vocabulary_size must be greater than zero")
        if item_embedding_dim <= 0:
            raise ValueError("item_embedding_dim must be greater than zero")

        self.input_feature_count = input_feature_count
        self.boolean_action_count = boolean_action_count
        self.hidden_dim = hidden_dim
        self.slot_count = 9
        self.item_slot_count = item_slot_count
        self.item_vocabulary_size = item_vocabulary_size
        self.item_embedding_dim = item_embedding_dim

        self.input_projection = nn.Linear(input_feature_count, hidden_dim)
        self.item_embedding = nn.Embedding(item_vocabulary_size, item_embedding_dim)
        self.item_projection = nn.Linear(item_slot_count * item_embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_encoder = nn.TransformerEncoder(
            num_layers=num_layers,
            dims=hidden_dim,
            num_heads=num_heads,
        )
        self.binary_head = nn.Linear(hidden_dim, boolean_action_count)
        self.slot_head = nn.Linear(hidden_dim, self.slot_count)
        self.continuous_head = nn.Linear(hidden_dim, 2)

    def _build_positional_encoding(self, sequence_length: int) -> mx.array:
        position = mx.arange(sequence_length, dtype=mx.float32)[:, None]
        div_term = mx.exp(
            mx.arange(0, self.hidden_dim, 2, dtype=mx.float32)
            * (-math.log(10000.0) / self.hidden_dim)
        )

        sinusoid = position * div_term[None, :]
        encoding = mx.stack([mx.sin(sinusoid), mx.cos(sinusoid)], axis=-1)
        encoding = mx.reshape(encoding, (sequence_length, self.hidden_dim))
        return mx.expand_dims(encoding, axis=0)

    def __call__(
        self,
        inputs: mx.array,
        categorical_inputs: mx.array | None = None,
    ) -> dict[str, mx.array]:
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape (batch_size, window_size, input_feature_count)"
            )
        if inputs.shape[-1] != self.input_feature_count:
            raise ValueError(
                f"expected {self.input_feature_count} input features, got {inputs.shape[-1]}"
            )

        projected = self.input_projection(inputs)

        if categorical_inputs is not None:
            if categorical_inputs.ndim != 3:
                raise ValueError(
                    "categorical_inputs must have shape (batch_size, window_size, item_slot_count)"
                )
            if categorical_inputs.shape[0] != inputs.shape[0] or categorical_inputs.shape[1] != inputs.shape[1]:
                raise ValueError("categorical_inputs must match inputs on batch and window dimensions")
            if categorical_inputs.shape[-1] != self.item_slot_count:
                raise ValueError(
                    f"expected {self.item_slot_count} categorical slots, got {categorical_inputs.shape[-1]}"
                )

            categorical_tokens = mx.array(categorical_inputs, dtype=mx.int32)
            categorical_embedded = self.item_embedding(categorical_tokens)
            categorical_embedded = mx.reshape(
                categorical_embedded,
                (
                    categorical_embedded.shape[0],
                    categorical_embedded.shape[1],
                    self.item_slot_count * self.item_embedding_dim,
                ),
            )
            categorical_projected = self.item_projection(categorical_embedded)
            projected = projected + categorical_projected

        positional_encoding = self._build_positional_encoding(projected.shape[1])
        encoded_inputs = self.dropout(projected + positional_encoding)
        encoded = self.transformer_encoder(encoded_inputs, mask=None)

        final_timestep = self.dropout(encoded[:, -1, :])
        binary_probabilities = mx.sigmoid(self.binary_head(final_timestep))
        slot_probabilities = mx.softmax(self.slot_head(final_timestep), axis=-1)
        continuous_deltas = self.continuous_head(final_timestep)

        return {
            "binary_probabilities": binary_probabilities,
            "slot_probabilities": slot_probabilities,
            "continuous_deltas": continuous_deltas,
        }
