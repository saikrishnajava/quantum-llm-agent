"""
Hybrid Quantum-Classical Embedding Layer  (NumPy)
==================================================
"""

from __future__ import annotations

import math

import numpy as np
import autograd.numpy as anp
import pennylane.numpy as pnp

from classical.nn import Module, Embedding, Linear, LayerNorm, Dropout, Parameter
from quantum.circuits.core import QuantumFeatureMapCircuit, QuantumPositionalCircuit
from hybrid.interfaces.optimizations import PositionalEncodingCache


class HybridEmbeddingLayer(Module):
    """
    Fuses classical token embeddings with quantum feature enhancement
    and quantum positional encodings.  Output dimension = d_model.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_length: int = 2048,
        quantum_enhancement: bool = True,
        n_qubits: int = 6,
        pos_qubits: int | None = None,
    ):
        self.d_model = d_model
        self.quantum_enhancement = quantum_enhancement
        self.max_seq_length = max_seq_length

        if quantum_enhancement:
            self.classical_dim = d_model // 2
            self.quantum_dim = d_model - self.classical_dim
            self.token_embedding = Embedding(vocab_size, self.classical_dim)

            self.n_qubits = n_qubits
            self.quantum_feature_map = QuantumFeatureMapCircuit(n_qubits, n_layers=2)

            if pos_qubits is None:
                pos_qubits = min(10, max(4, math.ceil(math.log2(max(max_seq_length, 2)))))
            self.pos_qubits = pos_qubits
            self.quantum_positional = QuantumPositionalCircuit(pos_qubits, max_seq_length)
            self._pos_cache = PositionalEncodingCache(self.quantum_positional, max_positions=max_seq_length)

            self.feature_proj = Linear(n_qubits, self.quantum_dim // 2)
            self.pos_proj = Linear(pos_qubits, self.quantum_dim - self.quantum_dim // 2)
        else:
            self.token_embedding = Embedding(vocab_size, d_model)
            self._build_sinusoidal_pe(d_model, max_seq_length)

        self.layer_norm = LayerNorm(d_model)
        self.dropout = Dropout(0.1)

    def _build_sinusoidal_pe(self, d_model: int, max_len: int) -> None:
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float64).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float64) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[: d_model // 2])
        self._pe = pe[np.newaxis, :, :]  # (1, max_len, d_model)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        input_ids : (batch, seq_len) integer array

        Returns
        -------
        (batch, seq_len, d_model)
        """
        input_ids = np.asarray(input_ids, dtype=int)
        batch_size, seq_len = input_ids.shape
        classical_embeds = self.token_embedding(input_ids)

        if not self.quantum_enhancement:
            return self.dropout(self.layer_norm(classical_embeds + self._pe[:, :seq_len]))

        # Quantum feature enhancement
        quantum_features = self.quantum_feature_map(classical_embeds)
        quantum_features = self.feature_proj(quantum_features)

        # Quantum positional encoding (cached in eval mode)
        positions = np.tile(np.arange(seq_len), (batch_size, 1))
        if not getattr(self, '_training', True):
            quantum_pos = self._pos_cache.get(positions)
        else:
            quantum_pos = self.quantum_positional(positions)
        quantum_pos = self.pos_proj(quantum_pos)

        quantum_combined = anp.concatenate([quantum_features, quantum_pos], axis=-1)
        hybrid_embeds = anp.concatenate([classical_embeds, quantum_combined], axis=-1)

        return self.dropout(self.layer_norm(hybrid_embeds))
