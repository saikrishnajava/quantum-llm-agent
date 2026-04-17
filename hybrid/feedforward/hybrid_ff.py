"""
Hybrid Feed-Forward Network  (NumPy)
======================================
"""

from __future__ import annotations

import logging
import math

import numpy as np
import autograd.numpy as anp

from classical.nn import Module, Linear, GELU, Dropout
from quantum.circuits.core import QuantumActivationCircuit

logger = logging.getLogger(__name__)


class HybridFeedForward(Module):
    """d_model → d_ff (with optional quantum activation) → d_model."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        use_quantum_activation: bool = True,
        activation_qubits: int = 4,
        quantum_chunk_limit: int = 512,
    ):
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_ff = d_ff
        self.use_quantum_activation = use_quantum_activation
        self.quantum_chunk_limit = quantum_chunk_limit

        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)
        self.classical_activation = GELU()

        if use_quantum_activation:
            self.activation_qubits = activation_qubits
            self.quantum_activation = QuantumActivationCircuit(activation_qubits, n_layers=1)
            self.chunk_size = 2 ** activation_qubits

    def _apply_quantum_activation(self, x):
        original_shape = x.shape
        flat = x.reshape(-1)
        out_parts = []
        for start in range(0, len(flat), self.chunk_size):
            chunk = flat[start : start + self.chunk_size]
            q_out = self.quantum_activation(chunk)
            n_tiles = math.ceil(chunk.shape[0] / q_out.shape[0])
            expanded = anp.concatenate([q_out] * n_tiles)[:chunk.shape[0]]
            out_parts.append(expanded)
        result = anp.concatenate(out_parts)[:flat.size]
        return result.reshape(original_shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = self.linear1(x)

        if self.use_quantum_activation and hidden.size <= self.quantum_chunk_limit:
            try:
                activated = self._apply_quantum_activation(hidden)
            except Exception:
                logger.debug("Quantum activation failed, using GELU.")
                activated = self.classical_activation(hidden)
        else:
            activated = self.classical_activation(hidden)

        return self.linear2(self.dropout(activated))
