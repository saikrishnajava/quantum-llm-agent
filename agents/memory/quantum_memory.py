"""
Quantum Memory Network  (NumPy)
=================================
"""

from __future__ import annotations

from collections import deque

import numpy as np

from classical.nn import Module, Linear
from agents.reasoning.quantum_reasoning import QuantumPatternMatcher


class QuantumMemoryNetwork(Module):
    def __init__(self, d_model: int = 128, memory_size: int = 64, n_qubits: int = 6):
        self.d_model = d_model
        self.memory_size = memory_size
        self.reg_dim = 2 ** (n_qubits // 2)
        self.encoder = Linear(d_model, self.reg_dim)
        self.pattern_matcher = QuantumPatternMatcher(n_qubits, n_layers=1)
        self._buffer: deque[np.ndarray] = deque(maxlen=memory_size)

    def store(self, experience: np.ndarray) -> None:
        encoded = self.encoder(experience)
        self._buffer.append(np.array(encoded).copy())

    def recall(self, query: np.ndarray, top_k: int = 5) -> list[np.ndarray]:
        if len(self._buffer) == 0:
            return []
        encoded_query = self.encoder(query)
        scores = []
        for mem in self._buffer:
            sim = self.pattern_matcher(encoded_query, mem)
            scores.append(float(sim.mean()))
        ranked = sorted(zip(scores, self._buffer), key=lambda x: x[0], reverse=True)
        return [mem for _, mem in ranked[:top_k]]

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def size(self) -> int:
        return len(self._buffer)
