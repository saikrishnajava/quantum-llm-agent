"""
Quantum Encoding Strategies
============================
Implements amplitude, angle, and basis encodings for mapping classical
data into quantum states -- the fundamental data-ingress layer for
every quantum component in the hybrid transformer.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pennylane as qml


def _pad_to_power_of_two(vec: np.ndarray) -> np.ndarray:
    """Pad *vec* with zeros so its length is a power of two."""
    n = len(vec)
    target = 1 << math.ceil(math.log2(max(n, 2)))
    if n == target:
        return vec
    padded = np.zeros(target, dtype=vec.dtype)
    padded[:n] = vec
    return padded


def _normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """L2-normalize *vec* (required for amplitude encoding)."""
    norm = np.linalg.norm(vec)
    if norm < eps:
        out = np.zeros_like(vec)
        out[0] = 1.0  # valid quantum state
        return out
    return vec / norm


# ------------------------------------------------------------------
# Amplitude Encoding
# ------------------------------------------------------------------

class AmplitudeEncoder:
    """
    Encode a classical vector into the amplitudes of a quantum state.

    Given a real-valued vector of length *d*, this encoder:
      1. L2-normalises it,
      2. pads to the next power of two,
      3. uses ``qml.AmplitudeEmbedding`` to prepare |ψ⟩.

    Qubit cost: ceil(log2(d))
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state_dim = 2**n_qubits

    def encode(self, wires, features: np.ndarray) -> None:
        """Apply amplitude encoding inside an active QNode."""
        features = np.asarray(features, dtype=np.float64)
        features = _pad_to_power_of_two(features)[:self.state_dim]
        features = _normalize(features)
        qml.AmplitudeEmbedding(features, wires=wires, normalize=False)

    @staticmethod
    def qubits_required(dim: int) -> int:
        return math.ceil(math.log2(max(dim, 2)))


# ------------------------------------------------------------------
# Angle Encoding
# ------------------------------------------------------------------

class AngleEncoder:
    """
    Encode each feature as a rotation angle on a dedicated qubit.

    Qubit cost: equal to the feature dimension.
    """

    def __init__(self, n_qubits: int, rotation: str = "Y"):
        self.n_qubits = n_qubits
        if rotation.upper() == "Y":
            self._gate = qml.RY
        elif rotation.upper() == "X":
            self._gate = qml.RX
        elif rotation.upper() == "Z":
            self._gate = qml.RZ
        else:
            raise ValueError(f"Unsupported rotation: {rotation}")

    def encode(self, wires, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float64)
        for i, w in enumerate(wires):
            angle = features[i] if i < len(features) else 0.0
            self._gate(angle, wires=w)


# ------------------------------------------------------------------
# Basis Encoding
# ------------------------------------------------------------------

class BasisEncoder:
    """
    Encode an integer as a computational-basis state |x⟩ via PauliX flips.

    Qubit cost: number of bits required to represent the integer.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def encode(self, wires, value: int) -> None:
        bits = format(int(value) % (2**self.n_qubits), f"0{self.n_qubits}b")
        for i, bit in enumerate(bits):
            if bit == "1":
                qml.PauliX(wires=wires[i])


# ------------------------------------------------------------------
# Variational Encoding (data re-uploading)
# ------------------------------------------------------------------

class VariationalEncoder:
    """
    Data re-uploading encoder: interleaves data-encoding rotations
    with trainable variational layers.

    This is the most expressive encoder and typically used when the
    classical dimension exceeds the available qubit count.
    """

    def __init__(self, n_qubits: int, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # 3 params per qubit per layer (RY, RZ data; RY trainable)
        self.n_params = n_qubits * n_layers * 3

    def encode(
        self,
        wires,
        features: np.ndarray,
        params: np.ndarray,
    ) -> None:
        """
        Apply variational encoding inside an active QNode.

        Parameters
        ----------
        wires : sequence of ints
            Qubit indices.
        features : (n_features,) array
            Classical data to encode (used cyclically if shorter than qubits).
        params : (n_params,) array
            Trainable variational parameters.
        """
        features = np.asarray(features, dtype=np.float64)
        idx = 0
        for layer in range(self.n_layers):
            # Data-encoding rotations
            for i, w in enumerate(wires):
                f = features[i % len(features)]
                qml.RY(f, wires=w)
                qml.RZ(f, wires=w)

            # Entangling layer
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])

            # Trainable rotations
            for i, w in enumerate(wires):
                qml.RY(params[idx], wires=w)
                idx += 1
                qml.RZ(params[idx], wires=w)
                idx += 1
                qml.RX(params[idx], wires=w)
                idx += 1
