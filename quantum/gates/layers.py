"""
Custom Quantum Gate Layers
===========================
Reusable variational and entangling gate layers used throughout the
hybrid transformer architecture.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


# ------------------------------------------------------------------
# Variational Layer
# ------------------------------------------------------------------

def variational_layer(
    params: np.ndarray,
    wires: list[int],
    rotation_gates: tuple = (qml.RY, qml.RZ),
) -> None:
    """
    Apply a single variational layer: per-qubit rotations followed by
    a linear chain of CNOTs.

    Parameters
    ----------
    params : shape (n_wires * len(rotation_gates),)
        Interleaved: [gate1(w0), gate2(w0), gate1(w1), gate2(w1), ...]
    wires  : qubit indices
    rotation_gates : sequence of PennyLane rotation constructors
    """
    n_wires = len(wires)
    idx = 0
    for w in wires:
        for gate in rotation_gates:
            gate(params[idx], wires=w)
            idx += 1

    # Linear entanglement
    for i in range(n_wires - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def strongly_entangling_layer(params: np.ndarray, wires: list[int]) -> None:
    """
    A single strongly-entangling layer: RX-RY-RZ rotations on every
    qubit followed by entangling CNOTs with varying offsets.

    Parameters
    ----------
    params : shape (n_wires * 3,)
    """
    n_wires = len(wires)
    idx = 0
    for w in wires:
        qml.RX(params[idx], wires=w)
        qml.RY(params[idx + 1], wires=w)
        qml.RZ(params[idx + 2], wires=w)
        idx += 3

    # Entangle with offset = 1
    for i in range(n_wires - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    if n_wires > 2:
        qml.CNOT(wires=[wires[-1], wires[0]])


# ------------------------------------------------------------------
# Quantum Attention Gate Pattern
# ------------------------------------------------------------------

def cross_attention_entanglement(
    query_wires: list[int],
    key_wires: list[int],
    params: np.ndarray,
) -> None:
    """
    Entangle query and key registers to compute quantum attention scores.

    Applies a CNOT from each query qubit to the corresponding key qubit,
    followed by parameterized RY rotations on the key register.
    """
    for i, (qw, kw) in enumerate(zip(query_wires, key_wires)):
        qml.CNOT(wires=[qw, kw])
        qml.RY(params[i], wires=kw)


# ------------------------------------------------------------------
# Circular Entanglement
# ------------------------------------------------------------------

def circular_entanglement(wires: list[int]) -> None:
    """CNOT ring: 0→1→2→…→(n-1)→0."""
    n = len(wires)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


# ------------------------------------------------------------------
# Measurement Helpers
# ------------------------------------------------------------------

def pauli_z_expectations(wires: list[int]):
    """Return PauliZ expectation values for all *wires*."""
    return [qml.expval(qml.PauliZ(w)) for w in wires]


def pauli_y_expectations(wires: list[int]):
    return [qml.expval(qml.PauliY(w)) for w in wires]


def probability_measurement(wires: list[int]):
    return qml.probs(wires=wires)
