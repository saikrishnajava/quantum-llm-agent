"""
Core Quantum Circuits  (PennyLane + NumPy — no PyTorch)
========================================================
Parameterised quantum circuits used as building blocks for the hybrid
transformer: feature-map circuits, attention circuits, activation
circuits, and reasoning circuits.
"""

from __future__ import annotations

import math

import numpy as np
import autograd.numpy as anp
import pennylane as qml
import pennylane.numpy as pnp

from classical.nn import Module, Parameter
from quantum.simulator import (
    apply_gate, apply_cnot, amplitude_embed,
    pauli_z_expval, pauli_y_expval, _ry, _rz, _rx,
)


def _pad_to_dim(vec, target_dim):
    """Pad vec to target_dim using autograd-compatible ops."""
    vec_slice = vec[:target_dim]
    slice_len = vec_slice.shape[0]
    pad_len = max(0, target_dim - slice_len)
    if pad_len > 0:
        return anp.concatenate([vec_slice, anp.zeros(pad_len)])
    return vec_slice


# ------------------------------------------------------------------
# Feature Map Circuit
# ------------------------------------------------------------------

class QuantumFeatureMapCircuit(Module):
    """
    Variational quantum circuit that maps a classical feature vector
    to a set of expectation values, acting as a trainable non-linear
    feature transform.
    """

    def __init__(self, n_qubits: int, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.state_dim = 2 ** n_qubits

        self.device = qml.device("default.qubit", wires=n_qubits)
        self._build_circuit()

        n_params = n_qubits * n_layers * 2
        self.params = Parameter(np.random.randn(n_params) * 0.1)

    def _build_circuit(self):
        @qml.qnode(self.device, interface="autograd", diff_method="backprop")
        def circuit(features, params):
            qml.AmplitudeEmbedding(features, wires=range(self.n_qubits), normalize=True)
            idx = 0
            for _ in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[idx], wires=i)
                    qml.RZ(params[idx + 1], wires=i)
                    idx += 2
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit = circuit

    def forward(self, features):
        original_shape = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        outputs = []
        for vec in flat:
            padded = _pad_to_dim(vec, self.state_dim)
            state = amplitude_embed(padded, self.n_qubits)
            idx = 0
            for _ in range(self.n_layers):
                for i in range(self.n_qubits):
                    state = apply_gate(state, _ry(self.params[idx]), i, self.n_qubits)
                    state = apply_gate(state, _rz(self.params[idx + 1]), i, self.n_qubits)
                    idx += 2
                for i in range(self.n_qubits - 1):
                    state = apply_cnot(state, i, i + 1, self.n_qubits)
            outputs.append(anp.array([pauli_z_expval(state, i, self.n_qubits)
                                       for i in range(self.n_qubits)]))
        return anp.stack(outputs).reshape(*original_shape, self.n_qubits)


# ------------------------------------------------------------------
# Positional Encoding Circuit
# ------------------------------------------------------------------

class QuantumPositionalCircuit(Module):
    """
    Encodes token positions via angle-modulated rotations followed by
    variational rotations.
    """

    def __init__(self, n_qubits: int, max_positions: int = 2048):
        self.n_qubits = n_qubits
        self.max_positions = max_positions
        self.device = qml.device("default.qubit", wires=n_qubits)
        self._build_circuit()
        self.params = Parameter(np.random.randn(n_qubits * 2) * 0.5)

    def _build_circuit(self):
        @qml.qnode(self.device, interface="autograd", diff_method="backprop")
        def circuit(position_bits, params):
            for i in range(self.n_qubits):
                qml.RX(np.pi * position_bits[i], wires=i)
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
                qml.RZ(params[i + self.n_qubits], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliY(i)) for i in range(self.n_qubits)]

        self._circuit = circuit

    def _position_to_bits(self, position: int) -> np.ndarray:
        bits_str = format(position % (2 ** self.n_qubits), f"0{self.n_qubits}b")
        return np.array([float(b) for b in bits_str])

    def forward(self, positions):
        positions = np.asarray(positions, dtype=int)
        batch_size, seq_len = positions.shape
        outputs = []
        for b in range(batch_size):
            seq_out = []
            for s in range(seq_len):
                bits = self._position_to_bits(int(positions[b, s]))
                state = anp.zeros(2 ** self.n_qubits, dtype=complex)
                state = state.at[0].set(1.0) if hasattr(state, 'at') else anp.array(
                    [1.0] + [0.0] * (2 ** self.n_qubits - 1), dtype=complex
                )
                for i in range(self.n_qubits):
                    state = apply_gate(state, _rx(np.pi * bits[i]), i, self.n_qubits)
                for i in range(self.n_qubits):
                    state = apply_gate(state, _ry(self.params[i]), i, self.n_qubits)
                    state = apply_gate(state, _rz(self.params[i + self.n_qubits]), i, self.n_qubits)
                for i in range(self.n_qubits - 1):
                    state = apply_cnot(state, i, i + 1, self.n_qubits)
                seq_out.append(anp.array([pauli_y_expval(state, i, self.n_qubits)
                                           for i in range(self.n_qubits)]))
            outputs.append(anp.stack(seq_out))
        return anp.stack(outputs)


# ------------------------------------------------------------------
# Quantum Attention Circuit
# ------------------------------------------------------------------

class QuantumAttentionCircuit(Module):
    """
    Quantum attention: computes Q-K similarity and applies it to V.

    Mechanism:
      1. Encode Q, K, V into separate qubit registers
      2. CNOT(Q, K) computes |Q⟩|K⊕Q⟩ — when Q≈K, K-register → |0⟩
      3. Controlled rotations from K to V transfer the similarity signal
      4. Trainable refinement on V-register preserves the Q-K→V structure
      5. Undo Q-K CNOT to reset K for next layer
      6. Measure PauliZ on V-register

    Backend selection:
      - n_qubits <= 10: fast numpy simulator (CPU, autograd-compatible)
      - n_qubits > 10: PennyLane QNode (GPU via lightning.gpu if available,
        else lightning.qubit CPU C++, else default.qubit)
    """

    GPU_QUBIT_THRESHOLD = 10

    def __init__(self, n_qubits: int = 6, n_layers: int = 2, force_qnode: bool = False):
        assert n_qubits % 3 == 0, "n_qubits must be divisible by 3"
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qubits_per_register = n_qubits // 3
        self.register_dim = 2 ** self.qubits_per_register
        self.use_qnode = (n_qubits > self.GPU_QUBIT_THRESHOLD) or force_qnode

        self.device = self._select_device(n_qubits)
        self._build_circuit()

        qpr = self.qubits_per_register
        n_params = n_layers * (qpr + qpr * 2)
        self.params = Parameter(np.random.randn(n_params) * 0.5)

    @staticmethod
    def _select_device(n_qubits: int):
        """Auto-select best available backend based on qubit count."""
        if n_qubits <= QuantumAttentionCircuit.GPU_QUBIT_THRESHOLD:
            return qml.device("default.qubit", wires=n_qubits)

        # For >10 qubits, try GPU first, then C++ CPU, then default
        for backend, kwargs in [
            ("lightning.gpu", {}),
            ("lightning.qubit", {}),
            ("default.qubit", {}),
        ]:
            try:
                dev = qml.device(backend, wires=n_qubits)
                return dev
            except Exception:
                continue
        return qml.device("default.qubit", wires=n_qubits)

    def _get_diff_method(self):
        """Select differentiation method based on device."""
        dev_name = getattr(self.device, 'short_name', '') or getattr(self.device, 'name', '') or str(type(self.device))
        if 'lightning' in dev_name:
            return "adjoint"
        return "backprop"

    def _build_circuit(self):
        qpr = self.qubits_per_register
        q_wires = list(range(0, qpr))
        k_wires = list(range(qpr, 2 * qpr))
        v_wires = list(range(2 * qpr, 3 * qpr))
        diff_method = self._get_diff_method()

        @qml.qnode(self.device, interface="autograd", diff_method=diff_method)
        def circuit(q_features, k_features, v_features, params):
            qml.AmplitudeEmbedding(q_features, wires=q_wires, normalize=True)
            qml.AmplitudeEmbedding(k_features, wires=k_wires, normalize=True)
            qml.AmplitudeEmbedding(v_features, wires=v_wires, normalize=True)

            idx = 0
            for layer in range(self.n_layers):
                for i in range(qpr):
                    qml.CNOT(wires=[q_wires[i], k_wires[i]])
                for i in range(qpr):
                    qml.CNOT(wires=[k_wires[i], v_wires[i]])
                    qml.RY(params[idx], wires=v_wires[i])
                    qml.CNOT(wires=[k_wires[i], v_wires[i]])
                    idx += 1
                for i in range(qpr):
                    qml.RY(params[idx], wires=v_wires[i])
                    qml.RZ(params[idx + 1], wires=v_wires[i])
                    idx += 2
                for i in range(qpr - 1):
                    qml.CNOT(wires=[v_wires[i], v_wires[i + 1]])
                for i in range(qpr):
                    qml.CNOT(wires=[q_wires[i], k_wires[i]])

            return [qml.expval(qml.PauliZ(w)) for w in v_wires]

        self._circuit = circuit

    def forward(self, q, k, v):
        if self.use_qnode:
            return self._forward_qnode(q, k, v)
        return self._forward_fast_sim(q, k, v)

    def _forward_qnode(self, q, k, v):
        """Use PennyLane QNode (for >10 qubits or GPU execution)."""
        dim = self.register_dim
        q_padded = _pad_to_dim(q, dim)
        k_padded = _pad_to_dim(k, dim)
        v_padded = _pad_to_dim(v, dim)
        result = self._circuit(q_padded, k_padded, v_padded, self.params)
        return anp.array(result)

    def _forward_fast_sim(self, q, k, v):
        """Use fast numpy simulator (for ≤10 qubits, CPU)."""
        dim = self.register_dim
        qpr = self.qubits_per_register
        n = self.n_qubits
        q_padded = _pad_to_dim(q, dim)
        k_padded = _pad_to_dim(k, dim)
        v_padded = _pad_to_dim(v, dim)

        sq = amplitude_embed(q_padded, qpr)
        sk = amplitude_embed(k_padded, qpr)
        sv = amplitude_embed(v_padded, qpr)
        state = anp.kron(anp.kron(sq, sk), sv)

        q_wires = list(range(0, qpr))
        k_wires = list(range(qpr, 2 * qpr))
        v_wires = list(range(2 * qpr, 3 * qpr))

        idx = 0
        for layer in range(self.n_layers):
            for i in range(qpr):
                state = apply_cnot(state, q_wires[i], k_wires[i], n)
            for i in range(qpr):
                state = apply_cnot(state, k_wires[i], v_wires[i], n)
                state = apply_gate(state, _ry(self.params[idx]), v_wires[i], n)
                state = apply_cnot(state, k_wires[i], v_wires[i], n)
                idx += 1
            for i in range(qpr):
                state = apply_gate(state, _ry(self.params[idx]), v_wires[i], n)
                state = apply_gate(state, _rz(self.params[idx + 1]), v_wires[i], n)
                idx += 2
            for i in range(qpr - 1):
                state = apply_cnot(state, v_wires[i], v_wires[i + 1], n)
            for i in range(qpr):
                state = apply_cnot(state, q_wires[i], k_wires[i], n)

        return anp.array([pauli_z_expval(state, w, n) for w in v_wires])


# ------------------------------------------------------------------
# Quantum Activation Circuit
# ------------------------------------------------------------------

class QuantumActivationCircuit(Module):
    """Short variational circuit acting as a learnable non-linear activation."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 1):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.state_dim = 2 ** n_qubits

        self.device = qml.device("default.qubit", wires=n_qubits)
        self._build_circuit()

        n_params = n_qubits * n_layers * 2
        self.params = Parameter(np.random.randn(n_params) * 0.1)

    def _build_circuit(self):
        @qml.qnode(self.device, interface="autograd", diff_method="backprop")
        def circuit(features, params):
            qml.AmplitudeEmbedding(features, wires=range(self.n_qubits), normalize=True)
            idx = 0
            for _ in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[idx], wires=i)
                    qml.RZ(params[idx + 1], wires=i)
                    idx += 2
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit = circuit

    def forward(self, features):
        padded = _pad_to_dim(features, self.state_dim)
        state = amplitude_embed(padded, self.n_qubits)
        idx = 0
        for _ in range(self.n_layers):
            for i in range(self.n_qubits):
                state = apply_gate(state, _ry(self.params[idx]), i, self.n_qubits)
                state = apply_gate(state, _rz(self.params[idx + 1]), i, self.n_qubits)
                idx += 2
            for i in range(self.n_qubits):
                state = apply_cnot(state, i, (i + 1) % self.n_qubits, self.n_qubits)
        return anp.array([pauli_z_expval(state, i, self.n_qubits)
                           for i in range(self.n_qubits)])
