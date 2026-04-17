"""
Quantum Reasoning Module  (NumPy)
===================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import autograd.numpy as anp
import pennylane as qml
import pennylane.numpy as pnp

from classical.nn import Module, Linear, Parameter
from quantum.simulator import (
    apply_gate, apply_cnot, amplitude_embed,
    pauli_z_expval, probability, _ry, _rz,
)


class QuantumDecisionCircuit(Module):
    def __init__(self, n_qubits: int = 6, n_layers: int = 2):
        assert n_qubits % 2 == 0
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reg_qubits = n_qubits // 2
        self.reg_dim = 2 ** self.reg_qubits

        self.device = qml.device("default.qubit", wires=n_qubits)
        self._build_circuit()
        n_params = n_qubits * n_layers * 2 + self.reg_qubits
        self.params = Parameter(np.random.randn(n_params) * 0.1)

    def _build_circuit(self):
        ctx_wires = list(range(self.reg_qubits))
        opt_wires = list(range(self.reg_qubits, self.n_qubits))

        @qml.qnode(self.device, interface="autograd", diff_method="backprop")
        def circuit(context_features, option_features, params):
            qml.AmplitudeEmbedding(context_features, wires=ctx_wires, normalize=True)
            qml.AmplitudeEmbedding(option_features, wires=opt_wires, normalize=True)
            idx = 0
            for i in range(self.reg_qubits):
                qml.CNOT(wires=[ctx_wires[i], opt_wires[i]])
                qml.RY(params[idx], wires=opt_wires[i])
                idx += 1
            for _ in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[idx], wires=i)
                    qml.RZ(params[idx + 1], wires=i)
                    idx += 2
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.probs(wires=opt_wires)

        self._circuit = circuit

    def forward(self, context: np.ndarray, options: np.ndarray) -> np.ndarray:
        dim = self.reg_dim
        n = self.n_qubits
        rq = self.reg_qubits
        ctx_wires = list(range(rq))
        opt_wires = list(range(rq, n))

        def _pad(vec, target_dim):
            vec_slice = vec[:target_dim]
            pad_len = max(0, target_dim - vec_slice.shape[0])
            if pad_len > 0:
                return anp.concatenate([vec_slice, anp.zeros(pad_len)])
            return vec_slice

        ctx = _pad(context, dim)
        opt = _pad(options, dim)

        s_ctx = amplitude_embed(ctx, rq)
        s_opt = amplitude_embed(opt, rq)
        state = anp.kron(s_ctx, s_opt)

        idx = 0
        for i in range(rq):
            state = apply_cnot(state, ctx_wires[i], opt_wires[i], n)
            state = apply_gate(state, _ry(self.params[idx]), opt_wires[i], n)
            idx += 1
        for _ in range(self.n_layers):
            for i in range(n):
                state = apply_gate(state, _ry(self.params[idx]), i, n)
                state = apply_gate(state, _rz(self.params[idx + 1]), i, n)
                idx += 2
            for i in range(n - 1):
                state = apply_cnot(state, i, i + 1, n)

        return probability(state, opt_wires, n)


class QuantumPatternMatcher(Module):
    def __init__(self, n_qubits: int = 6, n_layers: int = 1):
        assert n_qubits % 2 == 0
        self.n_qubits = n_qubits
        self.reg_qubits = n_qubits // 2
        self.reg_dim = 2 ** self.reg_qubits

        self.device = qml.device("default.qubit", wires=n_qubits)
        self._build_circuit()
        n_params = n_qubits * n_layers * 2
        self.params = Parameter(np.random.randn(n_params) * 0.1)

    def _build_circuit(self):
        in_wires = list(range(self.reg_qubits))
        mem_wires = list(range(self.reg_qubits, self.n_qubits))

        @qml.qnode(self.device, interface="autograd", diff_method="backprop")
        def circuit(input_features, memory_features, params):
            qml.AmplitudeEmbedding(input_features, wires=in_wires, normalize=True)
            qml.AmplitudeEmbedding(memory_features, wires=mem_wires, normalize=True)
            idx = 0
            for i in range(self.n_qubits):
                qml.RY(params[idx], wires=i)
                qml.RZ(params[idx + 1], wires=i)
                idx += 2
            for i in range(self.reg_qubits):
                qml.CNOT(wires=[in_wires[i], mem_wires[i]])
            return [qml.expval(qml.PauliZ(w)) for w in mem_wires]

        self._circuit = circuit

    def forward(self, input_pattern: np.ndarray, memory_pattern: np.ndarray) -> np.ndarray:
        dim = self.reg_dim
        n = self.n_qubits
        rq = self.reg_qubits
        in_wires = list(range(rq))
        mem_wires = list(range(rq, n))

        def _pad(vec, target_dim):
            vec_slice = vec[:target_dim]
            pad_len = max(0, target_dim - vec_slice.shape[0])
            if pad_len > 0:
                return anp.concatenate([vec_slice, anp.zeros(pad_len)])
            return vec_slice

        inp = _pad(input_pattern, dim)
        mem = _pad(memory_pattern, dim)

        s_in = amplitude_embed(inp, rq)
        s_mem = amplitude_embed(mem, rq)
        state = anp.kron(s_in, s_mem)

        idx = 0
        for i in range(n):
            state = apply_gate(state, _ry(self.params[idx]), i, n)
            state = apply_gate(state, _rz(self.params[idx + 1]), i, n)
            idx += 2
        for i in range(rq):
            state = apply_cnot(state, in_wires[i], mem_wires[i], n)

        return anp.array([pauli_z_expval(state, w, n) for w in mem_wires])


class QuantumReasoningModule(Module):
    def __init__(self, d_model: int = 128, reasoning_qubits: int = 6, n_layers: int = 2):
        self.d_model = d_model
        self.decision_circuit = QuantumDecisionCircuit(reasoning_qubits, n_layers)
        self.pattern_matcher = QuantumPatternMatcher(reasoning_qubits, n_layers=1)
        reg_dim = 2 ** (reasoning_qubits // 2)
        self.context_proj = Linear(d_model, reg_dim)
        self.option_proj = Linear(d_model, reg_dim)
        self.memory_proj = Linear(d_model, reg_dim)

    def forward(
        self,
        context: np.ndarray,
        options: np.ndarray,
        memory_state: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray | None]:
        squeeze = False
        if context.ndim == 1:
            context = context[np.newaxis, :]
            options = options[np.newaxis, :]
            if memory_state is not None:
                memory_state = memory_state[np.newaxis, :]
            squeeze = True

        batch_size = context.shape[0]
        decision_list, pattern_list = [], []

        for b in range(batch_size):
            ctx = self.context_proj(context[b])
            opt = self.option_proj(options[b])
            decision_list.append(self.decision_circuit(ctx, opt))
            if memory_state is not None:
                mem = self.memory_proj(memory_state[b])
                pattern_list.append(self.pattern_matcher(ctx, mem))

        decision_probs = anp.stack(decision_list)
        pattern_scores = anp.stack(pattern_list) if pattern_list else None

        if squeeze:
            decision_probs = decision_probs[0]
            if pattern_scores is not None:
                pattern_scores = pattern_scores[0]

        return {"decision_probs": decision_probs, "pattern_scores": pattern_scores}
