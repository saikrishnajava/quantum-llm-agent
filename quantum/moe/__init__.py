"""
Adaptive Qubit Mixture-of-Experts
==================================
Trainable router selects quantum circuit size per input.
Soft routing during training, hard routing during inference.
"""

from __future__ import annotations

import numpy as np
import autograd.numpy as anp

from classical.nn import Module, Linear, Softmax, Parameter
from quantum.circuits.core import QuantumAttentionCircuit


class QubitRouter(Module):
    """Trainable router that scores each quantum expert per input."""

    def __init__(self, d_model: int, n_experts: int):
        self.gate = Linear(d_model, n_experts)
        self.softmax = Softmax(axis=-1)

    def forward(self, x):
        """
        x: (batch, seq, d_model) → gate_probs: (batch, n_experts)
        """
        pooled = x.mean(axis=1)
        return self.gate(pooled)


def gumbel_softmax(logits, temperature=1.0):
    """Differentiable approximation to argmax."""
    u = np.random.uniform(0, 1, logits.shape) + 1e-20
    gumbel = -anp.log(-anp.log(u + 1e-20) + 1e-20)
    y = (logits + gumbel) / temperature
    shifted = y - y.max(axis=-1, keepdims=True)
    exp_y = anp.exp(shifted)
    return exp_y / exp_y.sum(axis=-1, keepdims=True)


class AdaptiveQuantumHead(Module):
    """
    MoE quantum attention head with multiple circuit sizes.
    Router selects which circuit to weight most per input.
    """

    def __init__(
        self,
        d_model: int,
        head_dim: int,
        qubit_configs: list[int] = (6, 9, 12),
        n_layers: int = 2,
        temperature: float = 1.0,
    ):
        self.head_dim = head_dim
        self.qubit_configs = list(qubit_configs)
        self.n_experts = len(qubit_configs)
        self.temperature = temperature

        self.router = QubitRouter(d_model, self.n_experts)

        self.experts = [
            QuantumAttentionCircuit(n_qubits=q, n_layers=n_layers)
            for q in qubit_configs
        ]

        self.projections = [
            Linear(q // 3, head_dim)
            for q in qubit_configs
        ]

    def forward(self, q, k, v):
        """
        q, k, v: (batch, seq, head_dim)
        Returns: (output, load_loss)
          output: (batch, seq, head_dim)
          load_loss: scalar
        """
        batch, seq, _ = q.shape
        is_training = getattr(self, '_training', True)

        router_logits = self.router(q)

        if is_training:
            gate_probs = gumbel_softmax(router_logits, self.temperature)
        else:
            gate_probs = self.router.softmax(router_logits)

        load_loss = anp.sum(gate_probs.mean(axis=0) ** 2)

        if not is_training:
            seq_outs = []
            for b in range(batch):
                expert_idx = int(anp.argmax(gate_probs[b]))
                circuit = self.experts[expert_idx]
                proj = self.projections[expert_idx]
                for s in range(seq):
                    qc_out = circuit(q[b, s], k[b, s], v[b, s])
                    seq_outs.append(proj(qc_out))
            output = anp.stack(seq_outs).reshape(batch, seq, self.head_dim)
            return output, load_loss

        # Soft routing: run all experts, weighted sum
        expert_outputs = []
        for exp_i in range(self.n_experts):
            circuit = self.experts[exp_i]
            proj = self.projections[exp_i]
            batch_seq_outs = []
            for b in range(batch):
                for s in range(seq):
                    qc_out = circuit(q[b, s], k[b, s], v[b, s])
                    batch_seq_outs.append(proj(qc_out))
            exp_out = anp.stack(batch_seq_outs).reshape(batch, seq, self.head_dim)
            expert_outputs.append(exp_out)

        output = anp.zeros((batch, seq, self.head_dim))
        for exp_i in range(self.n_experts):
            weight = gate_probs[:, exp_i].reshape(batch, 1, 1)
            output = output + weight * expert_outputs[exp_i]

        return output, load_loss
