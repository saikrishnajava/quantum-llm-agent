"""
Quantum Multi-Head Attention  (NumPy)
======================================
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import autograd.numpy as anp

from classical.nn import Module, Linear, Dropout, Softmax
from quantum.circuits.core import QuantumAttentionCircuit
from hybrid.interfaces.optimizations import CircuitResultCache


class QuantumMultiHeadAttention(Module):
    """
    Hybrid multi-head attention: *quantum_heads* go through parameterised
    quantum circuits, the rest use classical scaled-dot-product.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        quantum_heads: int = 1,
        quantum_qubits: int = 6,
        dropout: float = 0.1,
        max_quantum_seq: int = 16,
    ):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.quantum_heads = quantum_heads
        self.classical_heads = n_heads - quantum_heads
        self.head_dim = d_model // n_heads
        self.max_quantum_seq = max_quantum_seq
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

        if self.quantum_heads > 0:
            assert quantum_qubits % 3 == 0
            self.quantum_circuits = [
                QuantumAttentionCircuit(quantum_qubits, n_layers=2)
                for _ in range(self.quantum_heads)
            ]
            register_dim = quantum_qubits // 3
            self.quantum_out_proj = Linear(register_dim, self.head_dim)
            self._circuit_cache = CircuitResultCache(max_size=2048)

        self.softmax = Softmax(axis=-1)
        self.dropout = Dropout(dropout)

    # ------------------------------------------------------------------

    def _classical_attention(self, q, k, v, mask):
        scores = (q @ k.transpose((0, 1, 3, 2))) / self.scale
        if mask is not None:
            scores = anp.where(mask, scores, -1e9)
        weights = self.softmax(scores)
        return weights @ v

    def _quantum_head(self, q, k, v, circuit):
        batch_size, seq_len, _ = q.shape
        max_qs = min(seq_len, self.max_quantum_seq)
        use_cache = not getattr(self, '_training', True)

        all_outputs = []
        for b in range(batch_size):
            seq_outputs = []
            for s in range(max_qs):
                if use_cache:
                    cached = self._circuit_cache.get(q[b, s], k[b, s], v[b, s])
                    if cached is not None:
                        projected = self.quantum_out_proj(cached)
                        seq_outputs.append(projected)
                        continue
                qc_out = circuit(q[b, s], k[b, s], v[b, s])
                if use_cache:
                    self._circuit_cache.put(qc_out, q[b, s], k[b, s], v[b, s])
                projected = self.quantum_out_proj(qc_out)
                seq_outputs.append(projected)

            if seq_len > max_qs:
                rem_q = q[b, max_qs:]
                rem_k = k[b, max_qs:]
                rem_v = v[b, max_qs:]
                attn_w = self.softmax((rem_q @ rem_k.T) / self.scale)
                classical_out = attn_w @ rem_v
                for s in range(classical_out.shape[0]):
                    seq_outputs.append(classical_out[s])

            all_outputs.append(anp.stack(seq_outputs))
        return anp.stack(all_outputs)

    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        B, S, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).reshape(B, S, H, D)
        k = self.k_proj(x).reshape(B, S, H, D)
        v = self.v_proj(x).reshape(B, S, H, D)

        head_outputs = []

        if self.classical_heads > 0:
            cq = q[:, :, :self.classical_heads, :].transpose((0, 2, 1, 3))
            ck = k[:, :, :self.classical_heads, :].transpose((0, 2, 1, 3))
            cv = v[:, :, :self.classical_heads, :].transpose((0, 2, 1, 3))
            classical_out = self._classical_attention(cq, ck, cv, attention_mask)
            classical_out = classical_out.transpose((0, 2, 1, 3))
            head_outputs.append(classical_out)

        for h_idx in range(self.quantum_heads):
            head_offset = self.classical_heads + h_idx
            hq = q[:, :, head_offset, :]
            hk = k[:, :, head_offset, :]
            hv = v[:, :, head_offset, :]
            quantum_out = self._quantum_head(hq, hk, hv, self.quantum_circuits[h_idx])
            head_outputs.append(quantum_out[:, :, np.newaxis, :])

        combined = anp.concatenate(head_outputs, axis=2).reshape(B, S, self.d_model)
        return self.out_proj(self.dropout(combined))
