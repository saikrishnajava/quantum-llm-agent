"""
MoE Quantum Multi-Head Attention
==================================
Drop-in replacement for QuantumMultiHeadAttention with adaptive qubit routing.
"""

from __future__ import annotations

import math

import numpy as np
import autograd.numpy as anp

from classical.nn import Module, Linear, Dropout, Softmax
from quantum.moe import AdaptiveQuantumHead


class MoEQuantumMultiHeadAttention(Module):
    """
    Multi-head attention where quantum heads use mixture-of-experts
    with different qubit counts. Classical heads unchanged.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        moe_heads: int = 1,
        qubit_configs: list[int] = (6, 9, 12),
        n_circuit_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.moe_heads = moe_heads
        self.classical_heads = n_heads - moe_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

        if moe_heads > 0:
            self.moe_modules = [
                AdaptiveQuantumHead(
                    d_model=self.head_dim,
                    head_dim=self.head_dim,
                    qubit_configs=qubit_configs,
                    n_layers=n_circuit_layers,
                    temperature=temperature,
                )
                for _ in range(moe_heads)
            ]

        self.softmax = Softmax(axis=-1)
        self.dropout = Dropout(dropout)
        self.load_balance_losses = []

    def _classical_attention(self, q, k, v, mask):
        scores = (q @ k.transpose((0, 1, 3, 2))) / self.scale
        if mask is not None:
            scores = anp.where(mask, scores, -1e9)
        weights = self.softmax(scores)
        return weights @ v

    def forward(self, x, attention_mask=None):
        B, S, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).reshape(B, S, H, D)
        k = self.k_proj(x).reshape(B, S, H, D)
        v = self.v_proj(x).reshape(B, S, H, D)

        head_outputs = []
        self.load_balance_losses = []

        if self.classical_heads > 0:
            cq = q[:, :, :self.classical_heads, :].transpose((0, 2, 1, 3))
            ck = k[:, :, :self.classical_heads, :].transpose((0, 2, 1, 3))
            cv = v[:, :, :self.classical_heads, :].transpose((0, 2, 1, 3))
            classical_out = self._classical_attention(cq, ck, cv, attention_mask)
            classical_out = classical_out.transpose((0, 2, 1, 3))
            head_outputs.append(classical_out)

        for h_idx in range(self.moe_heads):
            head_offset = self.classical_heads + h_idx
            hq = q[:, :, head_offset, :]
            hk = k[:, :, head_offset, :]
            hv = v[:, :, head_offset, :]

            moe_out, load_loss = self.moe_modules[h_idx](hq, hk, hv)
            head_outputs.append(moe_out[:, :, np.newaxis, :])
            self.load_balance_losses.append(load_loss)

        combined = anp.concatenate(head_outputs, axis=2).reshape(B, S, self.d_model)
        return self.out_proj(self.dropout(combined))
