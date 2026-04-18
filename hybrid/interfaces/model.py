"""
Hybrid Quantum-Classical Transformer  (NumPy)
===============================================
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from classical.nn import Module, Linear, LayerNorm, Dropout, Softmax
from hybrid.embeddings.hybrid_embedding import HybridEmbeddingLayer
from hybrid.attention.quantum_attention import QuantumMultiHeadAttention
from hybrid.attention.moe_attention import MoEQuantumMultiHeadAttention
from hybrid.feedforward.hybrid_ff import HybridFeedForward

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


class HybridTransformerBlock(Module):
    """Pre-norm transformer block with quantum attention + hybrid FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        quantum_heads: int = 1,
        d_ff: int | None = None,
        dropout: float = 0.1,
        quantum_config: dict | None = None,
    ):
        if d_ff is None:
            d_ff = 4 * d_model
        if quantum_config is None:
            quantum_config = {}

        self.norm1 = LayerNorm(d_model)

        use_moe = quantum_config.get("use_moe", False)
        if use_moe and quantum_heads > 0:
            self.attention = MoEQuantumMultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                moe_heads=quantum_heads,
                qubit_configs=quantum_config.get("moe_qubit_configs", [6, 9, 12]),
                dropout=dropout,
                temperature=quantum_config.get("moe_temperature", 1.0),
            )
        else:
            self.attention = QuantumMultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                quantum_heads=quantum_heads,
                quantum_qubits=quantum_config.get("attention_qubits", 6),
                dropout=dropout,
            )
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = HybridFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            use_quantum_activation=quantum_config.get("use_quantum_activation", True),
            activation_qubits=quantum_config.get("activation_qubits", 4),
        )
        self.dropout = Dropout(dropout)

    def forward(self, x: np.ndarray, attention_mask=None) -> np.ndarray:
        residual = x
        x_normed = self.norm1(x)
        x = residual + self.dropout(self.attention(x_normed, attention_mask))

        residual = x
        x_normed = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x_normed))
        return x


class HybridQuantumLLM(Module):
    """
    Complete hybrid quantum-classical language model.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        quantum_heads_per_layer: int = 1,
        d_ff: int | None = None,
        max_seq_length: int = 128,
        dropout: float = 0.1,
        quantum_config: dict | None = None,
    ):
        if d_ff is None:
            d_ff = 4 * d_model
        if quantum_config is None:
            quantum_config = {
                "use_quantum_embedding": True,
                "embedding_qubits": 6,
                "attention_qubits": 6,
                "activation_qubits": 4,
                "use_quantum_activation": True,
            }

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        self.embedding = HybridEmbeddingLayer(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_length=max_seq_length,
            quantum_enhancement=quantum_config.get("use_quantum_embedding", True),
            n_qubits=quantum_config.get("embedding_qubits", 6),
        )

        self.layers = [
            HybridTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                quantum_heads=quantum_heads_per_layer,
                d_ff=d_ff,
                dropout=dropout,
                quantum_config=quantum_config,
            )
            for _ in range(n_layers)
        ]

        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: np.ndarray, attention_mask=None) -> np.ndarray:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln_f(x)
        return self.lm_head(x)

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> np.ndarray:
        self.eval()
        ids = np.array(input_ids)
        softmax = Softmax(axis=-1)
        for _ in range(max_new_tokens):
            # Sliding window: only feed the last max_seq_length tokens
            context = ids[:, -self.max_seq_length:]
            logits = self.forward(context)
            next_logits = logits[:, -1, :] / temperature
            if do_sample:
                probs = softmax(next_logits)
                # Sample from probability distribution
                next_token = np.array([
                    [np.random.choice(len(p), p=np.clip(p, 0, None) / np.clip(p, 0, None).sum())]
                    for p in probs
                ])
            else:
                next_token = next_logits.argmax(axis=-1, keepdims=True)
            ids = np.concatenate([ids, next_token], axis=1)
        return ids

    @classmethod
    def from_config(cls, config_name: str = "proof_of_concept") -> "HybridQuantumLLM":
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for from_config(). pip install pyyaml")
        path = CONFIG_DIR / "model_configs.yaml"
        with open(path) as f:
            configs = yaml.safe_load(f)
        cfg = configs[config_name]
        return cls(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            quantum_heads_per_layer=cfg["quantum_heads_per_layer"],
            d_ff=cfg.get("d_ff"),
            max_seq_length=cfg["max_seq_length"],
            dropout=cfg.get("dropout", 0.1),
            quantum_config=cfg.get("quantum"),
        )

    def count_parameters(self) -> dict:
        classical = 0
        quantum = 0
        for name, p in self.named_parameters():
            n = p.size
            if "quantum" in name or name.endswith(".params"):
                quantum += n
            else:
                classical += n
        return {"classical": classical, "quantum": quantum, "total": classical + quantum}
