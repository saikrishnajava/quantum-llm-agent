"""
Integration tests for the full hybrid transformer (NumPy).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hybrid.interfaces.model import HybridQuantumLLM, HybridTransformerBlock
from hybrid.embeddings.hybrid_embedding import HybridEmbeddingLayer
from hybrid.attention.quantum_attention import QuantumMultiHeadAttention
from hybrid.feedforward.hybrid_ff import HybridFeedForward


def test_embedding_quantum_shape():
    layer = HybridEmbeddingLayer(
        vocab_size=100, d_model=32, max_seq_length=16,
        quantum_enhancement=True, n_qubits=3,
    )
    out = layer(np.random.randint(0, 100, (2, 8)))
    assert out.shape == (2, 8, 32)


def test_embedding_classical_shape():
    layer = HybridEmbeddingLayer(
        vocab_size=100, d_model=32, max_seq_length=16,
        quantum_enhancement=False,
    )
    out = layer(np.random.randint(0, 100, (2, 8)))
    assert out.shape == (2, 8, 32)


def test_attention_output_shape():
    attn = QuantumMultiHeadAttention(
        d_model=32, n_heads=4, quantum_heads=1,
        quantum_qubits=6, max_quantum_seq=4,
    )
    out = attn(np.random.randn(1, 8, 32))
    assert out.shape == (1, 8, 32)


def test_attention_all_classical():
    attn = QuantumMultiHeadAttention(
        d_model=32, n_heads=4, quantum_heads=0, quantum_qubits=6,
    )
    out = attn(np.random.randn(2, 4, 32))
    assert out.shape == (2, 4, 32)


def test_feedforward_classical_shape():
    ff = HybridFeedForward(d_model=32, d_ff=64, use_quantum_activation=False)
    out = ff(np.random.randn(2, 4, 32))
    assert out.shape == (2, 4, 32)


def test_transformer_block_shape():
    block = HybridTransformerBlock(
        d_model=32, n_heads=4, quantum_heads=1, d_ff=64,
        quantum_config={"attention_qubits": 6, "activation_qubits": 3,
                        "use_quantum_activation": False},
    )
    out = block(np.random.randn(1, 4, 32))
    assert out.shape == (1, 4, 32)


def test_llm_forward_shape():
    model = HybridQuantumLLM(
        vocab_size=100, d_model=32, n_layers=1, n_heads=4,
        quantum_heads_per_layer=1, d_ff=64, max_seq_length=16,
        quantum_config={
            "use_quantum_embedding": True, "embedding_qubits": 3,
            "attention_qubits": 6, "activation_qubits": 3,
            "use_quantum_activation": False,
        },
    )
    logits = model(np.random.randint(0, 100, (1, 8)))
    assert logits.shape == (1, 8, 100)


def test_llm_generate_shape():
    model = HybridQuantumLLM(
        vocab_size=100, d_model=32, n_layers=1, n_heads=4,
        quantum_heads_per_layer=1, d_ff=64, max_seq_length=16,
        quantum_config={
            "use_quantum_embedding": True, "embedding_qubits": 3,
            "attention_qubits": 6, "activation_qubits": 3,
            "use_quantum_activation": False,
        },
    )
    generated = model.generate(np.random.randint(0, 100, (1, 4)), max_new_tokens=3, do_sample=False)
    assert generated.shape == (1, 7)


def test_llm_count_parameters():
    model = HybridQuantumLLM(
        vocab_size=50, d_model=16, n_layers=1, n_heads=4,
        quantum_heads_per_layer=0, max_seq_length=8,
        quantum_config={
            "use_quantum_embedding": False, "embedding_qubits": 3,
            "attention_qubits": 6, "activation_qubits": 3,
            "use_quantum_activation": False,
        },
    )
    counts = model.count_parameters()
    assert counts["total"] > 0
    assert counts["classical"] + counts["quantum"] == counts["total"]


def test_classical_only_model():
    model = HybridQuantumLLM(
        vocab_size=50, d_model=16, n_layers=1, n_heads=4,
        quantum_heads_per_layer=0, max_seq_length=8,
        quantum_config={
            "use_quantum_embedding": False, "embedding_qubits": 3,
            "attention_qubits": 6, "activation_qubits": 3,
            "use_quantum_activation": False,
        },
    )
    logits = model(np.random.randint(0, 50, (2, 6)))
    assert logits.shape == (2, 6, 50)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
