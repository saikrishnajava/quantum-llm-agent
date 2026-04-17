"""
Quantum gradient flow verification tests.
"""

import sys
from pathlib import Path

import autograd
import autograd.numpy as anp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from quantum.circuits.core import (
    QuantumFeatureMapCircuit,
    QuantumAttentionCircuit,
    QuantumActivationCircuit,
)


def test_feature_map_gradient():
    circuit = QuantumFeatureMapCircuit(3, n_layers=1)
    x = np.random.randn(1, 1, 8)

    def loss(params):
        circuit.params = params
        return anp.sum(circuit(x))

    g = autograd.grad(loss)(circuit.params.copy())
    assert np.linalg.norm(np.array(g)) > 1e-10


def test_attention_circuit_gradient():
    circuit = QuantumAttentionCircuit(6, n_layers=1)
    q, k, v = np.random.randn(4), np.random.randn(4), np.random.randn(4)

    def loss(params):
        circuit.params = params
        return anp.sum(circuit(q, k, v))

    g = autograd.grad(loss)(circuit.params.copy())
    assert np.linalg.norm(np.array(g)) > 1e-10


def test_activation_circuit_gradient():
    circuit = QuantumActivationCircuit(3, n_layers=1)
    x = np.random.randn(8)

    def loss(params):
        circuit.params = params
        return anp.sum(circuit(x))

    g = autograd.grad(loss)(circuit.params.copy())
    assert np.linalg.norm(np.array(g)) > 1e-10


def test_quantum_training_step():
    from hybrid.interfaces.model import HybridQuantumLLM
    from classical.optimizers.trainer import HybridQuantumTrainer

    model = HybridQuantumLLM(
        vocab_size=30, d_model=16, n_layers=1, n_heads=4,
        quantum_heads_per_layer=1, d_ff=32, max_seq_length=16, dropout=0.0,
        quantum_config={
            "use_quantum_embedding": False, "embedding_qubits": 3,
            "attention_qubits": 6, "activation_qubits": 3,
            "use_quantum_activation": False,
        },
    )
    trainer = HybridQuantumTrainer(model, learning_rate=1e-3)
    x = np.array([[1, 2, 3, 4]])
    y = np.array([[2, 3, 4, 5]])

    result = trainer.training_step(x, y)
    assert result["grad_norm"] > 1e-10
    assert result["loss"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
