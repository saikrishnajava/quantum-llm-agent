"""
Unit tests for core quantum circuits (NumPy).
"""

import sys
from pathlib import Path

import numpy as np
import pennylane as qml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from quantum.circuits.core import (
    QuantumFeatureMapCircuit,
    QuantumPositionalCircuit,
    QuantumAttentionCircuit,
    QuantumActivationCircuit,
)


def test_feature_map_output_shape():
    circuit = QuantumFeatureMapCircuit(4, n_layers=1)
    x = np.random.randn(2, 3, 8)
    out = circuit(x)
    assert out.shape == (2, 3, 4)


def test_feature_map_single_vector():
    circuit = QuantumFeatureMapCircuit(3, n_layers=1)
    x = np.random.randn(1, 1, 4)
    out = circuit(x)
    assert out.shape == (1, 1, 3)
    assert np.all(np.isfinite(out))


def test_feature_map_bounded():
    circuit = QuantumFeatureMapCircuit(4, n_layers=2)
    x = np.random.randn(1, 2, 8)
    out = circuit(x)
    assert np.all(out >= -1.0 - 1e-6)
    assert np.all(out <= 1.0 + 1e-6)


def test_positional_output_shape():
    circuit = QuantumPositionalCircuit(4, max_positions=64)
    positions = np.array([[0, 1, 2, 3]])
    out = circuit(positions)
    assert out.shape == (1, 4, 4)


def test_positional_different_outputs():
    circuit = QuantumPositionalCircuit(4, max_positions=64)
    out_a = circuit(np.array([[0]]))
    out_b = circuit(np.array([[5]]))
    assert not np.allclose(out_a, out_b, atol=1e-3)


def test_attention_circuit_output_shape():
    circuit = QuantumAttentionCircuit(6, n_layers=1)
    out = circuit(np.random.randn(4), np.random.randn(4), np.random.randn(4))
    assert out.shape == (2,)  # 6 // 3 = 2


def test_attention_circuit_bounded():
    circuit = QuantumAttentionCircuit(6, n_layers=1)
    out = circuit(np.random.randn(4), np.random.randn(4), np.random.randn(4))
    assert np.all(out >= -1.0 - 1e-6)
    assert np.all(out <= 1.0 + 1e-6)


def test_activation_circuit_output_shape():
    circuit = QuantumActivationCircuit(3, n_layers=1)
    out = circuit(np.random.randn(8))
    assert out.shape == (3,)


def test_activation_circuit_bounded():
    circuit = QuantumActivationCircuit(3, n_layers=1)
    out = circuit(np.random.randn(4))
    assert np.all(out >= -1.0 - 1e-6)
    assert np.all(out <= 1.0 + 1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
