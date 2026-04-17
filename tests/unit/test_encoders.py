"""
Unit tests for quantum encoders (NumPy).
"""

import sys
from pathlib import Path

import numpy as np
import pennylane as qml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from quantum.encodings.encoders import (
    AmplitudeEncoder, AngleEncoder, BasisEncoder, VariationalEncoder,
    _pad_to_power_of_two, _normalize,
)


def test_pad_already_power_of_two():
    assert len(_pad_to_power_of_two(np.array([1, 2, 3, 4.]))) == 4


def test_pad_adds_zeros():
    padded = _pad_to_power_of_two(np.array([1, 2, 3.]))
    assert len(padded) == 4
    assert padded[3] == 0.0


def test_normalize_unit_norm():
    normed = _normalize(np.array([3., 4.]))
    np.testing.assert_almost_equal(np.linalg.norm(normed), 1.0)


def test_normalize_zero_vector():
    normed = _normalize(np.zeros(4))
    assert normed[0] == 1.0


def test_amplitude_qubits_required():
    assert AmplitudeEncoder.qubits_required(4) == 2
    assert AmplitudeEncoder.qubits_required(8) == 3
    assert AmplitudeEncoder.qubits_required(5) == 3


def test_amplitude_encoder_runs():
    n_qubits = 3
    dev = qml.device("default.qubit", wires=n_qubits)
    enc = AmplitudeEncoder(n_qubits)

    @qml.qnode(dev)
    def circuit(features):
        enc.encode(range(n_qubits), features)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    result = circuit(np.array([1., .5, .3, .2, .1, 0, 0, 0]))
    assert len(result) == n_qubits


def test_angle_encoder_runs():
    dev = qml.device("default.qubit", wires=3)
    enc = AngleEncoder(3, rotation="Y")

    @qml.qnode(dev)
    def circuit(features):
        enc.encode(range(3), features)
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    assert len(circuit(np.array([.5, 1., 1.5]))) == 3


def test_basis_encoder_zero():
    dev = qml.device("default.qubit", wires=3)
    enc = BasisEncoder(3)

    @qml.qnode(dev)
    def circuit():
        enc.encode(range(3), 0)
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    np.testing.assert_array_almost_equal(circuit(), [1, 1, 1])


def test_basis_encoder_seven():
    dev = qml.device("default.qubit", wires=3)
    enc = BasisEncoder(3)

    @qml.qnode(dev)
    def circuit():
        enc.encode(range(3), 7)
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    np.testing.assert_array_almost_equal(circuit(), [-1, -1, -1])


def test_variational_encoder_param_count():
    enc = VariationalEncoder(4, n_layers=2)
    assert enc.n_params == 4 * 2 * 3


def test_variational_encoder_runs():
    enc = VariationalEncoder(3, n_layers=1)
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(features, params):
        enc.encode(range(3), features, params)
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    result = circuit(np.array([.1, .2, .3]), np.random.randn(enc.n_params))
    assert len(result) == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
