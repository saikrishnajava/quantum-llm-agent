"""
Unit tests for quantum reasoning and memory (NumPy).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agents.reasoning.quantum_reasoning import (
    QuantumDecisionCircuit, QuantumPatternMatcher, QuantumReasoningModule,
)
from agents.memory.quantum_memory import QuantumMemoryNetwork


def test_decision_output_shape():
    c = QuantumDecisionCircuit(n_qubits=4, n_layers=1)
    probs = c(np.random.randn(4), np.random.randn(4))
    assert probs.shape == (4,)  # 2^(4//2)


def test_decision_probs_sum_to_one():
    c = QuantumDecisionCircuit(n_qubits=4, n_layers=1)
    probs = c(np.random.randn(4), np.random.randn(4))
    assert abs(probs.sum() - 1.0) < 1e-5


def test_pattern_matcher_shape():
    m = QuantumPatternMatcher(n_qubits=4, n_layers=1)
    scores = m(np.random.randn(4), np.random.randn(4))
    assert scores.shape == (2,)


def test_reasoning_module_keys():
    mod = QuantumReasoningModule(d_model=16, reasoning_qubits=4)
    result = mod(np.random.randn(16), np.random.randn(16))
    assert "decision_probs" in result
    assert "pattern_scores" in result


def test_reasoning_with_memory():
    mod = QuantumReasoningModule(d_model=16, reasoning_qubits=4)
    result = mod(np.random.randn(16), np.random.randn(16), memory_state=np.random.randn(16))
    assert result["pattern_scores"] is not None


def test_reasoning_batch():
    mod = QuantumReasoningModule(d_model=16, reasoning_qubits=4)
    result = mod(np.random.randn(2, 16), np.random.randn(2, 16))
    assert result["decision_probs"].shape[0] == 2


def test_memory_store_and_size():
    mem = QuantumMemoryNetwork(d_model=16, memory_size=10, n_qubits=4)
    assert mem.size == 0
    mem.store(np.random.randn(16))
    assert mem.size == 1


def test_memory_recall_empty():
    mem = QuantumMemoryNetwork(d_model=16, memory_size=10, n_qubits=4)
    assert mem.recall(np.random.randn(16)) == []


def test_memory_recall_returns_items():
    mem = QuantumMemoryNetwork(d_model=16, memory_size=10, n_qubits=4)
    for _ in range(5):
        mem.store(np.random.randn(16))
    assert len(mem.recall(np.random.randn(16), top_k=3)) == 3


def test_memory_limit():
    mem = QuantumMemoryNetwork(d_model=16, memory_size=3, n_qubits=4)
    for _ in range(10):
        mem.store(np.random.randn(16))
    assert mem.size == 3


def test_memory_clear():
    mem = QuantumMemoryNetwork(d_model=16, memory_size=10, n_qubits=4)
    mem.store(np.random.randn(16))
    mem.clear()
    assert mem.size == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
