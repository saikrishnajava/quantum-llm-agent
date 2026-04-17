"""
Performance benchmark: quantum vs classical attention (NumPy).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hybrid.attention.quantum_attention import QuantumMultiHeadAttention
from classical.nn import MultiheadAttention


def _classical_attention_time(d_model, n_heads, seq_len, batch_size, n_iterations):
    attn = MultiheadAttention(d_model, n_heads)
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float64)
    attn(x)

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        attn(x)
    return (time.perf_counter() - t0) / n_iterations * 1000


def _quantum_attention_time(d_model, n_heads, quantum_heads, seq_len, batch_size, n_iterations):
    attn = QuantumMultiHeadAttention(
        d_model=d_model, n_heads=n_heads,
        quantum_heads=quantum_heads, quantum_qubits=6,
        max_quantum_seq=min(seq_len, 4),
    )
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float64)
    attn(x)

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        attn(x)
    return (time.perf_counter() - t0) / n_iterations * 1000


class TestAttentionBenchmark:
    """
    Benchmarks comparing quantum vs classical attention latency.
    These are informational — no assertions on speedup yet.
    """

    def test_small_benchmark(self):
        d_model, n_heads, seq_len, batch_size = 32, 4, 8, 1
        n_iter = 3

        classical_ms = _classical_attention_time(d_model, n_heads, seq_len, batch_size, n_iter)
        quantum_ms = _quantum_attention_time(d_model, n_heads, 1, seq_len, batch_size, n_iter)

        print(f"\n--- Attention Benchmark (d={d_model}, heads={n_heads}, seq={seq_len}) ---")
        print(f"  Classical: {classical_ms:.2f} ms")
        print(f"  Quantum (1 qhead): {quantum_ms:.2f} ms")
        print(f"  Ratio (quantum/classical): {quantum_ms / classical_ms:.1f}x")

        assert classical_ms >= 0
        assert quantum_ms >= 0
