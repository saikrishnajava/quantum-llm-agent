"""
Benchmark Task 1: Generalized k-Bit Parity
============================================
Classify binary sequences by the parity of a hidden subset of k bits.
Extends the original XOR-all benchmark to partial parity.

Why quantum should help: CNOT gate structure = native parity computation.
Classical shallow nets need exponential params for parity (Razborov-Smolensky).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmarks.runner import TaskConfig, run_task, save_results


def generate_k_parity(
    n_bits: int = 8,
    k: int = 3,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate binary sequences labeled by parity of a hidden k-bit subset.

    Returns (X_train, y_train, X_test, y_test) where:
      X_train: (n_train, n_bits) input sequences
      y_train: (n_train, n_bits) shifted input with parity label at end
      X_test, y_test: same format
    """
    rng = np.random.RandomState(seed)
    subset = sorted(rng.choice(n_bits, k, replace=False))

    X_all = rng.randint(0, 2, size=(n_train + n_test, n_bits))
    labels = np.bitwise_xor.reduce(X_all[:, subset], axis=1)

    X_train = X_all[:n_train]
    X_test = X_all[n_train:]
    labels_train = labels[:n_train]
    labels_test = labels[n_train:]

    y_train = np.concatenate([X_train[:, 1:], labels_train[:, None]], axis=1)
    y_test = np.concatenate([X_test[:, 1:], labels_test[:, None]], axis=1)

    return X_train, y_train, X_test, y_test


def main():
    print("=" * 60)
    print("BENCHMARK 1: GENERALIZED k-BIT PARITY")
    print("=" * 60)
    print("Task: Predict parity of hidden k-bit subset within n-bit input")
    print("Quantum advantage: CNOT computes XOR natively")
    print()

    task_config = TaskConfig(
        task_name="k_parity",
        vocab_size=2,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        max_seq_length=16,
        batch_size=16,
        learning_rate=5e-3,
        epochs=20,
        eval_every=5,
        n_seeds=5,
        models=["classical", "quantum_6q"],
    )

    scaling_params = [
        {"n_bits": 8, "k": 2},
        {"n_bits": 8, "k": 3},
        {"n_bits": 8, "k": 4},
        {"n_bits": 8, "k": 5},
    ]

    results = run_task(
        task_config=task_config,
        data_generator=generate_k_parity,
        scaling_params=scaling_params,
    )

    save_results(results, "benchmark_k_parity.json")

    print("\n" + "=" * 60)
    print("BENCHMARK 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
