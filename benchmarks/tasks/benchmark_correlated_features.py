"""
Benchmark Task 2: Correlated Feature Pairs
============================================
Classify inputs based on XOR of sign-agreement patterns between feature pairs.

Why quantum should help: CNOT(Q,K) produces |K XOR Q> — detects agreement/
disagreement natively. Classical dot-product = linear correlation only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmarks.runner import TaskConfig, run_task, save_results


def generate_correlated_features(
    n_features: int = 8,
    n_pairs: int = 3,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate continuous features, discretize, and label by XOR of pair agreements.

    The label is XOR of (sign(X[i]) == sign(X[j])) for selected pairs.
    Features are discretized into 8 bins for token input.
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test

    X_cont = rng.randn(n_total, n_features)

    pairs = []
    for _ in range(n_pairs):
        i, j = rng.choice(n_features, 2, replace=False)
        pairs.append((i, j))

    agreements = np.zeros((n_total, n_pairs), dtype=int)
    for p_idx, (i, j) in enumerate(pairs):
        agreements[:, p_idx] = (np.sign(X_cont[:, i]) == np.sign(X_cont[:, j])).astype(int)

    labels = np.bitwise_xor.reduce(agreements, axis=1)

    n_bins = 8
    X_discrete = np.zeros((n_total, n_features), dtype=int)
    for f in range(n_features):
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        bin_edges = np.percentile(X_cont[:, f], percentiles)
        X_discrete[:, f] = np.digitize(X_cont[:, f], bin_edges)

    X_train = X_discrete[:n_train]
    X_test = X_discrete[n_train:]
    labels_train = labels[:n_train]
    labels_test = labels[n_train:]

    y_train = np.concatenate([X_train[:, 1:], labels_train[:, None]], axis=1)
    y_test = np.concatenate([X_test[:, 1:], labels_test[:, None]], axis=1)

    return X_train, y_train, X_test, y_test


def main():
    print("=" * 60)
    print("BENCHMARK 2: CORRELATED FEATURE PAIRS")
    print("=" * 60)
    print("Task: Predict XOR of sign-agreement between feature pairs")
    print("Quantum advantage: CNOT detects XOR agreement natively")
    print()

    task_config = TaskConfig(
        task_name="correlated_features",
        vocab_size=8,
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
        {"n_features": 8, "n_pairs": 2},
        {"n_features": 8, "n_pairs": 3},
        {"n_features": 8, "n_pairs": 4},
    ]

    results = run_task(
        task_config=task_config,
        data_generator=generate_correlated_features,
        scaling_params=scaling_params,
    )

    save_results(results, "benchmark_correlated_features.json")

    print("\n" + "=" * 60)
    print("BENCHMARK 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
