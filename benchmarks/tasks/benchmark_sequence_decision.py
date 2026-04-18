"""
Benchmark Task 6: Sequence-Dependent XOR Decision (Full Model)
===============================================================
Predict the correct action from a token sequence, where rules involve
XOR relationships between tokens at specific positions.

Why quantum should help: Quantum attention computes Q-K via CNOT (XOR similarity).
Classical dot-product attention captures linear correlations only.
Includes control with equality-based rules where classical should match.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmarks.runner import TaskConfig, run_task, save_results


def generate_xor_sequence_decision(
    seq_len: int = 8,
    token_vocab: int = 4,
    n_actions: int = 4,
    n_rules: int = 2,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sequence decision task with XOR-based rules.

    Rule: if (X[p1] XOR X[p2]) == target_xor → action = a
    Default action for unmatched sequences is 0.
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test

    rules = []
    for _ in range(n_rules):
        p1, p2 = sorted(rng.choice(seq_len, 2, replace=False).tolist())
        target_xor = rng.randint(0, token_vocab)
        action = rng.randint(1, n_actions)
        rules.append((p1, p2, target_xor, action))

    X = rng.randint(0, token_vocab, (n_total, seq_len))
    labels = np.zeros(n_total, dtype=int)

    for p1, p2, target_xor, action in rules:
        mask = (X[:, p1] ^ X[:, p2]) == target_xor
        labels[mask] = action

    vocab_size = max(token_vocab, n_actions)
    X_train = X[:n_train]
    X_test = X[n_train:]
    labels_train = labels[:n_train]
    labels_test = labels[n_train:]

    y_train = np.concatenate([X_train[:, 1:], labels_train[:, None]], axis=1)
    y_test = np.concatenate([X_test[:, 1:], labels_test[:, None]], axis=1)

    return X_train, y_train, X_test, y_test


def generate_equality_sequence_decision(
    seq_len: int = 8,
    token_vocab: int = 4,
    n_actions: int = 4,
    n_rules: int = 2,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sequence decision task with equality-based rules (control).

    Rule: if X[p1] == target_val → action = a
    Classical dot-product should handle this fine (linear relationship).
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test

    rules = []
    for _ in range(n_rules):
        p1 = rng.randint(0, seq_len)
        target_val = rng.randint(0, token_vocab)
        action = rng.randint(1, n_actions)
        rules.append((p1, target_val, action))

    X = rng.randint(0, token_vocab, (n_total, seq_len))
    labels = np.zeros(n_total, dtype=int)

    for p1, target_val, action in rules:
        mask = X[:, p1] == target_val
        labels[mask] = action

    vocab_size = max(token_vocab, n_actions)
    X_train = X[:n_train]
    X_test = X[n_train:]
    labels_train = labels[:n_train]
    labels_test = labels[n_train:]

    y_train = np.concatenate([X_train[:, 1:], labels_train[:, None]], axis=1)
    y_test = np.concatenate([X_test[:, 1:], labels_test[:, None]], axis=1)

    return X_train, y_train, X_test, y_test


def main():
    print("=" * 60)
    print("BENCHMARK 6: SEQUENCE-DEPENDENT XOR DECISION")
    print("=" * 60)
    print("Task: Predict action from token sequence (XOR vs equality rules)")
    print("Quantum advantage: XOR rules match CNOT attention structure")
    print()

    base_config = TaskConfig(
        task_name="sequence_xor_decision",
        vocab_size=4,
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

    # XOR-based rules (quantum should win)
    print("\n--- XOR Rules (quantum-favorable) ---")
    xor_scaling = [
        {"seq_len": 8, "token_vocab": 4, "n_actions": 4, "n_rules": 1},
        {"seq_len": 8, "token_vocab": 4, "n_actions": 4, "n_rules": 2},
        {"seq_len": 8, "token_vocab": 4, "n_actions": 4, "n_rules": 3},
    ]

    xor_results = run_task(
        task_config=base_config,
        data_generator=generate_xor_sequence_decision,
        scaling_params=xor_scaling,
    )
    save_results(xor_results, "benchmark_sequence_xor.json")

    # Equality-based rules (control — no quantum advantage expected)
    print("\n--- Equality Rules (control — classical should match) ---")
    eq_config = TaskConfig(
        task_name="sequence_equality_decision",
        vocab_size=4,
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

    eq_scaling = [
        {"seq_len": 8, "token_vocab": 4, "n_actions": 4, "n_rules": 1},
        {"seq_len": 8, "token_vocab": 4, "n_actions": 4, "n_rules": 2},
        {"seq_len": 8, "token_vocab": 4, "n_actions": 4, "n_rules": 3},
    ]

    eq_results = run_task(
        task_config=eq_config,
        data_generator=generate_equality_sequence_decision,
        scaling_params=eq_scaling,
    )
    save_results(eq_results, "benchmark_sequence_equality.json")

    print("\n" + "=" * 60)
    print("BENCHMARK 6 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
