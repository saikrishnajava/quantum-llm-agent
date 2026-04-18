"""
Benchmark Task 5: Random Boolean Function Learning (XOR-SAT)
=============================================================
Learn random Boolean functions in XOR-SAT vs CNF form.

Why quantum should help: Quantum depth-d circuits represent strictly more
Boolean functions than classical depth-d (Bravyi, Gosset, Konig 2018).
XOR-SAT clauses match CNOT gate structure exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmarks.runner import TaskConfig, run_task, save_results


def generate_xor_sat(
    n_bits: int = 8,
    n_clauses: int = 3,
    clause_size: int = 3,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate XOR-parity Boolean function over multiple clauses.

    Label = XOR of individual clause parities. This maintains ~50% class
    balance regardless of n_clauses (XOR of balanced bits is balanced).
    Each clause: parity of clause_size randomly chosen bits.
    The overall label = XOR(clause1_parity, clause2_parity, ...).
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test

    clauses = []
    for _ in range(n_clauses):
        bits = sorted(rng.choice(n_bits, clause_size, replace=False).tolist())
        clauses.append(bits)

    X = rng.randint(0, 2, (n_total, n_bits))

    clause_parities = np.zeros((n_total, n_clauses), dtype=int)
    for c_idx, bits in enumerate(clauses):
        clause_parities[:, c_idx] = np.bitwise_xor.reduce(X[:, bits], axis=1)

    labels = np.bitwise_xor.reduce(clause_parities, axis=1)

    X_train = X[:n_train]
    X_test = X[n_train:]
    labels_train = labels[:n_train]
    labels_test = labels[n_train:]

    y_train = np.concatenate([X_train[:, 1:], labels_train[:, None]], axis=1)
    y_test = np.concatenate([X_test[:, 1:], labels_test[:, None]], axis=1)

    return X_train, y_train, X_test, y_test


def generate_cnf(
    n_bits: int = 8,
    n_clauses: int = 3,
    clause_size: int = 3,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate CNF Boolean function: AND of OR clauses (control — no quantum advantage expected).

    Label = 1 iff ALL OR clauses are satisfied.
    Each clause: OR of (possibly negated) clause_size bits.
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test

    clauses = []
    for _ in range(n_clauses):
        bits = sorted(rng.choice(n_bits, clause_size, replace=False).tolist())
        negations = rng.randint(0, 2, clause_size).tolist()
        clauses.append((bits, negations))

    X = rng.randint(0, 2, (n_total, n_bits))

    labels = np.ones(n_total, dtype=int)
    for bits, negations in clauses:
        clause_satisfied = np.zeros(n_total, dtype=int)
        for bit_idx, neg in zip(bits, negations):
            literal = X[:, bit_idx] ^ neg
            clause_satisfied |= literal
        labels &= clause_satisfied

    X_train = X[:n_train]
    X_test = X[n_train:]
    labels_train = labels[:n_train]
    labels_test = labels[n_train:]

    y_train = np.concatenate([X_train[:, 1:], labels_train[:, None]], axis=1)
    y_test = np.concatenate([X_test[:, 1:], labels_test[:, None]], axis=1)

    return X_train, y_train, X_test, y_test


def main():
    print("=" * 60)
    print("BENCHMARK 5: BOOLEAN FUNCTION LEARNING")
    print("=" * 60)
    print("Task: Learn XOR-SAT vs CNF Boolean functions")
    print("Quantum advantage: XOR-SAT matches CNOT structure; CNF does not")
    print()

    task_config = TaskConfig(
        task_name="boolean_xor_sat",
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

    # XOR-SAT experiments (quantum should win)
    print("\n--- XOR-SAT Functions (quantum-favorable) ---")
    xor_scaling = [
        {"n_bits": 8, "n_clauses": 2, "clause_size": 3},
        {"n_bits": 8, "n_clauses": 3, "clause_size": 3},
        {"n_bits": 8, "n_clauses": 4, "clause_size": 3},
    ]

    xor_results = run_task(
        task_config=task_config,
        data_generator=generate_xor_sat,
        scaling_params=xor_scaling,
    )
    save_results(xor_results, "benchmark_boolean_xor_sat.json")

    # CNF experiments (control — no quantum advantage expected)
    print("\n--- CNF Functions (control — classical-favorable) ---")
    task_config_cnf = TaskConfig(
        task_name="boolean_cnf",
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

    cnf_scaling = [
        {"n_bits": 8, "n_clauses": 2, "clause_size": 3},
        {"n_bits": 8, "n_clauses": 3, "clause_size": 3},
        {"n_bits": 8, "n_clauses": 4, "clause_size": 3},
    ]

    cnf_results = run_task(
        task_config=task_config_cnf,
        data_generator=generate_cnf,
        scaling_params=cnf_scaling,
    )
    save_results(cnf_results, "benchmark_boolean_cnf.json")

    print("\n" + "=" * 60)
    print("BENCHMARK 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
