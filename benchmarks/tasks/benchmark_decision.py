"""
Benchmark Task 3: Multi-Option Decision Under Constraints (Circuit-Level)
==========================================================================
Standalone benchmark of QuantumDecisionCircuit vs classical MLP.
Given context + options, select the option best satisfying constraints.

Why quantum should help: Decision circuit encodes all options in superposition
and evaluates compatibility via entanglement. Classical MLP learns from scratch.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import autograd
import autograd.numpy as anp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from classical.nn import Linear, Parameter, AdamW, Module
from agents.reasoning.quantum_reasoning import QuantumDecisionCircuit
from benchmarks.runner import SEEDS, RESULTS_DIR, save_results
from benchmarks.analysis.statistics import confidence_interval, paired_significance

import pennylane.numpy as pnp


class ClassicalDecisionMLP(Module):
    """Classical MLP baseline matching quantum decision circuit param count."""

    def __init__(self, input_dim: int, n_options: int, hidden_dim: int = 16):
        self.fc1 = Linear(input_dim * 2, hidden_dim)
        self.fc2 = Linear(hidden_dim, n_options)

    def forward(self, context: np.ndarray, options: np.ndarray) -> np.ndarray:
        combined = anp.concatenate([context, options])
        h = anp.maximum(self.fc1(combined), 0)  # ReLU
        logits = self.fc2(h)
        shifted = logits - logits.max()
        exp_l = anp.exp(shifted)
        return exp_l / exp_l.sum()


def generate_decision_task(
    n_options: int = 4,
    context_dim: int = 8,
    n_constraints: int = 3,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate decision task: context + options → best option under constraints.

    Returns (X_train, y_train, X_test, y_test) where:
      X_train: (n_samples, context_dim + context_dim) — context + flattened mean options
      y_train: (n_samples,) — integer label of best option
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test

    W_constraints = rng.randn(n_constraints, context_dim)

    X_ctx = rng.randn(n_total, context_dim)
    X_opts = rng.randn(n_total, n_options, context_dim)

    labels = np.zeros(n_total, dtype=int)
    for i in range(n_total):
        scores = []
        ctx_proj = X_ctx[i] @ W_constraints.T
        for j in range(n_options):
            opt_proj = X_opts[i, j] @ W_constraints.T
            satisfaction = np.sum(np.sign(ctx_proj) == np.sign(opt_proj))
            scores.append(satisfaction + rng.normal(0, 0.1))
        labels[i] = np.argmax(scores)

    X_combined = np.concatenate([X_ctx, X_opts.mean(axis=1)], axis=1)

    X_train = X_combined[:n_train]
    X_test = X_combined[n_train:]
    y_train = labels[:n_train]
    y_test = labels[n_train:]

    return X_train, y_train, X_test, y_test


def train_circuit_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    context_dim: int,
    n_options: int,
    epochs: int = 20,
    lr: float = 0.01,
    is_quantum: bool = False,
) -> dict:
    """Train a circuit-level model (quantum or classical MLP)."""
    params = model.parameters()
    optimizer = AdamW(params, lr=lr)
    n_train = len(X_train)
    t_start = time.perf_counter()

    for epoch in range(epochs):
        indices = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_correct = 0

        for i in indices[:min(200, n_train)]:
            ctx = X_train[i, :context_dim]
            opts = X_train[i, context_dim:]
            label = int(y_train[i])

            def loss_fn(flat_params):
                idx = 0
                for p in params:
                    size = p.size
                    p[...] = flat_params[idx:idx + size].reshape(p.shape)
                    idx += size
                if is_quantum:
                    probs = model(ctx, opts)
                else:
                    probs = model(ctx, opts)
                return -anp.log(probs[label] + 1e-10)

            flat = np.concatenate([np.array(p).flatten() for p in params])
            grad_fn = autograd.grad(loss_fn)
            try:
                grads = grad_fn(flat)
                loss_val = loss_fn(flat)
                epoch_loss += float(loss_val)

                idx = 0
                grad_list = []
                for p in params:
                    size = p.size
                    grad_list.append(np.array(grads[idx:idx + size]).reshape(p.shape))
                    idx += size
                optimizer.step(grad_list)

                if is_quantum:
                    probs = model(ctx, opts)
                else:
                    probs = model(ctx, opts)
                if int(np.argmax(probs)) == label:
                    n_correct += 1
            except Exception:
                continue

        if (epoch + 1) % 5 == 0:
            test_acc = _evaluate_circuit(model, X_test, y_test, context_dim, is_quantum)
            train_acc = n_correct / min(200, n_train)
            print(f"    Epoch {epoch+1}: loss={epoch_loss/min(200,n_train):.4f}, "
                  f"train_acc={train_acc:.1%}, test_acc={test_acc:.1%}")

    total_time = time.perf_counter() - t_start
    test_acc = _evaluate_circuit(model, X_test, y_test, context_dim, is_quantum)
    return {"final_test_acc": test_acc, "total_time_s": total_time}


def _evaluate_circuit(model, X, y, context_dim, is_quantum):
    correct = 0
    n = min(len(X), 200)
    for i in range(n):
        ctx = X[i, :context_dim]
        opts = X[i, context_dim:]
        if is_quantum:
            probs = model(ctx, opts)
        else:
            probs = model(ctx, opts)
        if int(np.argmax(probs)) == int(y[i]):
            correct += 1
    return correct / n


def main():
    print("=" * 60)
    print("BENCHMARK 3: MULTI-OPTION DECISION (CIRCUIT-LEVEL)")
    print("=" * 60)
    print("Task: Select best option under constraints")
    print("Quantum: QuantumDecisionCircuit vs Classical MLP")
    print()

    scaling_params = [
        {"n_options": 4, "context_dim": 8, "n_constraints": 2},
        {"n_options": 4, "context_dim": 8, "n_constraints": 3},
        {"n_options": 4, "context_dim": 8, "n_constraints": 4},
    ]

    all_results = {}

    for sp in scaling_params:
        sp_key = f"opts={sp['n_options']}_constraints={sp['n_constraints']}"
        print(f"\n--- {sp_key} ---")

        classical_accs = []
        quantum_accs = []

        for seed in SEEDS:
            np.random.seed(seed)
            X_train, y_train, X_test, y_test = generate_decision_task(seed=seed, **sp)

            n_options = sp["n_options"]
            context_dim = sp["context_dim"]

            print(f"  Seed {seed} — Classical MLP:")
            mlp = ClassicalDecisionMLP(context_dim, n_options, hidden_dim=16)
            c_result = train_circuit_model(
                mlp, X_train, y_train, X_test, y_test,
                context_dim, n_options, epochs=20, lr=0.01, is_quantum=False,
            )
            classical_accs.append(c_result["final_test_acc"])

            print(f"  Seed {seed} — Quantum Decision Circuit:")
            qdc = QuantumDecisionCircuit(n_qubits=6, n_layers=2)
            q_result = train_circuit_model(
                qdc, X_train, y_train, X_test, y_test,
                context_dim, n_options, epochs=20, lr=0.01, is_quantum=True,
            )
            quantum_accs.append(q_result["final_test_acc"])

        ci_c = confidence_interval(classical_accs)
        ci_q = confidence_interval(quantum_accs)
        sig = paired_significance(classical_accs, quantum_accs)

        all_results[sp_key] = {
            "scaling_params": sp,
            "classical": {"mean_acc": ci_c["mean"], "ci_95": [ci_c["ci_low"], ci_c["ci_high"]]},
            "quantum": {"mean_acc": ci_q["mean"], "ci_95": [ci_q["ci_low"], ci_q["ci_high"]]},
            "significance": sig,
        }

        print(f"\n  Classical: {ci_c['mean']*100:.1f}% [{ci_c['ci_low']*100:.1f}%, {ci_c['ci_high']*100:.1f}%]")
        print(f"  Quantum:   {ci_q['mean']*100:.1f}% [{ci_q['ci_low']*100:.1f}%, {ci_q['ci_high']*100:.1f}%]")
        print(f"  p-value: {sig['p_value']:.4f}, Cohen's d: {sig['cohens_d']:.2f}")

    save_results({"task": "decision", "results": all_results}, "benchmark_decision.json")
    print("\n" + "=" * 60)
    print("BENCHMARK 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
