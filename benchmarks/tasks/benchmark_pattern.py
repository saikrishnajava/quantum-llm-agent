"""
Benchmark Task 4: Quantum-Separable Pattern Matching (Circuit-Level)
=====================================================================
Detect which template a noisy input matches, where templates are designed
to be separable by XOR sign structure but NOT by Euclidean distance.

Why quantum should help: CNOT entanglement in QuantumPatternMatcher computes
XOR-based similarity. Templates are constructed so cosine similarity fails
but XOR-sign similarity succeeds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import autograd
import autograd.numpy as anp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from classical.nn import Linear, AdamW, Module
from agents.reasoning.quantum_reasoning import QuantumPatternMatcher
from benchmarks.runner import SEEDS, save_results
from benchmarks.analysis.statistics import confidence_interval, paired_significance

import pennylane.numpy as pnp


class ClassicalPatternMatcher(Module):
    """Classical baseline: learned projection + cosine similarity."""

    def __init__(self, pattern_dim: int, n_templates: int, hidden_dim: int = 12):
        self.proj = Linear(pattern_dim, hidden_dim)
        self.template_weight = Linear(hidden_dim, n_templates)

    def forward(self, input_pattern: np.ndarray) -> np.ndarray:
        h = self.proj(input_pattern)
        h = anp.maximum(h, 0)
        logits = self.template_weight(h)
        shifted = logits - logits.max()
        exp_l = anp.exp(shifted)
        return exp_l / exp_l.sum()


def generate_quantum_separable_patterns(
    n_templates: int = 4,
    pattern_dim: int = 8,
    noise: float = 0.3,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate patterns separated by XOR sign structure.

    Templates differ by sign-flip masks such that their Euclidean distances
    are nearly equal, but XOR-sign distances differ maximally.
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test

    base = rng.randn(pattern_dim)
    base = base / np.linalg.norm(base)

    templates = []
    for t in range(n_templates):
        mask = np.array([(-1.0) ** ((t >> i) & 1) for i in range(pattern_dim)])
        templates.append(base * mask)
    templates = np.array(templates)

    template_ids = rng.randint(0, n_templates, n_total)
    X = templates[template_ids] + noise * rng.randn(n_total, pattern_dim)

    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = template_ids[:n_train]
    y_test = template_ids[n_train:]

    return X_train, y_train, X_test, y_test


def train_pattern_model(
    model,
    templates: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 20,
    lr: float = 0.01,
    is_quantum: bool = False,
) -> dict:
    """Train pattern matching model."""
    params = model.parameters()
    optimizer = AdamW(params, lr=lr)
    n_train = len(X_train)
    t_start = time.perf_counter()

    for epoch in range(epochs):
        indices = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_correct = 0
        n_samples = min(200, n_train)

        for i in indices[:n_samples]:
            input_pattern = X_train[i]
            label = int(y_train[i])

            def loss_fn(flat_params):
                idx = 0
                for p in params:
                    size = p.size
                    p[...] = flat_params[idx:idx + size].reshape(p.shape)
                    idx += size

                if is_quantum:
                    scores = []
                    for t in range(len(templates)):
                        sim = model(input_pattern, templates[t])
                        scores.append(anp.mean(sim))
                    scores_arr = anp.array(scores)
                    shifted = scores_arr - scores_arr.max()
                    exp_s = anp.exp(shifted)
                    probs = exp_s / exp_s.sum()
                else:
                    probs = model(input_pattern)
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
                    scores = []
                    for t in range(len(templates)):
                        sim = model(input_pattern, templates[t])
                        scores.append(float(np.mean(sim)))
                    pred = int(np.argmax(scores))
                else:
                    probs = model(input_pattern)
                    pred = int(np.argmax(probs))
                if pred == label:
                    n_correct += 1
            except Exception:
                continue

        if (epoch + 1) % 5 == 0:
            test_acc = _eval_pattern(model, templates, X_test, y_test, is_quantum)
            print(f"    Epoch {epoch+1}: loss={epoch_loss/n_samples:.4f}, "
                  f"train_acc={n_correct/n_samples:.1%}, test_acc={test_acc:.1%}")

    total_time = time.perf_counter() - t_start
    test_acc = _eval_pattern(model, templates, X_test, y_test, is_quantum)
    return {"final_test_acc": test_acc, "total_time_s": total_time}


def _eval_pattern(model, templates, X, y, is_quantum):
    correct = 0
    n = min(len(X), 200)
    for i in range(n):
        if is_quantum:
            scores = []
            for t in range(len(templates)):
                sim = model(X[i], templates[t])
                scores.append(float(np.mean(sim)))
            pred = int(np.argmax(scores))
        else:
            probs = model(X[i])
            pred = int(np.argmax(probs))
        if pred == int(y[i]):
            correct += 1
    return correct / n


def main():
    print("=" * 60)
    print("BENCHMARK 4: QUANTUM-SEPARABLE PATTERN MATCHING")
    print("=" * 60)
    print("Task: Match noisy input to templates (XOR-sign separated)")
    print("Quantum: QuantumPatternMatcher vs Classical MLP")
    print()

    scaling_params = [
        {"n_templates": 2, "pattern_dim": 8, "noise": 0.3},
        {"n_templates": 4, "pattern_dim": 8, "noise": 0.3},
        {"n_templates": 4, "pattern_dim": 8, "noise": 0.5},
    ]

    all_results = {}

    for sp in scaling_params:
        sp_key = f"templates={sp['n_templates']}_noise={sp['noise']}"
        print(f"\n--- {sp_key} ---")

        classical_accs = []
        quantum_accs = []

        for seed in SEEDS:
            np.random.seed(seed)
            X_train, y_train, X_test, y_test = generate_quantum_separable_patterns(
                seed=seed, **sp
            )
            n_templates = sp["n_templates"]
            pattern_dim = sp["pattern_dim"]

            rng = np.random.RandomState(seed)
            base = rng.randn(pattern_dim)
            base = base / np.linalg.norm(base)
            templates = np.array([
                base * np.array([(-1.0) ** ((t >> i) & 1) for i in range(pattern_dim)])
                for t in range(n_templates)
            ])

            print(f"  Seed {seed} — Classical:")
            cpm = ClassicalPatternMatcher(pattern_dim, n_templates, hidden_dim=12)
            c_result = train_pattern_model(
                cpm, templates, X_train, y_train, X_test, y_test,
                epochs=20, lr=0.01, is_quantum=False,
            )
            classical_accs.append(c_result["final_test_acc"])

            print(f"  Seed {seed} — Quantum:")
            qpm = QuantumPatternMatcher(n_qubits=6, n_layers=1)
            q_result = train_pattern_model(
                qpm, templates, X_train, y_train, X_test, y_test,
                epochs=20, lr=0.01, is_quantum=True,
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

        print(f"\n  Classical: {ci_c['mean']*100:.1f}%")
        print(f"  Quantum:   {ci_q['mean']*100:.1f}%")
        print(f"  p-value: {sig['p_value']:.4f}, Cohen's d: {sig['cohens_d']:.2f}")

    save_results({"task": "pattern", "results": all_results}, "benchmark_pattern.json")
    print("\n" + "=" * 60)
    print("BENCHMARK 4 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
