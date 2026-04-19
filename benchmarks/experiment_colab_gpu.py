#!/usr/bin/env python3
"""
Colab GPU Experiment: Circuit-Level Qubit Scaling
==================================================
Self-contained script for Google Colab with T4/A100 GPU.
Tests the QuantumAttentionCircuit in ISOLATION (Option B) at 12, 15, 18 qubits
on a classification task designed to exploit CNOT-based Q-K similarity.

Setup (run in Colab cell):
    !pip install pennylane pennylane-lightning pennylane-lightning[gpu] autograd scipy

Usage:
    python benchmarks/experiment_colab_gpu.py

This script is self-contained — it embeds all needed circuit logic.
No imports from the main project required (works standalone on Colab).
"""

from __future__ import annotations

import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import autograd
import autograd.numpy as anp
import pennylane as qml
import pennylane.numpy as pnp


# ======================================================================
# Device Selection
# ======================================================================

def select_device(n_qubits: int):
    """Pick best available backend. GPU first, then C++ CPU, then default."""
    for backend in ["lightning.gpu", "lightning.qubit", "default.qubit"]:
        try:
            dev = qml.device(backend, wires=n_qubits)
            print(f"  [{n_qubits}q] Using backend: {backend}")
            return dev, backend
        except Exception:
            continue
    dev = qml.device("default.qubit", wires=n_qubits)
    print(f"  [{n_qubits}q] Using backend: default.qubit (fallback)")
    return dev, "default.qubit"


def get_diff_method(backend: str) -> str:
    if "lightning" in backend:
        return "adjoint"
    return "backprop"


# ======================================================================
# Quantum Attention Circuit (self-contained)
# ======================================================================

def build_attention_circuit(n_qubits: int, n_layers: int = 2):
    """Build the Q-K→V quantum attention circuit as a PennyLane QNode."""
    assert n_qubits % 3 == 0
    qpr = n_qubits // 3
    q_wires = list(range(0, qpr))
    k_wires = list(range(qpr, 2 * qpr))
    v_wires = list(range(2 * qpr, 3 * qpr))

    dev, backend = select_device(n_qubits)
    diff_method = get_diff_method(backend)

    @qml.qnode(dev, interface="autograd", diff_method=diff_method)
    def circuit(q_features, k_features, v_features, params):
        qml.AmplitudeEmbedding(q_features, wires=q_wires, normalize=True)
        qml.AmplitudeEmbedding(k_features, wires=k_wires, normalize=True)
        qml.AmplitudeEmbedding(v_features, wires=v_wires, normalize=True)

        idx = 0
        for layer in range(n_layers):
            for i in range(qpr):
                qml.CNOT(wires=[q_wires[i], k_wires[i]])
            for i in range(qpr):
                qml.CNOT(wires=[k_wires[i], v_wires[i]])
                qml.RY(params[idx], wires=v_wires[i])
                qml.CNOT(wires=[k_wires[i], v_wires[i]])
                idx += 1
            for i in range(qpr):
                qml.RY(params[idx], wires=v_wires[i])
                qml.RZ(params[idx + 1], wires=v_wires[i])
                idx += 2
            for i in range(qpr - 1):
                qml.CNOT(wires=[v_wires[i], v_wires[i + 1]])
            for i in range(qpr):
                qml.CNOT(wires=[q_wires[i], k_wires[i]])

        return [qml.expval(qml.PauliZ(w)) for w in v_wires]

    n_params = n_layers * (qpr + qpr * 2)
    params = pnp.array(np.random.randn(n_params) * 0.5, requires_grad=True)

    return circuit, params, {"n_qubits": n_qubits, "qpr": qpr, "backend": backend}


# ======================================================================
# Classical Baseline (matched capacity)
# ======================================================================

def build_classical_model(input_dim: int, output_dim: int, n_quantum_params: int):
    """Classical MLP with approximately matched parameter count."""
    hidden = max(output_dim, n_quantum_params // 3)
    W1 = pnp.array(np.random.randn(input_dim * 3, hidden) * 0.1, requires_grad=True)
    b1 = pnp.array(np.zeros(hidden), requires_grad=True)
    W2 = pnp.array(np.random.randn(hidden, output_dim) * 0.1, requires_grad=True)
    b2 = pnp.array(np.zeros(output_dim), requires_grad=True)
    params = [W1, b1, W2, b2]
    total_params = sum(p.size for p in params)
    print(f"  Classical MLP: {total_params} params (hidden={hidden})")
    return params


def classical_forward(q, k, v, params):
    """Classical attention-like: concat(q,k,v) → MLP → output."""
    W1, b1, W2, b2 = params
    x = anp.concatenate([q, k, v])
    h = anp.maximum(x @ W1 + b1, 0)  # ReLU
    return anp.tanh(W2.T @ h + b2)  # Bounded output like PauliZ


# ======================================================================
# XOR-Sign Classification Task
# ======================================================================

def generate_xor_sign_task(dim: int, n_samples: int, seed: int = 42):
    """
    Task: Given Q and K vectors, predict XOR of their sign patterns.
    Label = XOR(sign(Q[i]) == sign(K[i])) for a hidden subset of dimensions.

    This task directly exploits the CNOT(Q,K) mechanism: when Q≈K (same signs),
    CNOT produces |0⟩ on K-register. The quantum circuit should learn this faster.
    """
    rng = np.random.RandomState(seed)

    n_hidden = min(dim // 2, 4)
    hidden_dims = sorted(rng.choice(dim, n_hidden, replace=False).tolist())

    Q = rng.randn(n_samples, dim).astype(np.float64)
    K = rng.randn(n_samples, dim).astype(np.float64)
    V = rng.randn(n_samples, dim).astype(np.float64)

    agreements = np.array([(np.sign(Q[:, i]) == np.sign(K[:, i])).astype(int)
                           for i in hidden_dims])
    labels = np.bitwise_xor.reduce(agreements, axis=0)

    return Q, K, V, labels, hidden_dims


# ======================================================================
# Training Loop
# ======================================================================

def train_quantum(circuit, params, Q_train, K_train, V_train, y_train,
                  epochs=30, lr=0.01, batch_size=32):
    """Train quantum circuit via parameter-shift or adjoint gradients."""
    n_train = len(Q_train)
    dim = Q_train.shape[1]

    for epoch in range(epochs):
        indices = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, min(n_train, 200) - batch_size + 1, batch_size):
            batch_idx = indices[start:start + batch_size]

            def loss_fn(p):
                total_loss = 0.0
                for i in batch_idx:
                    q = Q_train[i]
                    k = K_train[i]
                    v = V_train[i]
                    out = circuit(q, k, v, p)
                    score = anp.mean(out)
                    pred = (score + 1) / 2  # map [-1,1] to [0,1]
                    target = float(y_train[i])
                    total_loss = total_loss - (target * anp.log(pred + 1e-10) +
                                               (1 - target) * anp.log(1 - pred + 1e-10))
                return total_loss / len(batch_idx)

            grad_fn = autograd.grad(loss_fn)
            grads = grad_fn(params)
            loss_val = float(loss_fn(params))
            params = params - lr * np.array(grads)
            params = pnp.array(params, requires_grad=True)
            epoch_loss += loss_val
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = evaluate_quantum(circuit, params, Q_train[:200], K_train[:200],
                                   V_train[:200], y_train[:200])
            print(f"    Epoch {epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}, train_acc={acc:.1%}")

    return params


def train_classical(params, Q_train, K_train, V_train, y_train,
                    epochs=30, lr=0.01, batch_size=32):
    """Train classical MLP baseline."""
    n_train = len(Q_train)

    for epoch in range(epochs):
        indices = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, min(n_train, 200) - batch_size + 1, batch_size):
            batch_idx = indices[start:start + batch_size]

            def loss_fn(flat):
                sizes = [p.size for p in params]
                shapes = [p.shape for p in params]
                ps = []
                idx = 0
                for size, shape in zip(sizes, shapes):
                    ps.append(flat[idx:idx+size].reshape(shape))
                    idx += size

                total_loss = 0.0
                for i in batch_idx:
                    out = classical_forward(Q_train[i], K_train[i], V_train[i], ps)
                    score = anp.mean(out)
                    pred = (score + 1) / 2
                    target = float(y_train[i])
                    total_loss = total_loss - (target * anp.log(pred + 1e-10) +
                                               (1 - target) * anp.log(1 - pred + 1e-10))
                return total_loss / len(batch_idx)

            flat = np.concatenate([np.array(p).flatten() for p in params])
            grad_fn = autograd.grad(loss_fn)
            grads = grad_fn(flat)
            flat = flat - lr * np.array(grads)

            idx = 0
            for i, p in enumerate(params):
                size = p.size
                params[i] = pnp.array(flat[idx:idx+size].reshape(p.shape), requires_grad=True)
                idx += size

            epoch_loss += float(loss_fn(flat))
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = evaluate_classical(params, Q_train[:200], K_train[:200],
                                     V_train[:200], y_train[:200])
            print(f"    Epoch {epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}, train_acc={acc:.1%}")

    return params


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_quantum(circuit, params, Q, K, V, y):
    correct = 0
    for i in range(len(Q)):
        out = circuit(Q[i], K[i], V[i], params)
        score = float(np.mean(out))
        pred = 1 if score > 0 else 0
        if pred == int(y[i]):
            correct += 1
    return correct / len(Q)


def evaluate_classical(params, Q, K, V, y):
    correct = 0
    for i in range(len(Q)):
        out = classical_forward(Q[i], K[i], V[i], params)
        score = float(np.mean(out))
        pred = 1 if score > 0 else 0
        if pred == int(y[i]):
            correct += 1
    return correct / len(Q)


# ======================================================================
# Main Experiment
# ======================================================================

def run_experiment(n_qubits: int, n_train: int = 400, n_test: int = 100,
                   epochs: int = 30, seed: int = 42):
    """Run one (quantum vs classical) comparison at given qubit count."""
    dim = 2 ** (n_qubits // 3)  # register dimension
    print(f"\n{'='*50}")
    print(f"  {n_qubits} QUBITS (register_dim={dim})")
    print(f"{'='*50}")

    np.random.seed(seed)
    Q_train, K_train, V_train, y_train, hidden = generate_xor_sign_task(
        dim, n_train, seed)
    Q_test, K_test, V_test, y_test, _ = generate_xor_sign_task(
        dim, n_test, seed + 1000)
    # Use same hidden dims for test
    agreements = np.array([(np.sign(Q_test[:, i]) == np.sign(K_test[:, i])).astype(int)
                           for i in hidden])
    y_test = np.bitwise_xor.reduce(agreements, axis=0)

    print(f"  Data: {n_train} train, {n_test} test, dim={dim}")
    print(f"  Hidden XOR dims: {hidden}")
    print(f"  Class balance: {y_train.mean():.2f}")

    # --- Quantum ---
    print(f"\n  Training QUANTUM ({n_qubits}q):")
    t0 = time.perf_counter()
    circuit, q_params, info = build_attention_circuit(n_qubits, n_layers=2)
    n_q_params = q_params.size
    print(f"  Quantum params: {n_q_params}")
    q_params = train_quantum(circuit, q_params, Q_train, K_train, V_train, y_train,
                             epochs=epochs, lr=0.01)
    q_time = time.perf_counter() - t0
    q_test_acc = evaluate_quantum(circuit, q_params, Q_test, K_test, V_test, y_test)
    print(f"  Quantum test acc: {q_test_acc:.1%} (time: {q_time:.1f}s)")

    # --- Classical ---
    print(f"\n  Training CLASSICAL (matched params):")
    t0 = time.perf_counter()
    c_params = build_classical_model(dim, n_qubits // 3, n_q_params)
    c_params = train_classical(c_params, Q_train, K_train, V_train, y_train,
                               epochs=epochs, lr=0.01)
    c_time = time.perf_counter() - t0
    c_test_acc = evaluate_classical(c_params, Q_test, K_test, V_test, y_test)
    print(f"  Classical test acc: {c_test_acc:.1%} (time: {c_time:.1f}s)")

    advantage = q_test_acc - c_test_acc
    print(f"\n  ADVANTAGE: {advantage*100:+.1f}% {'★' if advantage > 0.05 else ''}")

    return {
        "n_qubits": n_qubits,
        "quantum_acc": q_test_acc,
        "classical_acc": c_test_acc,
        "advantage": advantage,
        "quantum_time_s": q_time,
        "classical_time_s": c_time,
        "quantum_params": n_q_params,
        "backend": info["backend"],
        "seed": seed,
    }


def main():
    print("=" * 60)
    print("COLAB GPU EXPERIMENT: CIRCUIT-LEVEL QUBIT SCALING")
    print("=" * 60)
    print("Task: XOR-sign classification (directly exploits CNOT similarity)")
    print("Comparing: Quantum attention circuit vs Classical MLP")
    print()

    QUBIT_CONFIGS = [12, 15, 18]
    SEEDS = [42, 99, 137]
    EPOCHS = 30

    all_results = []

    for n_qubits in QUBIT_CONFIGS:
        for seed in SEEDS:
            try:
                result = run_experiment(n_qubits, n_train=400, n_test=100,
                                       epochs=EPOCHS, seed=seed)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR at {n_qubits}q seed={seed}: {e}")
                continue

    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Qubits':<8} {'Quantum Acc':<14} {'Classical Acc':<14} {'Advantage':<12} {'Time (Q)':<10}")
    print("-" * 60)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[r["n_qubits"]].append(r)

    for n_q in sorted(grouped.keys()):
        results = grouped[n_q]
        q_accs = [r["quantum_acc"] for r in results]
        c_accs = [r["classical_acc"] for r in results]
        advs = [r["advantage"] for r in results]
        times = [r["quantum_time_s"] for r in results]
        print(f"{n_q:<8} {np.mean(q_accs)*100:.1f}% ±{np.std(q_accs)*100:.1f}  "
              f"{np.mean(c_accs)*100:.1f}% ±{np.std(c_accs)*100:.1f}  "
              f"{np.mean(advs)*100:+.1f}%       {np.mean(times):.0f}s")

    # Save
    output = {
        "experiment": "colab_gpu_circuit_scaling",
        "task": "xor_sign_classification",
        "qubit_configs": QUBIT_CONFIGS,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "results": all_results,
    }
    with open("colab_gpu_results.json", "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print("\nResults saved to colab_gpu_results.json")

    # Verdict
    print("\n" + "=" * 60)
    print("SCALING TREND")
    print("=" * 60)
    qubit_order = sorted(grouped.keys())
    if len(qubit_order) >= 2:
        first_adv = np.mean([r["advantage"] for r in grouped[qubit_order[0]]])
        last_adv = np.mean([r["advantage"] for r in grouped[qubit_order[-1]]])
        if last_adv > first_adv + 0.03:
            print("✓ Advantage GROWS with qubit count → scaling thesis alive!")
        elif last_adv > 0.05:
            print("✓ Advantage exists at higher qubits (even if not growing)")
        else:
            print("✗ No clear scaling advantage observed")


if __name__ == "__main__":
    main()
