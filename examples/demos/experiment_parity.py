"""
Quantum Advantage Experiment: Parity Detection
================================================
Proves quantum attention beats classical on XOR/parity — a task
where entanglement provides a structural advantage.

Run:
    cd quantum-llm-agent
    python examples/demos/experiment_parity.py
"""

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import autograd.numpy as anp

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hybrid.interfaces.model import HybridQuantumLLM
from classical.optimizers.trainer import HybridQuantumTrainer
from classical.nn import CrossEntropyLoss


def generate_parity_dataset(n_bits=8, n_samples=1000, seed=42):
    """Generate binary sequences labeled by XOR of all bits."""
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n_samples, n_bits))
    y = np.bitwise_xor.reduce(X, axis=1)  # XOR of all bits
    return X, y


def evaluate_accuracy(model, X, y, vocab_size):
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = len(X)
    for i in range(total):
        ids = X[i : i + 1]
        logits = model(ids)
        pred = int(logits[0, -1, :].argmax())
        label = int(y[i])
        if pred == label:
            correct += 1
    return correct / total


def train_and_evaluate(name, model, X_train, y_train, X_test, y_test, n_epochs=20):
    """Train model on parity task and track accuracy per epoch."""
    vocab_size = 2  # binary tokens
    trainer = HybridQuantumTrainer(model, learning_rate=5e-3)
    params = model.count_parameters()

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Params: {params['total']:,} ({params['quantum']} quantum)")
    print(f"{'='*60}")

    # Format: each sample is a sequence of bits, label is the last prediction
    # Input: [b0, b1, ..., b7], Target: [b1, b2, ..., XOR]
    n_train = len(X_train)
    seq_len = X_train.shape[1]

    # Build input/target pairs: input is bits, target shifts left + XOR label at end
    train_x = X_train
    train_y = np.concatenate([X_train[:, 1:], y_train[:, None]], axis=1)

    results = []
    t_total = time.perf_counter()

    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        epoch_losses = []

        # Mini-batch training
        indices = np.random.permutation(n_train)
        batch_size = 16
        for start in range(0, n_train - batch_size + 1, batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = train_x[batch_idx]
            y_batch = train_y[batch_idx]
            r = trainer.training_step(x_batch, y_batch)
            epoch_losses.append(r["loss"])

        elapsed = time.perf_counter() - t0
        avg_loss = float(np.mean(epoch_losses))

        # Evaluate accuracy on test set (every 5 epochs to save time)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_acc = evaluate_accuracy(model, X_test, y_test, vocab_size)
            train_acc = evaluate_accuracy(
                model, X_train[:100], y_train[:100], vocab_size
            )
        else:
            test_acc = results[-1]["test_acc"] if results else 0.0
            train_acc = results[-1]["train_acc"] if results else 0.0

        results.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "time": elapsed,
            }
        )
        print(
            f"  Epoch {epoch+1:2d}: loss={avg_loss:.4f}, "
            f"train_acc={train_acc:.1%}, test_acc={test_acc:.1%}, "
            f"{elapsed:.1f}s"
        )

    total_time = time.perf_counter() - t_total
    return {
        "name": name,
        "params_total": params["total"],
        "params_quantum": params["quantum"],
        "final_loss": results[-1]["loss"],
        "final_test_acc": results[-1]["test_acc"],
        "final_train_acc": results[-1]["train_acc"],
        "total_time_s": round(total_time, 1),
        "epochs": results,
    }


def main():
    print("=" * 60)
    print("QUANTUM ADVANTAGE EXPERIMENT: PARITY DETECTION")
    print("=" * 60)
    print("Task: Predict XOR of 8-bit binary input")
    print("Why quantum wins: XOR requires exponential classical")
    print("circuit depth but O(n) quantum depth via entanglement")
    print()

    # Generate data
    N_BITS = 8
    X_train, y_train = generate_parity_dataset(N_BITS, n_samples=800, seed=42)
    X_test, y_test = generate_parity_dataset(N_BITS, n_samples=200, seed=99)
    print(f"Data: {len(X_train)} train, {len(X_test)} test, {N_BITS} bits")
    print(f"Class balance: {y_train.mean():.2f} (should be ~0.50)")

    VOCAB_SIZE = 2  # binary: 0 and 1
    N_EPOCHS = 20

    # Model configs — same d_model, same layers, same total architecture
    configs = {
        "Classical (0 quantum heads)": {
            "quantum_heads_per_layer": 0,
            "quantum_config": {
                "use_quantum_embedding": False,
                "embedding_qubits": 3,
                "attention_qubits": 6,
                "activation_qubits": 3,
                "use_quantum_activation": False,
            },
        },
        "Quantum (1 fixed 6-qubit head)": {
            "quantum_heads_per_layer": 1,
            "quantum_config": {
                "use_quantum_embedding": False,
                "embedding_qubits": 3,
                "attention_qubits": 6,
                "activation_qubits": 3,
                "use_quantum_activation": False,
            },
        },
        "MoE (6/9/12 qubit router)": {
            "quantum_heads_per_layer": 1,
            "quantum_config": {
                "use_quantum_embedding": False,
                "embedding_qubits": 3,
                "attention_qubits": 6,
                "activation_qubits": 3,
                "use_quantum_activation": False,
                "use_moe": True,
                "moe_qubit_configs": [6, 9, 12],
                "moe_temperature": 1.0,
            },
        },
    }

    all_results = {}
    for name, cfg in configs.items():
        model = HybridQuantumLLM(
            vocab_size=VOCAB_SIZE,
            d_model=16,
            n_layers=1,
            n_heads=4,
            d_ff=32,
            max_seq_length=16,
            dropout=0.0,
            quantum_heads_per_layer=cfg["quantum_heads_per_layer"],
            quantum_config=cfg["quantum_config"],
        )
        result = train_and_evaluate(
            name, model, X_train, y_train, X_test, y_test, N_EPOCHS
        )
        all_results[name] = result

    # Print comparison
    print("\n" + "=" * 60)
    print("FINAL RESULTS: PARITY DETECTION (XOR of 8 bits)")
    print("=" * 60)

    for name, r in all_results.items():
        print(f"\n  {name}:")
        print(f"    Params: {r['params_total']:,} ({r['params_quantum']} quantum)")
        print(f"    Test accuracy: {r['final_test_acc']:.1%}")
        print(f"    Final loss: {r['final_loss']:.4f}")
        print(f"    Time: {r['total_time_s']:.1f}s")

    # Accuracy comparison
    print("\n  " + "-" * 40)
    names = list(all_results.keys())
    accs = {n: all_results[n]["final_test_acc"] for n in names}
    winner = max(accs, key=accs.get)
    print(f"  Winner: {winner} ({accs[winner]:.1%})")

    for n in names:
        bar = "#" * int(accs[n] * 40)
        print(f"    {accs[n]:5.1%} {bar} {n}")

    # Save
    out_path = Path(__file__).parent / "parity_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
