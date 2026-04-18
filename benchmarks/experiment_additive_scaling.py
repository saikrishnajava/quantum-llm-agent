#!/usr/bin/env python3
"""
Experiment: Additive Quantum + Qubit Scaling
=============================================
Tests two hypotheses:
  1. ADDITIVE: Keep all 4 classical heads + add quantum as extra capacity
     (vs current REPLACEMENT approach that removes a classical head)
  2. SCALING: Does advantage grow with qubit count? (6 → 9 → 12)

Both tested on full 8-bit XOR parity — the one task that showed quantum signal.

Usage:
    python benchmarks/experiment_additive_scaling.py
    python benchmarks/experiment_additive_scaling.py --parallel

Expected runtime (12 cores, parallel): ~2-3 hours
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import autograd.numpy as anp

from classical.nn import Module, Linear, Dropout, Softmax, Parameter, LayerNorm
from classical.optimizers.trainer import HybridQuantumTrainer
from hybrid.interfaces.model import HybridQuantumLLM
from quantum.circuits.core import QuantumAttentionCircuit
from benchmarks.analysis.statistics import confidence_interval, paired_significance
from benchmarks.runner import SEEDS, RESULTS_DIR, _json_default

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ======================================================================
# Additive Quantum Attention Module
# ======================================================================

class AdditiveQuantumAttention(Module):
    """
    Full classical attention (all n_heads) PLUS quantum as additive side-channel.

    Unlike the standard QuantumMultiHeadAttention which replaces classical heads,
    this keeps all classical heads intact and adds quantum information via a
    gated residual: output = classical_output + gate * quantum_contribution.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        quantum_qubits: int = 6,
        n_quantum_circuits: int = 1,
        dropout: float = 0.0,
        max_quantum_seq: int = 16,
    ):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.max_quantum_seq = max_quantum_seq

        # Full classical attention (ALL heads, no replacement)
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

        # Quantum side-channel
        assert quantum_qubits % 3 == 0
        self.quantum_qubits = quantum_qubits
        register_dim = quantum_qubits // 3

        self.quantum_circuits = [
            QuantumAttentionCircuit(quantum_qubits, n_layers=2)
            for _ in range(n_quantum_circuits)
        ]
        self.n_quantum_circuits = n_quantum_circuits

        # Project input to quantum circuit dimensions
        self.q_quantum_proj = Linear(d_model, register_dim)
        self.k_quantum_proj = Linear(d_model, register_dim)
        self.v_quantum_proj = Linear(d_model, register_dim)

        # Project quantum output back to d_model and learn a gate
        self.quantum_up_proj = Linear(register_dim * n_quantum_circuits, d_model)
        self.gate = Parameter(np.zeros(1))  # Learned scalar gate, starts at 0

        self.softmax = Softmax(axis=-1)
        self.dropout = Dropout(dropout)

    def forward(self, x: np.ndarray, attention_mask=None) -> np.ndarray:
        B, S, _ = x.shape
        H, D = self.n_heads, self.head_dim

        # --- Full classical attention (unchanged) ---
        q = self.q_proj(x).reshape(B, S, H, D).transpose((0, 2, 1, 3))
        k = self.k_proj(x).reshape(B, S, H, D).transpose((0, 2, 1, 3))
        v = self.v_proj(x).reshape(B, S, H, D).transpose((0, 2, 1, 3))

        scores = (q @ k.transpose((0, 1, 3, 2))) / self.scale
        if attention_mask is not None:
            scores = anp.where(attention_mask, scores, -1e9)
        weights = self.softmax(scores)
        classical_out = (weights @ v).transpose((0, 2, 1, 3)).reshape(B, S, self.d_model)
        classical_out = self.out_proj(classical_out)

        # --- Quantum side-channel (additive) ---
        max_qs = min(S, self.max_quantum_seq)
        q_qc = self.q_quantum_proj(x)
        k_qc = self.k_quantum_proj(x)
        v_qc = self.v_quantum_proj(x)

        quantum_outputs = []
        for b in range(B):
            seq_out = []
            for s in range(max_qs):
                circuit_outs = []
                for circuit in self.quantum_circuits:
                    qc_out = circuit(q_qc[b, s], k_qc[b, s], v_qc[b, s])
                    circuit_outs.append(qc_out)
                seq_out.append(anp.concatenate(circuit_outs))
            if S > max_qs:
                zero_pad = anp.zeros(self.quantum_circuits[0].qubits_per_register * self.n_quantum_circuits)
                for s in range(S - max_qs):
                    seq_out.append(zero_pad)
            quantum_outputs.append(anp.stack(seq_out))

        quantum_tensor = anp.stack(quantum_outputs)
        quantum_contribution = self.quantum_up_proj(quantum_tensor)

        # Sigmoid gate (starts near 0.5, learns how much quantum to mix in)
        gate_value = 1.0 / (1.0 + anp.exp(-self.gate[0]))
        output = classical_out + gate_value * quantum_contribution

        return self.dropout(output)


# ======================================================================
# Model Builder
# ======================================================================

class AdditiveQuantumLLM(Module):
    """LLM with additive quantum attention (quantum adds on top of classical)."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, max_seq_length,
                 quantum_qubits=6, n_quantum_circuits=1):
        from classical.nn import Embedding, GELU

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        self.embedding = Embedding(vocab_size, d_model)
        self.norm1 = LayerNorm(d_model)
        self.attention = AdditiveQuantumAttention(
            d_model=d_model,
            n_heads=n_heads,
            quantum_qubits=quantum_qubits,
            n_quantum_circuits=n_quantum_circuits,
            max_quantum_seq=max_seq_length,
        )
        self.norm2 = LayerNorm(d_model)
        self.ff1 = Linear(d_model, d_ff)
        self.gelu = GELU()
        self.ff2 = Linear(d_ff, d_model)
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        residual = x
        x = residual + self.attention(self.norm1(x))
        residual = x
        x = residual + self.ff2(self.gelu(self.ff1(self.norm2(x))))
        x = self.ln_f(x)
        return self.lm_head(x)

    def count_parameters(self):
        classical, quantum = 0, 0
        for name, p in self.named_parameters():
            n = p.size
            if "quantum" in name or name.endswith(".params"):
                quantum += n
            else:
                classical += n
        return {"classical": classical, "quantum": quantum, "total": classical + quantum}


# ======================================================================
# Data Generation
# ======================================================================

def generate_parity_dataset(n_bits=8, n_samples=1000, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n_samples, n_bits))
    y = np.bitwise_xor.reduce(X, axis=1)
    return X, y


# ======================================================================
# Training
# ======================================================================

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs=20, lr=5e-3):
    """Train and return final metrics."""
    trainer = HybridQuantumTrainer(model, learning_rate=lr)
    n_train = len(X_train)
    train_x = X_train
    train_y = np.concatenate([X_train[:, 1:], y_train[:, None]], axis=1)
    test_y_full = np.concatenate([X_test[:, 1:], y_test[:, None]], axis=1)

    t_start = time.perf_counter()
    batch_size = 16

    for epoch in range(epochs):
        indices = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train - batch_size + 1, batch_size):
            batch_idx = indices[start:start + batch_size]
            r = trainer.training_step(train_x[batch_idx], train_y[batch_idx])
            epoch_loss += r["loss"]
            n_batches += 1

    # Final evaluation
    model.eval()
    correct = 0
    for i in range(len(X_test)):
        logits = model(X_test[i:i+1])
        pred = int(logits[0, -1, :].argmax())
        if pred == int(y_test[i]):
            correct += 1
    test_acc = correct / len(X_test)

    total_time = time.perf_counter() - t_start
    params = model.count_parameters()
    return {
        "final_test_acc": test_acc,
        "total_time_s": total_time,
        "params": params,
    }


# ======================================================================
# Experiment Jobs (for parallel execution)
# ======================================================================

def run_single_experiment(job: dict) -> dict:
    """Run a single (model_config, seed) experiment."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore")

    config_name = job["config_name"]
    config = job["config"]
    seed = job["seed"]
    n_bits = job.get("n_bits", 8)
    epochs = job.get("epochs", 20)

    np.random.seed(seed)
    X_train, y_train = generate_parity_dataset(n_bits, 800, seed)
    X_test, y_test = generate_parity_dataset(n_bits, 200, seed + 1000)

    model_type = config["type"]
    qubits = config.get("qubits", 6)

    if model_type == "classical":
        model = HybridQuantumLLM(
            vocab_size=2, d_model=16, n_layers=1, n_heads=4, d_ff=32,
            max_seq_length=16, dropout=0.0, quantum_heads_per_layer=0,
            quantum_config={"use_quantum_embedding": False, "use_quantum_activation": False},
        )
    elif model_type == "replacement":
        model = HybridQuantumLLM(
            vocab_size=2, d_model=16, n_layers=1, n_heads=4, d_ff=32,
            max_seq_length=16, dropout=0.0, quantum_heads_per_layer=1,
            quantum_config={
                "use_quantum_embedding": False, "use_quantum_activation": False,
                "attention_qubits": qubits,
            },
        )
    elif model_type == "additive":
        model = AdditiveQuantumLLM(
            vocab_size=2, d_model=16, n_heads=4, d_ff=32,
            max_seq_length=16, quantum_qubits=qubits, n_quantum_circuits=1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    result = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs=epochs)
    result["config_name"] = config_name
    result["seed"] = seed
    return result


# ======================================================================
# Main
# ======================================================================

EXPERIMENT_CONFIGS = {
    # --- Additive vs Replacement (6 qubits) ---
    "classical_4heads": {"type": "classical"},
    "replacement_6q": {"type": "replacement", "qubits": 6},
    "additive_6q": {"type": "additive", "qubits": 6},

    # --- Qubit Scaling (additive) ---
    "additive_9q": {"type": "additive", "qubits": 9},
    "additive_12q": {"type": "additive", "qubits": 12},

    # --- Qubit Scaling (replacement) ---
    "replacement_9q": {"type": "replacement", "qubits": 9},
    "replacement_12q": {"type": "replacement", "qubits": 12},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", nargs="?", const=0, type=int, metavar="N")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    n_workers = args.parallel if args.parallel and args.parallel > 0 else mp.cpu_count()
    use_parallel = args.parallel is not None
    seeds = SEEDS[:args.seeds]

    print("=" * 60)
    print("EXPERIMENT: ADDITIVE QUANTUM + QUBIT SCALING")
    print("=" * 60)
    print(f"Task: Full 8-bit XOR parity (the proven quantum-favorable task)")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Parallel: {use_parallel} ({n_workers} workers)" if use_parallel else "Sequential")
    print()

    print("Configurations:")
    for name, cfg in EXPERIMENT_CONFIGS.items():
        print(f"  {name}: {cfg}")
    print()

    # Build all jobs
    jobs = []
    for config_name, config in EXPERIMENT_CONFIGS.items():
        for seed in seeds:
            jobs.append({
                "config_name": config_name,
                "config": config,
                "seed": seed,
                "epochs": args.epochs,
            })

    print(f"Total jobs: {len(jobs)}")
    t_start = time.perf_counter()

    # Execute
    if use_parallel:
        print(f"Running with {n_workers} parallel workers...")
        with mp.Pool(n_workers) as pool:
            all_results = pool.map(run_single_experiment, jobs)
    else:
        all_results = []
        for i, job in enumerate(jobs):
            print(f"  Job {i+1}/{len(jobs)}: {job['config_name']} seed={job['seed']}...", end=" ", flush=True)
            result = run_single_experiment(job)
            print(f"acc={result['final_test_acc']*100:.1f}%")
            all_results.append(result)

    total_time = time.perf_counter() - t_start
    print(f"\nAll jobs complete in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Aggregate results
    grouped = {}
    for r in all_results:
        grouped.setdefault(r["config_name"], []).append(r)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"{'Config':<20} {'Mean Acc':>10} {'95% CI':>20} {'Params':>8} {'Q Params':>8}")
    print("-" * 70)

    summary = {}
    for config_name in EXPERIMENT_CONFIGS:
        results = grouped.get(config_name, [])
        if not results:
            continue
        accs = [r["final_test_acc"] for r in results]
        ci = confidence_interval(accs)
        params = results[0]["params"]
        print(f"{config_name:<20} {ci['mean']*100:>9.1f}% "
              f"[{ci['ci_low']*100:.1f}%, {ci['ci_high']*100:.1f}%] "
              f"{params['total']:>7} {params['quantum']:>7}")
        summary[config_name] = {
            "mean_acc": ci["mean"],
            "ci_95": [ci["ci_low"], ci["ci_high"]],
            "std": ci["std"],
            "params": params,
            "accs": accs,
        }

    # Statistical comparisons
    print("\n" + "-" * 60)
    print("STATISTICAL COMPARISONS (vs classical_4heads)")
    print("-" * 60)

    classical_accs = summary.get("classical_4heads", {}).get("accs", [])
    comparisons = {}
    for config_name in EXPERIMENT_CONFIGS:
        if config_name == "classical_4heads":
            continue
        other_accs = summary.get(config_name, {}).get("accs", [])
        if not other_accs or not classical_accs:
            continue
        sig = paired_significance(classical_accs, other_accs)
        comparisons[config_name] = sig
        star = " ★" if sig["significant"] else ""
        print(f"  {config_name:<20}: diff={sig['mean_diff']*100:+.1f}%, "
              f"p={sig['p_value']:.4f}, d={sig['cohens_d']:.2f}{star}")

    # Key comparison: additive vs replacement
    if "additive_6q" in summary and "replacement_6q" in summary:
        print("\n" + "-" * 60)
        print("KEY COMPARISON: Additive 6q vs Replacement 6q")
        add_accs = summary["additive_6q"]["accs"]
        rep_accs = summary["replacement_6q"]["accs"]
        sig = paired_significance(rep_accs, add_accs)
        print(f"  Additive - Replacement = {sig['mean_diff']*100:+.1f}%, "
              f"p={sig['p_value']:.4f}, d={sig['cohens_d']:.2f}")

    # Qubit scaling trend
    print("\n" + "-" * 60)
    print("QUBIT SCALING TREND")
    print("-" * 60)
    for prefix in ["additive", "replacement"]:
        print(f"  {prefix}:")
        for qubits in [6, 9, 12]:
            key = f"{prefix}_{qubits}q"
            if key in summary:
                acc = summary[key]["mean_acc"]
                print(f"    {qubits}q: {acc*100:.1f}%")

    # Save results
    output = {
        "experiment": "additive_quantum_and_qubit_scaling",
        "task": "full_8bit_xor_parity",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": total_time,
        "epochs": args.epochs,
        "seeds": seeds,
        "summary": {k: {kk: vv for kk, vv in v.items() if kk != "accs"}
                    for k, v in summary.items()},
        "comparisons_vs_classical": comparisons,
        "raw_accs": {k: v["accs"] for k, v in summary.items()},
    }
    out_path = RESULTS_DIR / "experiment_additive_scaling.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\nResults saved to {out_path}")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    add_better = (summary.get("additive_6q", {}).get("mean_acc", 0) >
                  summary.get("replacement_6q", {}).get("mean_acc", 0))
    scaling_works = (summary.get("additive_12q", {}).get("mean_acc", 0) >
                     summary.get("additive_6q", {}).get("mean_acc", 0))

    if add_better:
        print("  ✓ Additive > Replacement: The architecture was the problem, not quantum.")
    else:
        print("  ✗ Additive ≤ Replacement: Architecture wasn't the bottleneck.")

    if scaling_works:
        print("  ✓ More qubits = more advantage: Scaling thesis validated.")
    else:
        print("  ✗ More qubits ≠ more advantage: Scale alone doesn't help.")

    add_sig = comparisons.get("additive_12q", {}).get("significant", False)
    if add_sig:
        print("  ★ Additive 12q shows SIGNIFICANT advantage over classical!")
        print("    → Proceed to Month 2 (real hardware).")
    else:
        print("  No configuration shows significant advantage on this task.")
        print("  → Consider: deeper circuits, more training epochs, or different tasks.")


if __name__ == "__main__":
    main()
