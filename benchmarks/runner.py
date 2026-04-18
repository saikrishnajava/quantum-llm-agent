"""
Benchmark Runner
=================
Shared infrastructure for running quantum vs classical experiments.
Handles model creation, training loops, multi-seed execution, and result collection.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classical.nn import Module, Linear, AdamW, CrossEntropyLoss, Softmax
from classical.optimizers.trainer import HybridQuantumTrainer
from hybrid.interfaces.model import HybridQuantumLLM
from benchmarks.analysis.statistics import confidence_interval, paired_significance

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [42, 99, 137, 256, 314]

MODEL_CONFIGS = {
    "classical": {
        "quantum_heads_per_layer": 0,
        "quantum": {
            "use_quantum_embedding": False,
            "use_quantum_activation": False,
            "attention_qubits": 6,
        },
    },
    "quantum_6q": {
        "quantum_heads_per_layer": 1,
        "quantum": {
            "use_quantum_embedding": False,
            "use_quantum_activation": False,
            "attention_qubits": 6,
        },
    },
    "moe_6_9_12": {
        "quantum_heads_per_layer": 1,
        "quantum": {
            "use_quantum_embedding": False,
            "use_quantum_activation": False,
            "use_moe": True,
            "moe_qubit_configs": [6, 9, 12],
            "moe_temperature": 1.0,
        },
    },
}


@dataclass
class TaskConfig:
    """Configuration for a single benchmark task."""
    task_name: str
    d_model: int = 16
    n_layers: int = 1
    n_heads: int = 4
    d_ff: int = 32
    max_seq_length: int = 16
    vocab_size: int = 2
    batch_size: int = 16
    learning_rate: float = 5e-3
    epochs: int = 20
    eval_every: int = 5
    n_seeds: int = 5
    models: list[str] = field(default_factory=lambda: ["classical", "quantum_6q"])


def build_model(task_config: TaskConfig, model_name: str) -> HybridQuantumLLM:
    """Build a HybridQuantumLLM with the given model variant config."""
    mcfg = MODEL_CONFIGS[model_name]
    return HybridQuantumLLM(
        vocab_size=task_config.vocab_size,
        d_model=task_config.d_model,
        n_layers=task_config.n_layers,
        n_heads=task_config.n_heads,
        quantum_heads_per_layer=mcfg["quantum_heads_per_layer"],
        d_ff=task_config.d_ff,
        max_seq_length=task_config.max_seq_length,
        dropout=0.0,
        quantum_config=mcfg.get("quantum"),
    )


def train_and_evaluate(
    model: HybridQuantumLLM,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_config: TaskConfig,
) -> dict:
    """Train model and return metrics per epoch."""
    trainer = HybridQuantumTrainer(model, learning_rate=task_config.learning_rate)
    n_train = X_train.shape[0]
    batch_size = task_config.batch_size
    n_batches = max(1, n_train // batch_size)

    epoch_results = []
    t_start = time.perf_counter()

    for epoch in range(1, task_config.epochs + 1):
        indices = np.random.permutation(n_train)
        epoch_loss = 0.0
        epoch_t0 = time.perf_counter()

        for b in range(n_batches):
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            x_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            result = trainer.training_step(x_batch, y_batch)
            epoch_loss += result["loss"]

        avg_loss = epoch_loss / n_batches
        epoch_time = time.perf_counter() - epoch_t0

        epoch_data = {"epoch": epoch, "loss": avg_loss, "time_s": epoch_time}

        if epoch % task_config.eval_every == 0 or epoch == task_config.epochs:
            train_acc = _compute_accuracy(model, X_train, y_train, batch_size)
            test_acc = _compute_accuracy(model, X_test, y_test, batch_size)
            epoch_data["train_acc"] = train_acc
            epoch_data["test_acc"] = test_acc
            logger.info(
                "  Epoch %d: loss=%.4f, train_acc=%.1f%%, test_acc=%.1f%%, time=%.1fs",
                epoch, avg_loss, train_acc * 100, test_acc * 100, epoch_time,
            )

        epoch_results.append(epoch_data)

    total_time = time.perf_counter() - t_start
    final = epoch_results[-1]

    return {
        "epochs": epoch_results,
        "final_loss": final["loss"],
        "final_train_acc": final.get("train_acc", 0.0),
        "final_test_acc": final.get("test_acc", 0.0),
        "total_time_s": total_time,
        "params": model.count_parameters(),
    }


def _compute_accuracy(
    model: HybridQuantumLLM,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> float:
    """Compute accuracy using last-position predictions."""
    model.eval()
    correct = 0
    total = 0
    for i in range(0, len(X), batch_size):
        x_batch = X[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        logits = model(x_batch)
        preds = logits[:, -1, :].argmax(axis=-1)
        targets = y_batch[:, -1] if y_batch.ndim > 1 else y_batch
        correct += int((preds == targets).sum())
        total += len(x_batch)
    model.train()
    return correct / total if total > 0 else 0.0


def run_task(
    task_config: TaskConfig,
    data_generator: Callable,
    scaling_params: list[dict],
    seeds: list[int] | None = None,
    skip_moe: bool = True,
) -> dict:
    """
    Run a full benchmark task across scaling params, models, and seeds.
    Returns structured results dict.
    """
    if seeds is None:
        seeds = SEEDS[: task_config.n_seeds]

    models_to_run = list(task_config.models)
    if skip_moe and "moe_6_9_12" in models_to_run:
        models_to_run.remove("moe_6_9_12")

    all_results = {}

    for sp in scaling_params:
        sp_key = "_".join(f"{k}={v}" for k, v in sp.items())
        print(f"\n{'='*60}")
        print(f"Task: {task_config.task_name} | Scaling: {sp_key}")
        print(f"{'='*60}")

        model_results = {}

        for model_name in models_to_run:
            print(f"\n  Model: {model_name}")
            seed_results = []

            for seed in seeds:
                print(f"    Seed {seed}...", end=" ", flush=True)
                np.random.seed(seed)

                X_train, y_train, X_test, y_test = data_generator(seed=seed, **sp)
                model = build_model(task_config, model_name)
                result = train_and_evaluate(
                    model, X_train, y_train, X_test, y_test, task_config
                )
                result["seed"] = seed
                seed_results.append(result)
                print(f"acc={result['final_test_acc']*100:.1f}%, "
                      f"loss={result['final_loss']:.4f}, "
                      f"time={result['total_time_s']:.1f}s")

            test_accs = [r["final_test_acc"] for r in seed_results]
            ci = confidence_interval(test_accs)

            model_results[model_name] = {
                "seeds": seed_results,
                "mean_test_acc": ci["mean"],
                "ci_95": [ci["ci_low"], ci["ci_high"]],
                "std": ci["std"],
                "params": seed_results[0]["params"],
            }

        significance = {}
        if "classical" in model_results and "quantum_6q" in model_results:
            c_accs = [r["final_test_acc"] for r in model_results["classical"]["seeds"]]
            q_accs = [r["final_test_acc"] for r in model_results["quantum_6q"]["seeds"]]
            significance["classical_vs_quantum"] = paired_significance(c_accs, q_accs)

        all_results[sp_key] = {
            "scaling_params": sp,
            "models": model_results,
            "significance": significance,
        }

        _print_comparison(sp_key, model_results, significance)

    return {
        "task": task_config.task_name,
        "config": {
            "d_model": task_config.d_model,
            "n_layers": task_config.n_layers,
            "epochs": task_config.epochs,
            "batch_size": task_config.batch_size,
            "lr": task_config.learning_rate,
        },
        "results": all_results,
    }


def _print_comparison(sp_key: str, model_results: dict, significance: dict):
    """Print a comparison summary for one scaling point."""
    print(f"\n  --- Results for {sp_key} ---")
    for name, data in model_results.items():
        ci = data["ci_95"]
        print(f"    {name:15s}: acc={data['mean_test_acc']*100:.1f}% "
              f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%] "
              f"params={data['params']['total']}")

    if "classical_vs_quantum" in significance:
        sig = significance["classical_vs_quantum"]
        star = "*" if sig["significant"] else ""
        print(f"    Significance: p={sig['p_value']:.4f}{star}, "
              f"Cohen's d={sig['cohens_d']:.2f}, "
              f"diff={sig['mean_diff']*100:+.1f}%")


def save_results(results: dict, filename: str):
    """Save results to JSON."""
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nResults saved to {path}")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
