#!/usr/bin/env python3
"""
Quantum Reasoning Co-Processor — Master Benchmark Script
=========================================================
Single script that runs ALL benchmarks in optimal order and produces
a comprehensive results file.

Phases:
  1. Classical baselines (all tasks, all seeds) — fast, ~1-2 hours
  2. Quantum quick probe (all tasks, seed=42 only) — ~6-8 hours
  3. Full quantum sweep (promising tasks, all 5 seeds) — ~2-3 days
  4. Generate final report

Usage:
    python benchmarks/run_benchmarks.py                # full run (all phases)
    python benchmarks/run_benchmarks.py --phase 1      # classical baselines only
    python benchmarks/run_benchmarks.py --phase 2      # quantum probe only
    python benchmarks/run_benchmarks.py --phase 3      # full sweep on promising tasks
    python benchmarks/run_benchmarks.py --report       # generate report from existing results

Output:
    benchmarks/results/benchmark_results.json   — full structured results
    benchmarks/results/benchmark_report.md      — human/LLM-readable summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from benchmarks.runner import (
    TaskConfig, build_model, train_and_evaluate,
    MODEL_CONFIGS, SEEDS, RESULTS_DIR, _json_default,
)
from benchmarks.analysis.statistics import confidence_interval, paired_significance

from benchmarks.tasks.benchmark_k_parity import generate_k_parity
from benchmarks.tasks.benchmark_correlated_features import generate_correlated_features
from benchmarks.tasks.benchmark_boolean import generate_xor_sat, generate_cnf
from benchmarks.tasks.benchmark_sequence_decision import (
    generate_xor_sequence_decision, generate_equality_sequence_decision,
)
from benchmarks.tasks.benchmark_decision import (
    generate_decision_task, ClassicalDecisionMLP, train_circuit_model,
)
from benchmarks.tasks.benchmark_pattern import (
    generate_quantum_separable_patterns, ClassicalPatternMatcher, train_pattern_model,
)
from agents.reasoning.quantum_reasoning import QuantumDecisionCircuit, QuantumPatternMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ======================================================================
# Task Definitions
# ======================================================================

ADVANTAGE_THRESHOLD = 0.03  # 3% advantage to qualify as "promising"

MODEL_LEVEL_TASKS = [
    {
        "name": "k_parity_k2",
        "display": "k-Parity (k=2)",
        "generator": generate_k_parity,
        "params": {"n_bits": 8, "k": 2},
        "category": "quantum_favorable",
        "vocab_size": 2,
    },
    {
        "name": "k_parity_k3",
        "display": "k-Parity (k=3)",
        "generator": generate_k_parity,
        "params": {"n_bits": 8, "k": 3},
        "category": "quantum_favorable",
        "vocab_size": 2,
    },
    {
        "name": "k_parity_k4",
        "display": "k-Parity (k=4)",
        "generator": generate_k_parity,
        "params": {"n_bits": 8, "k": 4},
        "category": "quantum_favorable",
        "vocab_size": 2,
    },
    {
        "name": "k_parity_k5",
        "display": "k-Parity (k=5)",
        "generator": generate_k_parity,
        "params": {"n_bits": 8, "k": 5},
        "category": "quantum_favorable",
        "vocab_size": 2,
    },
    {
        "name": "correlated_pairs2",
        "display": "Correlated Features (pairs=2)",
        "generator": generate_correlated_features,
        "params": {"n_features": 8, "n_pairs": 2},
        "category": "quantum_favorable",
        "vocab_size": 8,
    },
    {
        "name": "correlated_pairs3",
        "display": "Correlated Features (pairs=3)",
        "generator": generate_correlated_features,
        "params": {"n_features": 8, "n_pairs": 3},
        "category": "quantum_favorable",
        "vocab_size": 8,
    },
    {
        "name": "correlated_pairs4",
        "display": "Correlated Features (pairs=4)",
        "generator": generate_correlated_features,
        "params": {"n_features": 8, "n_pairs": 4},
        "category": "quantum_favorable",
        "vocab_size": 8,
    },
    {
        "name": "xor_sat_2clause",
        "display": "XOR-SAT (2 clauses)",
        "generator": generate_xor_sat,
        "params": {"n_bits": 8, "n_clauses": 2, "clause_size": 3},
        "category": "quantum_favorable",
        "vocab_size": 2,
    },
    {
        "name": "xor_sat_3clause",
        "display": "XOR-SAT (3 clauses)",
        "generator": generate_xor_sat,
        "params": {"n_bits": 8, "n_clauses": 3, "clause_size": 3},
        "category": "quantum_favorable",
        "vocab_size": 2,
    },
    {
        "name": "xor_sat_4clause",
        "display": "XOR-SAT (4 clauses)",
        "generator": generate_xor_sat,
        "params": {"n_bits": 8, "n_clauses": 4, "clause_size": 3},
        "category": "quantum_favorable",
        "vocab_size": 2,
    },
    {
        "name": "cnf_3clause",
        "display": "CNF (3 clauses) [CONTROL]",
        "generator": generate_cnf,
        "params": {"n_bits": 8, "n_clauses": 3, "clause_size": 3},
        "category": "control",
        "vocab_size": 2,
    },
    {
        "name": "seq_xor_1rule",
        "display": "Sequence XOR (1 rule)",
        "generator": generate_xor_sequence_decision,
        "params": {"seq_len": 8, "token_vocab": 4, "n_actions": 4, "n_rules": 1},
        "category": "quantum_favorable",
        "vocab_size": 4,
    },
    {
        "name": "seq_xor_2rules",
        "display": "Sequence XOR (2 rules)",
        "generator": generate_xor_sequence_decision,
        "params": {"seq_len": 8, "token_vocab": 4, "n_actions": 4, "n_rules": 2},
        "category": "quantum_favorable",
        "vocab_size": 4,
    },
    {
        "name": "seq_xor_3rules",
        "display": "Sequence XOR (3 rules)",
        "generator": generate_xor_sequence_decision,
        "params": {"seq_len": 8, "token_vocab": 4, "n_actions": 4, "n_rules": 3},
        "category": "quantum_favorable",
        "vocab_size": 4,
    },
    {
        "name": "seq_eq_2rules",
        "display": "Sequence Equality (2 rules) [CONTROL]",
        "generator": generate_equality_sequence_decision,
        "params": {"seq_len": 8, "token_vocab": 4, "n_actions": 4, "n_rules": 2},
        "category": "control",
        "vocab_size": 4,
    },
]

CIRCUIT_LEVEL_TASKS = [
    {
        "name": "decision_c2",
        "display": "Decision (2 constraints)",
        "params": {"n_options": 4, "context_dim": 8, "n_constraints": 2},
        "category": "quantum_favorable",
    },
    {
        "name": "decision_c3",
        "display": "Decision (3 constraints)",
        "params": {"n_options": 4, "context_dim": 8, "n_constraints": 3},
        "category": "quantum_favorable",
    },
    {
        "name": "pattern_t2_n03",
        "display": "Pattern Match (2 templates, noise=0.3)",
        "params": {"n_templates": 2, "pattern_dim": 8, "noise": 0.3},
        "category": "quantum_favorable",
    },
    {
        "name": "pattern_t4_n03",
        "display": "Pattern Match (4 templates, noise=0.3)",
        "params": {"n_templates": 4, "pattern_dim": 8, "noise": 0.3},
        "category": "quantum_favorable",
    },
    {
        "name": "pattern_t4_n05",
        "display": "Pattern Match (4 templates, noise=0.5)",
        "params": {"n_templates": 4, "pattern_dim": 8, "noise": 0.5},
        "category": "quantum_favorable",
    },
]


# ======================================================================
# Phase 1: Classical Baselines
# ======================================================================

def run_phase1_classical(results: dict) -> dict:
    """Run all tasks with classical model only. Fast (~1-2 hours total)."""
    logger.info("=" * 60)
    logger.info("PHASE 1: CLASSICAL BASELINES")
    logger.info("=" * 60)

    phase1 = results.setdefault("phase1_classical", {})
    t_phase = time.perf_counter()

    for task in MODEL_LEVEL_TASKS:
        name = task["name"]
        if name in phase1 and phase1[name].get("completed"):
            logger.info(f"  Skipping {name} (already completed)")
            continue

        logger.info(f"\n  Task: {task['display']}")
        task_results = []

        for seed in SEEDS:
            np.random.seed(seed)
            X_tr, y_tr, X_te, y_te = task["generator"](seed=seed, **task["params"])

            config = TaskConfig(
                task_name=name, vocab_size=task["vocab_size"],
                d_model=16, n_layers=1, n_heads=4, d_ff=32,
                max_seq_length=16, batch_size=16, learning_rate=5e-3,
                epochs=20, eval_every=5,
            )
            model = build_model(config, "classical")
            result = train_and_evaluate(model, X_tr, y_tr, X_te, y_te, config)
            result["seed"] = seed
            task_results.append(result)
            logger.info(f"    seed={seed}: acc={result['final_test_acc']*100:.1f}%, "
                       f"loss={result['final_loss']:.4f}, time={result['total_time_s']:.1f}s")

        accs = [r["final_test_acc"] for r in task_results]
        ci = confidence_interval(accs)
        phase1[name] = {
            "display": task["display"],
            "category": task["category"],
            "seeds": task_results,
            "mean_acc": ci["mean"],
            "ci_95": [ci["ci_low"], ci["ci_high"]],
            "std": ci["std"],
            "params": task_results[0]["params"],
            "completed": True,
        }
        _save_checkpoint(results)

    # Circuit-level classical baselines
    for task in CIRCUIT_LEVEL_TASKS:
        name = task["name"]
        if name in phase1 and phase1[name].get("completed"):
            logger.info(f"  Skipping {name} (already completed)")
            continue

        logger.info(f"\n  Task: {task['display']} (circuit-level)")
        task_results = []
        sp = task["params"]

        for seed in SEEDS:
            np.random.seed(seed)

            if "decision" in name:
                X_tr, y_tr, X_te, y_te = generate_decision_task(seed=seed, **sp)
                context_dim = sp["context_dim"]
                n_options = sp["n_options"]
                mlp = ClassicalDecisionMLP(context_dim, n_options, hidden_dim=16)
                result = train_circuit_model(
                    mlp, X_tr, y_tr, X_te, y_te,
                    context_dim, n_options, epochs=20, lr=0.01, is_quantum=False,
                )
            else:  # pattern
                X_tr, y_tr, X_te, y_te = generate_quantum_separable_patterns(seed=seed, **sp)
                pattern_dim = sp["pattern_dim"]
                n_templates = sp["n_templates"]
                rng = np.random.RandomState(seed)
                base = rng.randn(pattern_dim)
                base = base / np.linalg.norm(base)
                templates = np.array([
                    base * np.array([(-1.0)**((t >> i) & 1) for i in range(pattern_dim)])
                    for t in range(n_templates)
                ])
                cpm = ClassicalPatternMatcher(pattern_dim, n_templates, hidden_dim=12)
                result = train_pattern_model(
                    cpm, templates, X_tr, y_tr, X_te, y_te,
                    epochs=20, lr=0.01, is_quantum=False,
                )

            result["seed"] = seed
            task_results.append(result)
            logger.info(f"    seed={seed}: acc={result['final_test_acc']*100:.1f}%")

        accs = [r["final_test_acc"] for r in task_results]
        ci = confidence_interval(accs)
        phase1[name] = {
            "display": task["display"],
            "category": task["category"],
            "seeds": task_results,
            "mean_acc": ci["mean"],
            "ci_95": [ci["ci_low"], ci["ci_high"]],
            "std": ci["std"],
            "completed": True,
        }
        _save_checkpoint(results)

    elapsed = time.perf_counter() - t_phase
    logger.info(f"\n  Phase 1 complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    results["phase1_time_s"] = elapsed
    _save_checkpoint(results)
    return results


# ======================================================================
# Phase 2: Quantum Quick Probe (seed=42 only)
# ======================================================================

def run_phase2_quantum_probe(results: dict) -> dict:
    """Run all tasks with quantum model, seed=42 only. Get quick signal."""
    logger.info("=" * 60)
    logger.info("PHASE 2: QUANTUM PROBE (seed=42)")
    logger.info("=" * 60)

    phase2 = results.setdefault("phase2_quantum_probe", {})
    t_phase = time.perf_counter()

    for task in MODEL_LEVEL_TASKS:
        name = task["name"]
        if name in phase2 and phase2[name].get("completed"):
            logger.info(f"  Skipping {name} (already completed)")
            continue

        logger.info(f"\n  Task: {task['display']}")
        seed = 42
        np.random.seed(seed)
        X_tr, y_tr, X_te, y_te = task["generator"](seed=seed, **task["params"])

        config = TaskConfig(
            task_name=name, vocab_size=task["vocab_size"],
            d_model=16, n_layers=1, n_heads=4, d_ff=32,
            max_seq_length=16, batch_size=16, learning_rate=5e-3,
            epochs=20, eval_every=5,
        )
        model = build_model(config, "quantum_6q")
        result = train_and_evaluate(model, X_tr, y_tr, X_te, y_te, config)
        result["seed"] = seed

        classical_acc = results.get("phase1_classical", {}).get(name, {}).get("mean_acc", 0.5)
        advantage = result["final_test_acc"] - classical_acc

        phase2[name] = {
            "display": task["display"],
            "category": task["category"],
            "result": result,
            "quantum_acc": result["final_test_acc"],
            "classical_acc": classical_acc,
            "advantage": advantage,
            "promising": advantage > ADVANTAGE_THRESHOLD,
            "completed": True,
        }
        logger.info(f"    Quantum acc: {result['final_test_acc']*100:.1f}% "
                   f"(classical: {classical_acc*100:.1f}%, advantage: {advantage*100:+.1f}%)"
                   f"{' ★ PROMISING' if advantage > ADVANTAGE_THRESHOLD else ''}")
        _save_checkpoint(results)

    # Circuit-level quantum
    for task in CIRCUIT_LEVEL_TASKS:
        name = task["name"]
        if name in phase2 and phase2[name].get("completed"):
            logger.info(f"  Skipping {name} (already completed)")
            continue

        logger.info(f"\n  Task: {task['display']} (circuit-level)")
        seed = 42
        np.random.seed(seed)
        sp = task["params"]

        if "decision" in name:
            X_tr, y_tr, X_te, y_te = generate_decision_task(seed=seed, **sp)
            context_dim = sp["context_dim"]
            n_options = sp["n_options"]
            qdc = QuantumDecisionCircuit(n_qubits=6, n_layers=2)
            result = train_circuit_model(
                qdc, X_tr, y_tr, X_te, y_te,
                context_dim, n_options, epochs=20, lr=0.01, is_quantum=True,
            )
        else:  # pattern
            X_tr, y_tr, X_te, y_te = generate_quantum_separable_patterns(seed=seed, **sp)
            pattern_dim = sp["pattern_dim"]
            n_templates = sp["n_templates"]
            rng = np.random.RandomState(seed)
            base = rng.randn(pattern_dim)
            base = base / np.linalg.norm(base)
            templates = np.array([
                base * np.array([(-1.0)**((t >> i) & 1) for i in range(pattern_dim)])
                for t in range(n_templates)
            ])
            qpm = QuantumPatternMatcher(n_qubits=6, n_layers=1)
            result = train_pattern_model(
                qpm, templates, X_tr, y_tr, X_te, y_te,
                epochs=20, lr=0.01, is_quantum=True,
            )

        result["seed"] = seed
        classical_acc = results.get("phase1_classical", {}).get(name, {}).get("mean_acc", 0.25)
        advantage = result["final_test_acc"] - classical_acc

        phase2[name] = {
            "display": task["display"],
            "category": task["category"],
            "result": result,
            "quantum_acc": result["final_test_acc"],
            "classical_acc": classical_acc,
            "advantage": advantage,
            "promising": advantage > ADVANTAGE_THRESHOLD,
            "completed": True,
        }
        logger.info(f"    Quantum acc: {result['final_test_acc']*100:.1f}% "
                   f"(advantage: {advantage*100:+.1f}%)"
                   f"{' ★ PROMISING' if advantage > ADVANTAGE_THRESHOLD else ''}")
        _save_checkpoint(results)

    elapsed = time.perf_counter() - t_phase
    logger.info(f"\n  Phase 2 complete in {elapsed:.0f}s ({elapsed/3600:.1f} hours)")
    results["phase2_time_s"] = elapsed

    # Identify promising tasks
    promising = [name for name, data in phase2.items() if data.get("promising")]
    results["promising_tasks"] = promising
    logger.info(f"\n  Promising tasks ({len(promising)}): {promising}")
    _save_checkpoint(results)
    return results


# ======================================================================
# Phase 3: Full Quantum Sweep (promising tasks only, all 5 seeds)
# ======================================================================

def run_phase3_full_sweep(results: dict) -> dict:
    """Run full 5-seed sweep on tasks identified as promising in Phase 2."""
    logger.info("=" * 60)
    logger.info("PHASE 3: FULL QUANTUM SWEEP (5 seeds)")
    logger.info("=" * 60)

    promising = results.get("promising_tasks", [])
    if not promising:
        logger.info("  No promising tasks found. Skipping Phase 3.")
        logger.info("  (All tasks showed <3% quantum advantage in Phase 2)")
        return results

    logger.info(f"  Running full sweep on: {promising}")
    phase3 = results.setdefault("phase3_full_sweep", {})
    t_phase = time.perf_counter()

    # Model-level tasks
    for task in MODEL_LEVEL_TASKS:
        name = task["name"]
        if name not in promising:
            continue
        if name in phase3 and phase3[name].get("completed"):
            logger.info(f"  Skipping {name} (already completed)")
            continue

        logger.info(f"\n  Task: {task['display']} (full 5-seed sweep)")
        task_results = []

        for seed in SEEDS:
            np.random.seed(seed)
            X_tr, y_tr, X_te, y_te = task["generator"](seed=seed, **task["params"])

            config = TaskConfig(
                task_name=name, vocab_size=task["vocab_size"],
                d_model=16, n_layers=1, n_heads=4, d_ff=32,
                max_seq_length=16, batch_size=16, learning_rate=5e-3,
                epochs=20, eval_every=5,
            )
            model = build_model(config, "quantum_6q")
            result = train_and_evaluate(model, X_tr, y_tr, X_te, y_te, config)
            result["seed"] = seed
            task_results.append(result)
            logger.info(f"    seed={seed}: acc={result['final_test_acc']*100:.1f}%, "
                       f"time={result['total_time_s']:.1f}s")

        q_accs = [r["final_test_acc"] for r in task_results]
        c_accs = [r["final_test_acc"] for r in
                  results["phase1_classical"][name]["seeds"]]
        ci_q = confidence_interval(q_accs)
        sig = paired_significance(c_accs, q_accs)

        phase3[name] = {
            "display": task["display"],
            "category": task["category"],
            "seeds": task_results,
            "mean_acc": ci_q["mean"],
            "ci_95": [ci_q["ci_low"], ci_q["ci_high"]],
            "std": ci_q["std"],
            "significance": sig,
            "completed": True,
        }
        logger.info(f"    Mean: {ci_q['mean']*100:.1f}% [{ci_q['ci_low']*100:.1f}%, {ci_q['ci_high']*100:.1f}%]")
        logger.info(f"    vs Classical: p={sig['p_value']:.4f}, d={sig['cohens_d']:.2f}")
        _save_checkpoint(results)

    # Circuit-level tasks
    for task in CIRCUIT_LEVEL_TASKS:
        name = task["name"]
        if name not in promising:
            continue
        if name in phase3 and phase3[name].get("completed"):
            continue

        logger.info(f"\n  Task: {task['display']} (circuit-level, full sweep)")
        task_results = []
        sp = task["params"]

        for seed in SEEDS:
            np.random.seed(seed)

            if "decision" in name:
                X_tr, y_tr, X_te, y_te = generate_decision_task(seed=seed, **sp)
                context_dim = sp["context_dim"]
                n_options = sp["n_options"]
                qdc = QuantumDecisionCircuit(n_qubits=6, n_layers=2)
                result = train_circuit_model(
                    qdc, X_tr, y_tr, X_te, y_te,
                    context_dim, n_options, epochs=20, lr=0.01, is_quantum=True,
                )
            else:
                X_tr, y_tr, X_te, y_te = generate_quantum_separable_patterns(seed=seed, **sp)
                pattern_dim = sp["pattern_dim"]
                n_templates = sp["n_templates"]
                rng = np.random.RandomState(seed)
                base = rng.randn(pattern_dim)
                base = base / np.linalg.norm(base)
                templates = np.array([
                    base * np.array([(-1.0)**((t >> i) & 1) for i in range(pattern_dim)])
                    for t in range(n_templates)
                ])
                qpm = QuantumPatternMatcher(n_qubits=6, n_layers=1)
                result = train_pattern_model(
                    qpm, templates, X_tr, y_tr, X_te, y_te,
                    epochs=20, lr=0.01, is_quantum=True,
                )

            result["seed"] = seed
            task_results.append(result)
            logger.info(f"    seed={seed}: acc={result['final_test_acc']*100:.1f}%")

        q_accs = [r["final_test_acc"] for r in task_results]
        c_accs = [r["final_test_acc"] for r in
                  results["phase1_classical"][name]["seeds"]]
        ci_q = confidence_interval(q_accs)
        sig = paired_significance(c_accs, q_accs)

        phase3[name] = {
            "display": task["display"],
            "category": task["category"],
            "seeds": task_results,
            "mean_acc": ci_q["mean"],
            "ci_95": [ci_q["ci_low"], ci_q["ci_high"]],
            "std": ci_q["std"],
            "significance": sig,
            "completed": True,
        }
        _save_checkpoint(results)

    elapsed = time.perf_counter() - t_phase
    logger.info(f"\n  Phase 3 complete in {elapsed:.0f}s ({elapsed/3600:.1f} hours)")
    results["phase3_time_s"] = elapsed
    _save_checkpoint(results)
    return results


# ======================================================================
# Report Generation
# ======================================================================

def generate_report(results: dict) -> str:
    """Generate a comprehensive markdown report from results."""
    lines = []
    lines.append("# Quantum Reasoning Co-Processor — Benchmark Report")
    lines.append(f"\nGenerated: {results.get('timestamp', 'unknown')}")
    lines.append(f"Machine: {results.get('machine_info', 'unknown')}")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")

    phase2 = results.get("phase2_quantum_probe", {})
    phase3 = results.get("phase3_full_sweep", {})
    promising = results.get("promising_tasks", [])

    total_tasks = len(phase2)
    n_promising = len(promising)
    n_significant = sum(1 for v in phase3.values() if v.get("significance", {}).get("significant"))

    lines.append(f"- **Tasks tested:** {total_tasks}")
    lines.append(f"- **Promising (>3% advantage in probe):** {n_promising}")
    lines.append(f"- **Statistically significant (p<0.05, full sweep):** {n_significant}")
    lines.append("")

    if n_significant >= 4:
        lines.append("**Verdict: EXCEPTIONAL** — Strong quantum advantage demonstrated across multiple task classes.")
    elif n_significant >= 3:
        lines.append("**Verdict: GOOD** — Clear quantum advantage. Proceed to Month 2 (real hardware).")
    elif n_significant >= 2:
        lines.append("**Verdict: MINIMUM** — Some quantum advantage. Worth continuing.")
    else:
        lines.append("**Verdict: INSUFFICIENT** — Quantum advantage not convincingly demonstrated. Re-evaluate approach.")
    lines.append("")

    # Phase 1 results table
    lines.append("## Phase 1: Classical Baselines")
    lines.append("")
    lines.append("| Task | Category | Mean Acc | 95% CI | Params |")
    lines.append("|------|----------|----------|--------|--------|")
    phase1 = results.get("phase1_classical", {})
    for name, data in sorted(phase1.items()):
        if not data.get("completed"):
            continue
        ci = data.get("ci_95", [0, 0])
        params = data.get("params", {}).get("total", "?")
        lines.append(
            f"| {data.get('display', name)} | {data.get('category', '?')} | "
            f"{data.get('mean_acc', 0)*100:.1f}% | "
            f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%] | {params} |"
        )
    lines.append("")

    # Phase 2 results table
    if phase2:
        lines.append("## Phase 2: Quantum Probe (seed=42)")
        lines.append("")
        lines.append("| Task | Classical | Quantum | Advantage | Promising? |")
        lines.append("|------|-----------|---------|-----------|------------|")
        for name, data in sorted(phase2.items()):
            if not data.get("completed"):
                continue
            flag = "YES" if data.get("promising") else "no"
            lines.append(
                f"| {data.get('display', name)} | "
                f"{data.get('classical_acc', 0)*100:.1f}% | "
                f"{data.get('quantum_acc', 0)*100:.1f}% | "
                f"{data.get('advantage', 0)*100:+.1f}% | {flag} |"
            )
        lines.append("")

    # Phase 3 results table
    if phase3:
        lines.append("## Phase 3: Full Sweep (5 seeds)")
        lines.append("")
        lines.append("| Task | Classical Acc | Quantum Acc | Quantum CI | p-value | Cohen's d | Significant? |")
        lines.append("|------|--------------|-------------|------------|---------|-----------|--------------|")
        for name, data in sorted(phase3.items()):
            if not data.get("completed"):
                continue
            c_acc = phase1.get(name, {}).get("mean_acc", 0)
            sig = data.get("significance", {})
            ci = data.get("ci_95", [0, 0])
            lines.append(
                f"| {data.get('display', name)} | "
                f"{c_acc*100:.1f}% | "
                f"{data.get('mean_acc', 0)*100:.1f}% | "
                f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%] | "
                f"{sig.get('p_value', 1):.4f} | "
                f"{sig.get('cohens_d', 0):.2f} | "
                f"{'YES' if sig.get('significant') else 'no'} |"
            )
        lines.append("")

    # Timing
    lines.append("## Timing")
    lines.append("")
    lines.append(f"- Phase 1 (classical): {results.get('phase1_time_s', 0)/60:.1f} min")
    lines.append(f"- Phase 2 (quantum probe): {results.get('phase2_time_s', 0)/3600:.1f} hours")
    lines.append(f"- Phase 3 (full sweep): {results.get('phase3_time_s', 0)/3600:.1f} hours")
    lines.append(f"- **Total:** {results.get('total_time_s', 0)/3600:.1f} hours")
    lines.append("")

    # Conclusions
    lines.append("## Conclusions")
    lines.append("")
    if phase3:
        winners = [(name, data) for name, data in phase3.items()
                   if data.get("significance", {}).get("significant")]
        if winners:
            lines.append("**Tasks with statistically significant quantum advantage:**")
            for name, data in winners:
                sig = data["significance"]
                lines.append(f"- {data['display']}: +{sig['mean_diff']*100:.1f}% "
                           f"(p={sig['p_value']:.4f}, d={sig['cohens_d']:.2f})")
        else:
            lines.append("No tasks showed statistically significant quantum advantage at p<0.05.")

        losers = [(name, data) for name, data in phase3.items()
                  if not data.get("significance", {}).get("significant")]
        if losers:
            lines.append("")
            lines.append("**Tasks without significant advantage:**")
            for name, data in losers:
                sig = data.get("significance", {})
                lines.append(f"- {data['display']}: {sig.get('mean_diff', 0)*100:+.1f}% "
                           f"(p={sig.get('p_value', 1):.4f})")
    lines.append("")

    # Next steps
    lines.append("## Next Steps")
    lines.append("")
    if n_significant >= 2:
        lines.append("1. Deploy winning circuits on IBM Quantum hardware (Month 2)")
        lines.append("2. Measure noise impact on advantage")
        lines.append("3. Scale circuit size to 9/12 qubits on winning tasks")
        lines.append("4. Build FastAPI wrapper for reasoning module (Month 3)")
    else:
        lines.append("1. Investigate why advantage did not materialize")
        lines.append("2. Consider increasing circuit depth or qubit count")
        lines.append("3. Try different task formulations")
        lines.append("4. Assess whether the 6-qubit scale is fundamentally too small")
    lines.append("")

    return "\n".join(lines)


# ======================================================================
# Utilities
# ======================================================================

def _save_checkpoint(results: dict):
    """Save intermediate results to disk."""
    path = RESULTS_DIR / "benchmark_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)


def _load_checkpoint() -> dict:
    """Load previous results if they exist (for resumability)."""
    path = RESULTS_DIR / "benchmark_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _get_machine_info() -> str:
    """Get basic machine info."""
    import platform
    return f"{platform.system()} {platform.machine()} / Python {platform.python_version()}"


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Quantum Benchmark Master Script")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run specific phase only")
    parser.add_argument("--report", action="store_true", help="Generate report from existing results")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start fresh")
    args = parser.parse_args()

    if args.report:
        results = _load_checkpoint()
        if not results:
            print("No results found. Run benchmarks first.")
            return
        report = generate_report(results)
        report_path = RESULTS_DIR / "benchmark_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(report)
        print(f"\nReport saved to {report_path}")
        return

    # Load or create results
    if args.fresh:
        results = {}
    else:
        results = _load_checkpoint()

    results["timestamp"] = datetime.now().isoformat()
    results["machine_info"] = _get_machine_info()
    t_total = time.perf_counter()

    try:
        if args.phase is None or args.phase == 1:
            results = run_phase1_classical(results)

        if args.phase is None or args.phase == 2:
            if "phase1_classical" not in results:
                logger.error("Phase 1 must complete before Phase 2. Run --phase 1 first.")
                return
            results = run_phase2_quantum_probe(results)

        if args.phase is None or args.phase == 3:
            if "phase2_quantum_probe" not in results:
                logger.error("Phase 2 must complete before Phase 3. Run --phase 2 first.")
                return
            results = run_phase3_full_sweep(results)

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted! Saving checkpoint...")
    except Exception as e:
        logger.error(f"\n\nError: {e}")
        traceback.print_exc()
        logger.info("Saving checkpoint...")

    results["total_time_s"] = time.perf_counter() - t_total
    _save_checkpoint(results)

    # Generate report
    report = generate_report(results)
    report_path = RESULTS_DIR / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print("\n\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\nResults: {RESULTS_DIR / 'benchmark_results.json'}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
