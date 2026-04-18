"""
Statistical Analysis for Benchmark Results
============================================
Confidence intervals, paired t-tests, effect sizes.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def confidence_interval(values: list[float] | np.ndarray, confidence: float = 0.95) -> dict:
    """Compute mean and CI using t-distribution."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        m = float(values[0]) if n == 1 else 0.0
        return {"mean": m, "ci_low": m, "ci_high": m, "std": 0.0, "n": n}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * se
    return {
        "mean": mean,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
        "std": std,
        "n": n,
    }


def paired_significance(
    baseline_values: list[float] | np.ndarray,
    treatment_values: list[float] | np.ndarray,
) -> dict:
    """Paired t-test with Cohen's d effect size."""
    baseline = np.asarray(baseline_values, dtype=float)
    treatment = np.asarray(treatment_values, dtype=float)
    assert len(baseline) == len(treatment), "Must have same number of seeds"

    diff = treatment - baseline
    n = len(diff)

    if n < 2 or np.std(diff, ddof=1) == 0:
        return {
            "t_stat": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0,
            "mean_diff": float(np.mean(diff)),
            "significant": False,
        }

    t_stat, p_value = stats.ttest_rel(treatment, baseline)
    cohens_d = float(np.mean(diff) / np.std(diff, ddof=1))

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "mean_diff": float(np.mean(diff)),
        "significant": p_value < 0.05,
    }


def summarize_results(results: dict) -> str:
    """Generate a markdown comparison table from benchmark results."""
    lines = []
    lines.append("| Task | Scaling | Classical | Quantum 6Q | MoE | p-value | Winner |")
    lines.append("|------|---------|-----------|------------|-----|---------|--------|")

    for task_name, task_results in results.items():
        for scaling_key, scaling_data in task_results.items():
            models = scaling_data.get("models", {})
            classical_acc = models.get("classical", {}).get("mean_test_acc", 0)
            quantum_acc = models.get("quantum_6q", {}).get("mean_test_acc", 0)
            moe_acc = models.get("moe_6_9_12", {}).get("mean_test_acc", 0)

            sig = scaling_data.get("significance", {}).get("classical_vs_quantum", {})
            p_val = sig.get("p_value", 1.0)
            significant = sig.get("significant", False)

            q_str = f"{quantum_acc*100:.1f}%{'*' if significant else ''}"
            winner = "Quantum" if significant and quantum_acc > classical_acc else "Classical"

            lines.append(
                f"| {task_name} | {scaling_key} | {classical_acc*100:.1f}% | "
                f"{q_str} | {moe_acc*100:.1f}% | {p_val:.3f} | {winner} |"
            )

    return "\n".join(lines)
