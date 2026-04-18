# Quantum Reasoning Co-Processor — Benchmark Report

Generated: 2026-04-18T15:58:09.642714
Machine: Linux x86_64 / Python 3.12.3

## Executive Summary

- **Tasks tested:** 20
- **Promising (>3% advantage in probe):** 2
- **Statistically significant (p<0.05, full sweep):** 0

**Verdict: INSUFFICIENT** — Quantum advantage not convincingly demonstrated. Re-evaluate approach.

## Phase 1: Classical Baselines

| Task | Category | Mean Acc | 95% CI | Params |
|------|----------|----------|--------|--------|
| CNF (3 clauses) [CONTROL] | control | 81.5% | [73.8%, 89.2%] | 2352 |
| Correlated Features (pairs=2) | quantum_favorable | 48.4% | [41.6%, 55.2%] | 2544 |
| Correlated Features (pairs=3) | quantum_favorable | 48.2% | [43.4%, 53.0%] | 2544 |
| Correlated Features (pairs=4) | quantum_favorable | 50.8% | [46.3%, 55.3%] | 2544 |
| Decision (2 constraints) | quantum_favorable | 24.3% | [20.7%, 27.9%] | ? |
| Decision (3 constraints) | quantum_favorable | 24.5% | [21.9%, 27.1%] | ? |
| k-Parity (k=2) | quantum_favorable | 89.9% | [61.9%, 117.9%] | 2352 |
| k-Parity (k=3) | quantum_favorable | 62.1% | [35.8%, 88.4%] | 2352 |
| k-Parity (k=4) | quantum_favorable | 51.8% | [48.7%, 54.9%] | 2352 |
| k-Parity (k=5) | quantum_favorable | 47.8% | [45.4%, 50.2%] | 2352 |
| Pattern Match (2 templates, noise=0.3) | quantum_favorable | 52.2% | [47.0%, 57.4%] | ? |
| Pattern Match (4 templates, noise=0.3) | quantum_favorable | 25.1% | [21.6%, 28.6%] | ? |
| Pattern Match (4 templates, noise=0.5) | quantum_favorable | 25.2% | [22.1%, 28.3%] | ? |
| Sequence Equality (2 rules) [CONTROL] | control | 100.0% | [100.0%, 100.0%] | 2416 |
| Sequence XOR (1 rule) | quantum_favorable | 82.2% | [67.4%, 97.0%] | 2416 |
| Sequence XOR (2 rules) | quantum_favorable | 59.8% | [53.4%, 66.2%] | 2416 |
| Sequence XOR (3 rules) | quantum_favorable | 45.0% | [39.3%, 50.7%] | 2416 |
| XOR-SAT (2 clauses) | quantum_favorable | 60.4% | [32.7%, 88.1%] | 2352 |
| XOR-SAT (3 clauses) | quantum_favorable | 67.6% | [30.8%, 104.4%] | 2352 |
| XOR-SAT (4 clauses) | quantum_favorable | 59.7% | [31.6%, 87.8%] | 2352 |

## Phase 2: Quantum Probe (seed=42)

| Task | Classical | Quantum | Advantage | Promising? |
|------|-----------|---------|-----------|------------|
| CNF (3 clauses) [CONTROL] | 81.5% | 58.0% | -23.5% | no |
| Correlated Features (pairs=2) | 48.4% | 51.5% | +3.1% | YES |
| Correlated Features (pairs=3) | 48.2% | 45.0% | -3.2% | no |
| Correlated Features (pairs=4) | 50.8% | 47.0% | -3.8% | no |
| Decision (2 constraints) | 24.3% | 14.5% | -9.8% | no |
| Decision (3 constraints) | 24.5% | 14.5% | -10.0% | no |
| k-Parity (k=2) | 89.9% | 53.0% | -36.9% | no |
| k-Parity (k=3) | 62.1% | 49.5% | -12.6% | no |
| k-Parity (k=4) | 51.8% | 51.5% | -0.3% | no |
| k-Parity (k=5) | 47.8% | 51.5% | +3.7% | YES |
| Pattern Match (2 templates, noise=0.3) | 52.2% | 50.0% | -2.2% | no |
| Pattern Match (4 templates, noise=0.3) | 25.1% | 25.0% | -0.1% | no |
| Pattern Match (4 templates, noise=0.5) | 25.2% | 23.5% | -1.7% | no |
| Sequence Equality (2 rules) [CONTROL] | 100.0% | 100.0% | +0.0% | no |
| Sequence XOR (1 rule) | 82.2% | 76.0% | -6.2% | no |
| Sequence XOR (2 rules) | 59.8% | 58.0% | -1.8% | no |
| Sequence XOR (3 rules) | 45.0% | 41.5% | -3.5% | no |
| XOR-SAT (2 clauses) | 60.4% | 57.5% | -2.9% | no |
| XOR-SAT (3 clauses) | 67.6% | 47.5% | -20.1% | no |
| XOR-SAT (4 clauses) | 59.7% | 54.0% | -5.7% | no |

## Phase 3: Full Sweep (5 seeds)

| Task | Classical Acc | Quantum Acc | Quantum CI | p-value | Cohen's d | Significant? |
|------|--------------|-------------|------------|---------|-----------|--------------|
| Correlated Features (pairs=2) | 48.4% | 49.0% | [44.1%, 53.9%] | 0.8801 | 0.07 | no |
| k-Parity (k=5) | 47.8% | 48.2% | [45.3%, 51.1%] | 0.7630 | 0.14 | no |

## Timing

- Phase 1 (classical): 0.0 min
- Phase 2 (quantum probe): 0.0 hours
- Phase 3 (full sweep): 0.8 hours
- **Total:** 0.8 hours

## Conclusions

No tasks showed statistically significant quantum advantage at p<0.05.

**Tasks without significant advantage:**
- k-Parity (k=5): +0.4% (p=0.7630)
- Correlated Features (pairs=2): +0.6% (p=0.8801)

## Next Steps

1. Investigate why advantage did not materialize
2. Consider increasing circuit depth or qubit count
3. Try different task formulations
4. Assess whether the 6-qubit scale is fundamentally too small
