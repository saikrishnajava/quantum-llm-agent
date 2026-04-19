# Findings: Quantum Advantage in ML

**11 experiments, 20+ task variants, 6-24 qubits, 3 hardware configurations.**

---

## Summary

We tested whether parameterized quantum circuits (specifically, a CNOT-based Q-K→V attention mechanism) provide measurable advantage over classical neural networks on reasoning and classification tasks. They do not — at any qubit scale accessible via simulation (6-24 qubits).

---

## Experiment Timeline

| Run | Date | What | Result |
|-----|------|------|--------|
| 1 | Apr 17 | First training (Linux CPU) | Architecture works, 2450ms/step |
| 2 | Apr 17 | Colab T4 GPU | GPU slower than CPU at 6 qubits (overhead > compute) |
| 3 | Apr 17 | Performance tuning | 14x speedup via custom simulator (1050ms/step) |
| 4 | Apr 17 | MoE training (6/9/12q) | MoE converges but slower than fixed circuit |
| 5 | Apr 17 | Parity experiment (v1) | Quantum 59% vs Classical 54% on XOR (small signal) |
| 6 | Apr 18 | Parity experiment (v2, redesigned circuit) | Quantum 58% vs Classical 44% (signal confirmed) |
| 7 | Apr 18 | **20-task benchmark sweep** | **0/20 tasks show significant advantage (p<0.05)** |
| 8 | Apr 18 | Additive quantum + qubit scaling (CPU) | Exponential wall at 12+ qubits on CPU |
| 9 | Apr 18 | GPU scaling (first attempt) | Crashed — multi-expval + adjoint incompatibility |
| 10 | Apr 18 | GPU scaling (fixed) | +4% at 18q (single seed) — appeared promising |
| 11 | Apr 19 | **18q multi-seed confirmation** | **+4% was noise. Both models at ~50% (random).** |

---

## Critical Experiments in Detail

### Run 7: Comprehensive 20-Task Benchmark

Tested quantum advantage across 6 problem categories:
- k-bit parity (k=2,3,4,5)
- Correlated feature pairs (2,3,4 pairs)
- XOR-SAT Boolean functions (2,3,4 clauses)
- Sequence XOR decision (1,2,3 rules)
- Circuit-level decision making
- Circuit-level pattern matching

**Controls included:** CNF (classical-favorable), equality rules (classical should match).

**Result:** Phase 2 (single seed) flagged 2 tasks as promising (>3% advantage). Phase 3 (5 seeds) showed both were noise: p=0.76 and p=0.88.

**Why quantum hurt on most tasks:** Replacing 1 of 4 classical attention heads with a quantum head (24 params, 0.1x learning rate) is a net capacity loss. The quantum circuit can't compensate for losing a trained classical head.

### Run 11: The Expressivity Wall

At 18 qubits (register_dim=64), both models scored ~50% (random):
- **Quantum (36 params, 2 layers):** Under-parameterized. Too few rotations to express complex functions in 64-dim space.
- **Classical (10,465 params, 3-layer MLP):** MLPs are bad at XOR parity without exponential parameters. Known theoretical limitation.

Neither model could learn the task — making the comparison uninformative.

---

## Why No Advantage Was Found

### 1. Scale Mismatch
6 qubits = 64-dimensional Hilbert space. A classical network with 2,352 parameters spans this space trivially. The quantum circuit's exponential state space doesn't help when classical capacity is already sufficient.

### 2. Capacity Loss
The architecture REPLACES a classical head with a quantum head. The quantum head has fewer effective parameters (24 vs ~600 per classical head) and trains 10x slower. Net result: weaker model.

### 3. Task-Circuit Mismatch
CNOT-based Q-K similarity works for full XOR parity (the one task that showed signal). But most reasoning tasks don't have this exact structure. The inductive bias only helps on problems that match the gate topology.

### 4. Barren Plateaus
At higher qubit counts with shallow circuits, gradients become exponentially small. The 2-layer circuit at 18 qubits likely hits this — unable to train effectively.

### 5. Simulation Limits
Classical simulation of quantum circuits costs O(2^n). At 12+ qubits, each training step takes minutes. This prevents the thousands of steps needed for proper convergence.

---

## What IS Proven

| Claim | Status | Evidence |
|-------|--------|----------|
| Quantum circuits can train end-to-end with gradient flow | Proven | 55 tests, loss decreases across all runs |
| Custom simulator is 14x faster than PennyLane QNode | Proven | Benchmarked, consistent |
| MoE router learns to select circuit sizes | Proven | Run 4, loss converges |
| Architecture runs on GPU (Colab T4) | Proven | Runs 9-11 |
| Quantum provides practical ML advantage | **Not proven** | 11 runs, 0 significant results |

---

## Technical Contributions (Independent of Advantage)

1. **Fast quantum simulator** — autograd-compatible numpy tensor operations, 14x speedup
2. **Q-K→V attention circuit** — novel design computing real similarity (not generic variational)
3. **Adaptive qubit MoE** — Gumbel-Softmax routing over variable-size quantum circuits
4. **Pure-NumPy ML framework** — complete transformer training without PyTorch
5. **Rigorous benchmark methodology** — multi-seed, parallel, statistical significance testing
6. **GPU auto-selection** — transparent CPU/GPU dispatch based on qubit count

---

## Implications

1. **For quantum ML researchers:** Test against fair classical baselines. Single-seed results are unreliable. Report negative results.
2. **For quantum startups:** Advantage on synthetic tasks (XOR, engineered kernels) does not transfer to practical problems.
3. **For investors:** No quantum ML method demonstrates production-ready advantage. Market claims of "$X billion quantum ML market" are not supported by deployments.
4. **For practitioners:** Use classical ML. Monitor quantum hardware progress. Revisit when error-corrected 100+ qubit systems exist.

---

## What Would Change This Conclusion

- Error-corrected quantum hardware (1000+ logical qubits, <0.01% error) — estimated 10-15 years
- A provably quantum-advantaged ML algorithm for natural (not synthetic) data — none exists currently
- Fundamentally different circuit architectures that avoid barren plateaus at scale — active research area

---

*Last updated: April 19, 2026*
