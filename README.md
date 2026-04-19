# Quantum-Classical Hybrid ML: Research & Findings

**A rigorous experimental investigation into quantum circuit advantage for machine learning tasks.**

Built a complete hybrid quantum-classical transformer architecture, ran 11 experiments across multiple machines (CPU, GPU, Colab T4), and tested for quantum advantage on 20+ task variants at 6-24 qubits. Findings: no statistically significant quantum advantage over classical methods on any real-world task at current qubit scales.

[![License: Proprietary](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.44.1-purple.svg)](https://pennylane.ai/)

---

## Key Findings

| Claim Tested | Result | Evidence |
|-------------|--------|----------|
| Quantum attention beats classical on XOR parity | Small signal at 6q (58% vs 44%), vanishes with fair baselines | Runs 5-7 |
| Advantage grows with qubit count (6→18→24q) | No. Both models at random chance (~50%) at 18q+ | Runs 10-11 |
| Quantum helps on 20 reasoning task variants | 0/20 showed significant advantage (p<0.05) | Run 7 |
| Additive quantum (keep all classical heads) helps | No measurable benefit | Run 8 |
| Deeper circuits (4 layers vs 2) help | Untested (expressivity wall at current params) | Run 11 |

**Conclusion:** At 6-24 qubits on simulators, the quantum attention circuit provides no practical advantage over properly-sized classical baselines. The one positive signal (58% vs 44% on full 8-bit XOR) disappears when: (a) tested on other tasks, (b) scaled to more qubits, or (c) compared against fair classical models.

See [FINDINGS.md](FINDINGS.md) for detailed analysis and [RUN_LOG.md](RUN_LOG.md) for raw experiment data.

---

## What Was Built

A complete hybrid quantum-classical system — all working, tested, end-to-end trainable:

| Component | Description | Status |
|-----------|-------------|--------|
| Quantum Attention Circuit | Q-K similarity via CNOT → V modulation | Working, 6-24 qubits |
| Fast Numpy Simulator | Custom statevector sim, 14x faster than PennyLane | Working |
| Adaptive Qubit MoE | Trainable router selects 6/9/12 qubit circuits | Working |
| GPU Pipeline | lightning.gpu on Colab T4, adjoint differentiation | Working |
| Hybrid Transformer | Full LLM architecture (embedding, attention, FFN, generation) | Working |
| Training Pipeline | autograd + parameter-swapping, separate classical/quantum LRs | Working |
| Benchmark Suite | 20 tasks, multi-seed, parallel, statistical analysis | Working |
| Agent Loop | Quantum-enhanced tool selection + multi-agent coordination | Working |
| Test Suite | 55 passing tests (unit, integration, gradient flow) | Passing |

**Technical achievements:**
- Zero PyTorch dependency (PennyLane + NumPy only)
- 14x simulator speedup via tensor operation restructuring
- Auto GPU/CPU backend selection based on qubit count
- 12-core parallel benchmark execution
- Resumable experiment checkpointing

---

## Architecture

```
Input Text
    │
    ▼
┌─────────────────────────────────────┐
│  Character Tokenizer                │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Hybrid Embedding Layer             │
│  ├── Classical token lookup         │
│  ├── Quantum feature map (4-8 qub) │
│  └── Quantum positional enc        │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Hybrid Transformer Block (×N)      │
│  ├── LayerNorm                      │
│  ├── Multi-Head Attention           │
│  │   ├── Classical heads            │
│  │   └── Quantum head (optional)    │
│  │       └── CNOT Q-K→V circuit     │
│  ├── LayerNorm                      │
│  └── Feed-Forward                   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Language Model Head → Logits       │
└─────────────────────────────────────┘
```

---

## Project Structure

```
quantum-llm-agent/
├── quantum/              # Quantum circuit components
│   ├── circuits/core.py  #   Attention, feature map, activation, positional
│   ├── simulator/        #   Fast numpy statevector simulator
│   ├── moe/              #   Adaptive qubit router + MoE
│   └── backends/         #   Device management (GPU auto-selection)
├── classical/            # Classical ML components (pure NumPy)
│   ├── nn.py             #   Module, Linear, Embedding, LayerNorm, GELU, AdamW
│   ├── tokenizer.py      #   Character-level tokenizer
│   └── optimizers/       #   Hybrid quantum-classical trainer
├── hybrid/               # Quantum-classical integration
│   ├── attention/        #   Quantum multi-head attention + MoE attention
│   ├── embeddings/       #   Hybrid embedding layer
│   ├── feedforward/      #   Hybrid FFN
│   └── interfaces/       #   Full model, caching, profiling
├── agents/               # Agentic workflows
│   ├── reasoning/        #   Quantum decision + pattern matching circuits
│   ├── memory/           #   Quantum associative memory
│   └── coordination/     #   Agent loop + multi-agent coordinator
├── benchmarks/           # Rigorous benchmarking infrastructure
│   ├── run_benchmarks.py #   Master script (parallel, resumable, 3-phase)
│   ├── runner.py         #   Shared training + evaluation infrastructure
│   ├── analysis/         #   Statistics (CI, t-test, Cohen's d)
│   └── tasks/            #   6 benchmark tasks (20 variants)
├── notebooks/            # Colab notebooks
│   └── quantum_gpu_scaling.ipynb  # GPU qubit scaling experiment
├── tests/                # 55 tests
├── config/               # YAML configs
└── examples/demos/       # Runnable demos + experiments
```

---

## Quick Start

```bash
git clone https://github.com/saikrishnajava/quantum-llm-agent.git
cd quantum-llm-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Run tests (55 passing)
python -m pytest tests/ -v

# Run 12-qubit proof of concept
python examples/demos/poc_12qubit.py

# Run parity experiment (quantum vs classical)
python examples/demos/experiment_parity.py

# Run full benchmark suite (parallel, all 20 tasks)
python benchmarks/run_benchmarks.py --parallel
```

---

## Requirements

- Python 3.10+
- PennyLane >= 0.39.0
- NumPy >= 1.24.0
- SciPy >= 1.11.0
- PyYAML >= 6.0

Optional: `pennylane-lightning[gpu]` for GPU-accelerated simulation (NVIDIA Volta+).

---

## Citation

```
@software{garikipati2026quantum_hybrid_ml,
  author = {Garikipati, Saikrishna},
  title = {Quantum-Classical Hybrid ML: Research and Findings},
  year = {2026},
  url = {https://github.com/saikrishnajava/quantum-llm-agent}
}
```

## License

Copyright (c) 2026 Saikrishna Garikipati. **All Rights Reserved.**

See [LICENSE](LICENSE) for full terms.

---

*This is a research project demonstrating that quantum circuits can be integrated into ML architectures with end-to-end trainability — and that doing so does not currently provide practical advantage over classical methods at accessible qubit scales. The architecture, training pipeline, and benchmarking infrastructure represent genuine engineering contributions regardless of the advantage finding.*
