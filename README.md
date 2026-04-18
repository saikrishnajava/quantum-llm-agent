# Quantum-Simulated LLM with Agentic Workflows

**A hybrid quantum-classical language model with adaptive qubit routing and agentic workflows.**

Research proof-of-concept exploring whether quantum circuits can meaningfully enhance transformer attention. Not a production model — a working experiment with end-to-end trainability through quantum circuits.

[![License: Proprietary](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.44.1-purple.svg)](https://pennylane.ai/)

---

## What This Is

A hybrid transformer where key components run through parameterized quantum circuits on simulated qubits:

| Component | Classical | Quantum |
|-----------|-----------|---------|
| **Token Embedding** | Lookup table | + Quantum feature map (amplitude encoding) |
| **Positional Encoding** | Sinusoidal | + Quantum angle encoding per position |
| **Attention** | Scaled dot-product | + Adaptive qubit MoE (6/9/12 qubit experts with trainable router) |
| **Activation** | GELU | + Quantum activation circuit |
| **Reasoning** | — | Quantum decision circuit + pattern matcher |
| **Memory** | — | Quantum associative recall |
| **Agent** | Tool-use loop | Quantum-enhanced tool selection + multi-agent coordination |

Built on PennyLane + NumPy. Zero PyTorch dependency. Includes a fast numpy quantum simulator (14x faster than PennyLane QNode for small circuits).

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
│  ├── Quantum Multi-Head Attention   │
│  │   ├── Classical heads (3)        │
│  │   └── Quantum head (1, 6 qub)   │
│  ├── LayerNorm                      │
│  └── Hybrid Feed-Forward            │
│      └── GELU / Quantum Activation  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Language Model Head → Logits       │
└─────────────────────────────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/saikrishnajava/quantum-llm-agent.git
cd quantum-llm-agent

# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Run the 12-qubit proof of concept
python examples/demos/poc_12qubit.py

# Run training on Shakespeare
python examples/demos/train_gpu.py

# Run tests
python -m pytest tests/ -v
```

## Training

The model trains with real gradient descent through quantum circuits:

```python
from hybrid.interfaces.model import HybridQuantumLLM
from classical.optimizers.trainer import HybridQuantumTrainer
from classical.tokenizer import CharTokenizer

tokenizer = CharTokenizer.from_text("your training text here")
model = HybridQuantumLLM.from_config("proof_of_concept")
trainer = HybridQuantumTrainer(model, learning_rate=1e-3)

# Each step computes gradients through quantum circuits
result = trainer.training_step(input_ids, labels)
print(f"loss={result['loss']:.4f}, grad_norm={result['grad_norm']:.4f}")
```

Training speed depends on backend:

| Backend | Hardware | Speed | Command |
|---------|----------|-------|---------|
| `default.qubit` + backprop | CPU | ~800ms/step | Default |
| `lightning.qubit` + adjoint | CPU (C++) | ~200ms/step | Install `pennylane-lightning` |
| `lightning.gpu` + adjoint | NVIDIA GPU | ~50ms/step | Install `pennylane-lightning[gpu]` |

## Project Structure

```
quantum-llm-agent/
├── quantum/              # Quantum circuit components
│   ├── circuits/core.py  #   Feature map, attention, activation, positional circuits
│   ├── encodings/        #   Amplitude, angle, basis, variational encoders
│   ├── gates/layers.py   #   Reusable gate patterns
│   └── backends/         #   Device management + resource tracking
├── classical/            # Classical ML components
│   ├── nn.py             #   Module, Linear, Embedding, LayerNorm, GELU, AdamW, etc.
│   ├── tokenizer.py      #   Character-level tokenizer
│   ├── data.py           #   Dataset + DataLoader
│   └── optimizers/       #   Hybrid quantum-classical trainer
├── hybrid/               # Quantum-classical integration
│   ├── attention/        #   Quantum multi-head attention
│   ├── embeddings/       #   Hybrid embedding layer
│   ├── feedforward/      #   Hybrid FFN with quantum activation
│   └── interfaces/       #   Full model, caching, profiling
├── agents/               # Agentic workflows
│   ├── reasoning/        #   Quantum decision + pattern matching
│   ├── memory/           #   Quantum associative memory
│   └── coordination/     #   Agent loop + multi-agent coordinator
├── tests/                # 50 tests (unit, integration, gradient flow, benchmarks)
├── config/               # YAML configs for backends, models, training
└── examples/demos/       # Runnable demos
```

## Key Results

| Metric | Value |
|--------|-------|
| Quantum circuit types | 6 (feature map, positional, attention, activation, decision, pattern) |
| Adaptive qubit MoE | Trainable router over 6/9/12 qubit experts |
| Training | End-to-end gradient flow through quantum circuits |
| Fast simulator | 14x faster than PennyLane QNode (numpy tensor ops) |
| Tests | 55 passing |
| Dependencies | PennyLane + NumPy only |

**Current scale:** ~11K params, character-level, trained on Shakespeare. Output is not coherent — this is a proof-of-concept for the architecture, not a production language model.

## Requirements

- Python 3.10+
- PennyLane >= 0.39.0
- NumPy >= 1.24.0
- SciPy >= 1.11.0
- PyYAML >= 6.0

Optional for GPU acceleration:
- NVIDIA GPU with CUDA (compute capability 7.0+ / Volta or newer)
- `pennylane-lightning[gpu]`
- Note: Pascal GPUs (P2200, GTX 1080, etc.) are not supported by lightning.gpu — they fall back to lightning.qubit (C++ CPU) automatically

## GPU Setup (NVIDIA)

```bash
bash setup_gpu.sh
python examples/demos/train_gpu.py
```

The training script auto-detects the fastest available backend.

## Citation

If you use this work in your research, please cite:

```
@software{garikipati2026quantum_llm,
  author = {Garikipati, Saikrishna},
  title = {Quantum-Simulated LLM with Agentic Workflows},
  year = {2026},
  url = {https://github.com/saikrishnajava/quantum-llm-agent}
}
```

## License

Copyright (c) 2026 Saikrishna Garikipati. **All Rights Reserved.**

This software is proprietary. No permission is granted to copy, modify, distribute, or use this software without the prior written consent of the author. See [LICENSE](LICENSE) for full terms.

For licensing inquiries, research collaboration, or commercial use, contact Saikrishna Garikipati.

## Author

**Saikrishna Garikipati**

---

*This is a research proof-of-concept. It demonstrates that quantum circuits can be integrated into a transformer architecture with end-to-end trainability and adaptive qubit allocation. It does not produce coherent text at current scale — that requires more data, larger models, and more compute. The architecture is designed to be hardware-ready for future quantum computers.*
