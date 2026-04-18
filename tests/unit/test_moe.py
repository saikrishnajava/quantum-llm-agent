"""
Tests for Adaptive Qubit MoE (Mixture-of-Quantum-Experts).
"""

import sys
from pathlib import Path

import numpy as np
import autograd
import autograd.numpy as anp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def test_router_output_shape():
    from quantum.moe import QubitRouter
    router = QubitRouter(d_model=16, n_experts=3)
    x = np.random.randn(2, 4, 16)
    logits = router(x)
    assert logits.shape == (2, 3)


def test_adaptive_head_forward():
    from quantum.moe import AdaptiveQuantumHead
    head = AdaptiveQuantumHead(d_model=16, head_dim=16, qubit_configs=[6, 9])
    q = np.random.randn(1, 3, 16)
    k = np.random.randn(1, 3, 16)
    v = np.random.randn(1, 3, 16)
    out, load_loss = head(q, k, v)
    assert out.shape == (1, 3, 16)
    assert float(load_loss) > 0


def test_moe_attention_shape():
    from hybrid.attention.moe_attention import MoEQuantumMultiHeadAttention
    attn = MoEQuantumMultiHeadAttention(
        d_model=16, n_heads=4, moe_heads=1, qubit_configs=[6, 9], dropout=0.0)
    out = attn(np.random.randn(1, 4, 16))
    assert out.shape == (1, 4, 16)
    assert len(attn.load_balance_losses) == 1


def test_moe_gradient_flow():
    from quantum.moe import AdaptiveQuantumHead
    head = AdaptiveQuantumHead(d_model=16, head_dim=16, qubit_configs=[6, 9])
    q = np.random.randn(1, 2, 16)
    k = np.random.randn(1, 2, 16)
    v = np.random.randn(1, 2, 16)

    all_params = head.parameters()
    flat = np.concatenate([np.array(p).flatten() for p in all_params])
    locations = head._parameter_locations()

    def loss(fp):
        saved = []
        idx = 0
        for parent, attr, _, p in locations:
            s = p.size
            saved.append((parent, attr, getattr(parent, attr)))
            setattr(parent, attr, fp[idx:idx+s].reshape(p.shape))
            idx += s
        out, ll = head(q, k, v)
        result = anp.sum(out) + ll
        for pa, at, orig in saved:
            setattr(pa, at, orig)
        return result

    g = autograd.grad(loss)(flat)
    assert np.linalg.norm(np.array(g)) > 1e-10


def test_moe_training_loss_decreases():
    from hybrid.interfaces.model import HybridQuantumLLM
    from classical.optimizers.trainer import HybridQuantumTrainer

    model = HybridQuantumLLM(
        vocab_size=30, d_model=16, n_layers=1, n_heads=4,
        quantum_heads_per_layer=1, d_ff=32, max_seq_length=16, dropout=0.0,
        quantum_config={
            'use_quantum_embedding': False, 'embedding_qubits': 3,
            'attention_qubits': 6, 'activation_qubits': 3,
            'use_quantum_activation': False,
            'use_moe': True, 'moe_qubit_configs': [6, 9],
            'moe_temperature': 1.0,
        })
    trainer = HybridQuantumTrainer(model, learning_rate=1e-3)
    x = np.array([[1, 2, 3, 4]])
    y = np.array([[2, 3, 4, 5]])

    first = trainer.training_step(x, y)
    for _ in range(14):
        last = trainer.training_step(x, y)
    assert last["loss"] < first["loss"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
