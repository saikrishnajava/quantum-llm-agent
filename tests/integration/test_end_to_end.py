"""
End-to-end integration tests.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hybrid.interfaces.model import HybridQuantumLLM
from classical.optimizers.trainer import HybridQuantumTrainer
from classical.tokenizer import CharTokenizer
from classical.data import prepare_data


def test_training_loss_decreases():
    model = HybridQuantumLLM(
        vocab_size=50, d_model=16, n_layers=1, n_heads=4,
        quantum_heads_per_layer=0, d_ff=32, max_seq_length=16, dropout=0.0,
        quantum_config={
            "use_quantum_embedding": False, "embedding_qubits": 3,
            "attention_qubits": 6, "activation_qubits": 3,
            "use_quantum_activation": False,
        },
    )
    trainer = HybridQuantumTrainer(model, learning_rate=1e-3)
    x = np.array([[1, 2, 3, 4, 5, 6]])
    y = np.array([[2, 3, 4, 5, 6, 7]])

    first = trainer.training_step(x, y)
    for _ in range(14):
        last = trainer.training_step(x, y)

    assert last["loss"] < first["loss"], (
        f"loss did not decrease: {first['loss']:.4f} -> {last['loss']:.4f}"
    )
    assert last["grad_norm"] > 0


def test_generation_produces_valid_tokens():
    model = HybridQuantumLLM(
        vocab_size=100, d_model=16, n_layers=1, n_heads=4,
        quantum_heads_per_layer=0, d_ff=32, max_seq_length=16,
        quantum_config={
            "use_quantum_embedding": False, "embedding_qubits": 3,
            "attention_qubits": 6, "activation_qubits": 3,
            "use_quantum_activation": False,
        },
    )
    prompt = np.array([[1, 2, 3, 4]])
    generated = model.generate(prompt, max_new_tokens=5, do_sample=False)
    assert generated.shape == (1, 9)
    assert np.all(generated >= 0)
    assert np.all(generated < 100)


def test_tokenizer_roundtrip():
    text = "Hello, quantum world!"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_data_pipeline_shapes():
    corpus = "abcdef" * 50
    tok = CharTokenizer.from_text(corpus)
    train_loader, val_loader, _ = prepare_data(
        corpus, seq_len=8, batch_size=4, tokenizer=tok,
    )
    for x, y in train_loader:
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)
        break


def test_agent_with_tokenizer():
    from agents.coordination.quantum_agent import QuantumAgent, QuantumTool
    from agents.reasoning.quantum_reasoning import QuantumReasoningModule
    from agents.memory.quantum_memory import QuantumMemoryNetwork

    model = HybridQuantumLLM(
        vocab_size=128, d_model=32, n_layers=1, n_heads=4,
        quantum_heads_per_layer=0, d_ff=64, max_seq_length=32,
        quantum_config={
            "use_quantum_embedding": False, "embedding_qubits": 3,
            "attention_qubits": 6, "activation_qubits": 3,
            "use_quantum_activation": False,
        },
    )

    class DummyTool(QuantumTool):
        name = "dummy"
        description = "A test tool"
        def run(self, input_text):
            return "FINAL ANSWER: done"

    reasoner = QuantumReasoningModule(d_model=32, reasoning_qubits=4, n_layers=1)
    memory = QuantumMemoryNetwork(d_model=32, memory_size=8, n_qubits=4)
    tok = CharTokenizer.from_text("abcdefghijklmnopqrstuvwxyz 0123456789")

    agent = QuantumAgent(model, reasoner, memory, tools=[DummyTool()], max_steps=2)
    result = agent.run("test query", tokenizer=tok)

    assert "answer" in result
    assert "steps" in result
    assert len(result["steps"]) > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
