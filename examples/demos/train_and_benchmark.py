"""
Training & Benchmarking Pipeline
==================================
Trains the quantum LLM on a small embedded corpus and benchmarks
quantum vs classical variants side-by-side.

Run:
    cd quantum-llm-agent
    python examples/demos/train_and_benchmark.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore", category=ComplexWarning)

from classical.nn import CrossEntropyLoss, Softmax
from classical.tokenizer import CharTokenizer
from classical.data import TextDataset, DataLoader
from classical.optimizers.trainer import HybridQuantumTrainer
from hybrid.interfaces.model import HybridQuantumLLM
from hybrid.interfaces.optimizations import Timer, profile_model

# =====================================================================
# Embedded training corpus
# =====================================================================

CORPUS = """
To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them To die to sleep
No more and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to Tis a consummation
Devoutly to be wished To die to sleep
To sleep perchance to dream ay there s the rub
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil
Must give us pause There s the respect
That makes calamity of so long life
All that glitters is not gold
The fault dear Brutus is not in our stars
But in ourselves that we are underlings
Friends Romans countrymen lend me your ears
I come to bury Caesar not to praise him
The evil that men do lives after them
The good is oft interred with their bones
Now is the winter of our discontent
Made glorious summer by this sun of York
A horse a horse my kingdom for a horse
Out out brief candle life is but a walking shadow
A poor player that struts and frets his hour upon the stage
And then is heard no more it is a tale told by an idiot
Full of sound and fury signifying nothing
Double double toil and trouble fire burn and cauldron bubble
Something wicked this way comes
We are such stuff as dreams are made on
And our little life is rounded with a sleep
""".strip()


# =====================================================================
# Build models
# =====================================================================

def build_quantum_model(vocab_size: int) -> HybridQuantumLLM:
    return HybridQuantumLLM(
        vocab_size=vocab_size, d_model=32, n_layers=1, n_heads=4,
        quantum_heads_per_layer=1, d_ff=64, max_seq_length=32, dropout=0.0,
        quantum_config={
            "use_quantum_embedding": False, "embedding_qubits": 3,
            "attention_qubits": 6, "activation_qubits": 3,
            "use_quantum_activation": False,
        },
    )


def build_classical_model(vocab_size: int) -> HybridQuantumLLM:
    return HybridQuantumLLM(
        vocab_size=vocab_size, d_model=32, n_layers=1, n_heads=4,
        quantum_heads_per_layer=0, d_ff=64, max_seq_length=32, dropout=0.0,
        quantum_config={
            "use_quantum_embedding": False, "embedding_qubits": 3,
            "attention_qubits": 6, "activation_qubits": 3,
            "use_quantum_activation": False,
        },
    )


# =====================================================================
# Generation
# =====================================================================

def generate_text(model, tokenizer, prompt: str, max_tokens: int = 50) -> str:
    ids = np.array([tokenizer.encode(prompt)])
    generated = model.generate(ids, max_new_tokens=max_tokens, temperature=0.8)
    return tokenizer.decode(generated[0])


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("QUANTUM LLM — TRAINING & BENCHMARK PIPELINE")
    print("=" * 70)

    # --- Tokenizer & data ---
    tokenizer = CharTokenizer.from_text(CORPUS)
    token_ids = tokenizer.encode(CORPUS)
    print(f"\nCorpus: {len(CORPUS)} chars, {tokenizer.vocab_size} unique tokens")

    SEQ_LEN = 8
    BATCH_SIZE = 4
    N_EPOCHS = 5
    dataset = TextDataset(token_ids, SEQ_LEN)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    print(f"Dataset: {len(dataset)} sequences, {len(loader)} batches/epoch")

    # --- Build models ---
    print("\n--- Building Models ---")
    c_model = build_classical_model(tokenizer.vocab_size)
    q_model = build_quantum_model(tokenizer.vocab_size)
    c_params = c_model.count_parameters()
    q_params = q_model.count_parameters()
    print(f"Classical model: {c_params['total']:,} params ({c_params['quantum']} quantum)")
    print(f"Quantum model:   {q_params['total']:,} params ({q_params['quantum']} quantum)")

    loss_fn = CrossEntropyLoss()

    # --- Benchmark: Forward Pass Latency ---
    print("\n--- Forward Pass Latency ---")
    test_input = np.random.randint(0, tokenizer.vocab_size, (1, SEQ_LEN))

    with Timer("Classical forward") as tc:
        for _ in range(3):
            c_model(test_input)
    classical_ms = tc.elapsed_ms / 3

    with Timer("Quantum forward") as tq:
        for _ in range(3):
            q_model(test_input)
    quantum_ms = tq.elapsed_ms / 3

    print(f"  Classical:  {classical_ms:.1f} ms/forward")
    print(f"  Quantum:    {quantum_ms:.1f} ms/forward")
    print(f"  Overhead:   {quantum_ms / classical_ms:.1f}x")

    # --- Train Classical Model ---
    print(f"\n--- Training Classical Model ({N_EPOCHS} epochs, gradient descent) ---")
    c_trainer = HybridQuantumTrainer(c_model, learning_rate=1e-3)
    for epoch in range(N_EPOCHS):
        t0 = time.perf_counter()
        epoch_losses = []
        for x, y in loader:
            result = c_trainer.training_step(x, y)
            epoch_losses.append(result["loss"])
        elapsed = time.perf_counter() - t0
        print(f"  Epoch {epoch+1}: loss={np.mean(epoch_losses):.4f}  ({len(epoch_losses)} batches, {elapsed:.1f}s)")

    c_eval = c_trainer.evaluate(
        np.array([token_ids[:SEQ_LEN]]),
        np.array([token_ids[1 : SEQ_LEN + 1]]),
    )
    print(f"  Eval: loss={c_eval['eval_loss']:.4f}  accuracy={c_eval['accuracy']:.2%}")

    # --- Train Quantum Model (fewer epochs — slower) ---
    Q_EPOCHS = 2
    print(f"\n--- Training Quantum Model ({Q_EPOCHS} epochs, gradient descent) ---")
    q_trainer = HybridQuantumTrainer(q_model, learning_rate=1e-3)
    for epoch in range(Q_EPOCHS):
        t0 = time.perf_counter()
        epoch_losses = []
        for batch_i, (x, y) in enumerate(loader):
            result = q_trainer.training_step(x, y)
            epoch_losses.append(result["loss"])
            if batch_i % 20 == 0 and batch_i > 0:
                print(f"    batch {batch_i}/{len(loader)}: loss={result['loss']:.4f}")
        elapsed = time.perf_counter() - t0
        print(f"  Epoch {epoch+1}: loss={np.mean(epoch_losses):.4f}  ({len(epoch_losses)} batches, {elapsed:.1f}s)")

    q_eval = q_trainer.evaluate(
        np.array([token_ids[:SEQ_LEN]]),
        np.array([token_ids[1 : SEQ_LEN + 1]]),
    )
    print(f"  Eval: loss={q_eval['eval_loss']:.4f}  accuracy={q_eval['accuracy']:.2%}")

    # --- Generate Text ---
    print("\n--- Text Generation ---")
    prompt = "To be or"
    print(f"  Prompt: '{prompt}'")

    c_model.eval()
    q_model.eval()

    with Timer("Classical gen") as tc_gen:
        c_text = generate_text(c_model, tokenizer, prompt)
    print(f"  Classical ({tc_gen.elapsed_ms:.0f}ms): '{c_text}'")

    with Timer("Quantum gen") as tq_gen:
        q_text = generate_text(q_model, tokenizer, prompt)
    print(f"  Quantum  ({tq_gen.elapsed_ms:.0f}ms): '{q_text}'")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"""
                        Classical       Quantum         Delta
  Parameters            {c_params['total']:>8,}       {q_params['total']:>8,}       {q_params['total'] - c_params['total']:+,}
  Quantum params               0            {q_params['quantum']:>4}
  Forward (ms)          {classical_ms:>8.1f}       {quantum_ms:>8.1f}       {quantum_ms/classical_ms:.1f}x
  Eval loss             {c_eval['eval_loss']:>8.4f}       {q_eval['eval_loss']:>8.4f}       {q_eval['eval_loss'] - c_eval['eval_loss']:+.4f}
  Eval accuracy         {c_eval['accuracy']:>7.2%}       {q_eval['accuracy']:>7.2%}       {q_eval['accuracy'] - c_eval['accuracy']:+.2%}

  Note: Phase 1 proof-of-concept (6-qubit circuits, CPU simulation).
  Quantum overhead is expected. Run on GPU with lightning.gpu for speed.
""")


if __name__ == "__main__":
    main()
