"""
MoE Training Demo
==================
Train the adaptive qubit MoE model on Shakespeare.

Run:
    cd quantum-llm-agent
    python examples/demos/train_moe.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hybrid.interfaces.model import HybridQuantumLLM
from classical.optimizers.trainer import HybridQuantumTrainer
from classical.tokenizer import CharTokenizer
from classical.data import TextDataset, DataLoader

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


def main():
    print("=" * 60)
    print("ADAPTIVE QUBIT MoE — TRAINING DEMO")
    print("=" * 60)

    tokenizer = CharTokenizer.from_text(CORPUS)
    token_ids = tokenizer.encode(CORPUS)
    print(f"Corpus: {len(CORPUS)} chars, vocab: {tokenizer.vocab_size}")

    ds = TextDataset(token_ids, seq_len=8)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    print(f"Batches per epoch: {len(dl)}")

    model = HybridQuantumLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_layers=1, n_heads=4,
        quantum_heads_per_layer=1,
        d_ff=64, max_seq_length=32, dropout=0.0,
        quantum_config={
            "use_quantum_embedding": False,
            "embedding_qubits": 3,
            "attention_qubits": 6,
            "activation_qubits": 3,
            "use_quantum_activation": False,
            "use_moe": True,
            "moe_qubit_configs": [6, 9, 12],
            "moe_temperature": 1.0,
        },
    )
    params = model.count_parameters()
    print(f"Model: {params['total']:,} params ({params['quantum']} quantum)")
    print(f"MoE experts: 6-qubit, 9-qubit, 12-qubit")

    trainer = HybridQuantumTrainer(model, learning_rate=1e-3)

    N_EPOCHS = 5
    print(f"\n--- Training ({N_EPOCHS} epochs) ---")
    t_total = time.perf_counter()

    for epoch in range(N_EPOCHS):
        t0 = time.perf_counter()
        losses = []
        for x, y in dl:
            r = trainer.training_step(x, y)
            losses.append(r["loss"])
        elapsed = time.perf_counter() - t0
        ms_step = elapsed / len(losses) * 1000
        print(f"  Epoch {epoch+1}: loss={np.mean(losses):.4f} ({ms_step:.0f}ms/step, {elapsed:.1f}s)")

    total = time.perf_counter() - t_total
    print(f"\nTotal: {total:.1f}s ({total/60:.1f} min)")

    # Generate
    model.eval()
    prompt = "To be"
    ids = np.array([tokenizer.encode(prompt)])
    gen = model.generate(ids, max_new_tokens=40, temperature=0.7, do_sample=True)
    print(f"\nPrompt: '{prompt}'")
    print(f"Output: '{tokenizer.decode(gen[0])}'")


if __name__ == "__main__":
    main()
