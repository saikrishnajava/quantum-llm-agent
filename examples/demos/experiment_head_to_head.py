"""
Head-to-Head Experiment: Classical vs Quantum vs MoE
=====================================================
Fair comparison — same param budget, same data, same training.
Run overnight on Linux:
    cd quantum-llm-agent
    python examples/demos/experiment_head_to_head.py

Results are printed as a table and saved to experiment_results.json.
"""

import sys
import json
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

N_EPOCHS = 20
SEQ_LEN = 16
BATCH_SIZE = 4
LR = 1e-3
D_MODEL = 32
N_LAYERS = 1
N_HEADS = 4
D_FF = 64


def build_models(vocab_size):
    configs = {
        "classical": {
            "quantum_heads_per_layer": 0,
            "quantum_config": {
                "use_quantum_embedding": False,
                "embedding_qubits": 3,
                "attention_qubits": 6,
                "activation_qubits": 3,
                "use_quantum_activation": False,
            },
        },
        "quantum_6q": {
            "quantum_heads_per_layer": 1,
            "quantum_config": {
                "use_quantum_embedding": False,
                "embedding_qubits": 3,
                "attention_qubits": 6,
                "activation_qubits": 3,
                "use_quantum_activation": False,
            },
        },
        "moe_6_9_12": {
            "quantum_heads_per_layer": 1,
            "quantum_config": {
                "use_quantum_embedding": False,
                "embedding_qubits": 3,
                "attention_qubits": 6,
                "activation_qubits": 3,
                "use_quantum_activation": False,
                "use_moe": True,
                "moe_qubit_configs": [6, 9, 12],
                "moe_temperature": 1.0,
            },
        },
    }

    models = {}
    for name, cfg in configs.items():
        model = HybridQuantumLLM(
            vocab_size=vocab_size,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            d_ff=D_FF,
            max_seq_length=64,
            dropout=0.0,
            quantum_heads_per_layer=cfg["quantum_heads_per_layer"],
            quantum_config=cfg["quantum_config"],
        )
        models[name] = model
    return models


def train_and_evaluate(name, model, loader, n_epochs):
    trainer = HybridQuantumTrainer(model, learning_rate=LR)
    params = model.count_parameters()

    print(f"\n{'='*60}")
    print(f"  {name.upper()}")
    print(f"  Params: {params['total']:,} total ({params['quantum']} quantum)")
    print(f"  Training {n_epochs} epochs, {len(loader)} batches/epoch")
    print(f"{'='*60}")

    epoch_results = []
    t_total = time.perf_counter()

    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        losses = []
        for x, y in loader:
            r = trainer.training_step(x, y)
            losses.append(r["loss"])
        elapsed = time.perf_counter() - t0
        avg_loss = float(np.mean(losses))
        ms_step = elapsed / len(losses) * 1000
        epoch_results.append({"epoch": epoch + 1, "loss": avg_loss, "ms_step": ms_step})
        print(f"  Epoch {epoch+1:2d}: loss={avg_loss:.4f} ({ms_step:.0f}ms/step, {elapsed:.1f}s)")

    total_time = time.perf_counter() - t_total

    # Generate samples
    model.eval()
    generations = {}
    for prompt in ["To be", "The fault", "Now is"]:
        ids = np.array([tokenizer.encode(prompt)])
        gen = model.generate(ids, max_new_tokens=50, temperature=0.7, do_sample=True)
        generations[prompt] = tokenizer.decode(gen[0])

    return {
        "name": name,
        "params_total": params["total"],
        "params_quantum": params["quantum"],
        "epochs": epoch_results,
        "total_time_s": round(total_time, 1),
        "final_loss": epoch_results[-1]["loss"],
        "avg_ms_step": round(np.mean([e["ms_step"] for e in epoch_results]), 0),
        "generations": generations,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("HEAD-TO-HEAD: Classical vs Quantum vs MoE")
    print("=" * 60)

    tokenizer = CharTokenizer.from_text(CORPUS)
    token_ids = tokenizer.encode(CORPUS)
    dataset = TextDataset(token_ids, SEQ_LEN)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

    print(f"Corpus: {len(CORPUS)} chars | Vocab: {tokenizer.vocab_size}")
    print(f"Dataset: {len(dataset)} sequences | {len(loader)} batches/epoch")
    print(f"Config: d_model={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}")
    print(f"Training: {N_EPOCHS} epochs, lr={LR}")

    models = build_models(tokenizer.vocab_size)
    all_results = {}

    for name, model in models.items():
        result = train_and_evaluate(name, model, loader, N_EPOCHS)
        all_results[name] = result

    # Print comparison table
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    c = all_results["classical"]
    q = all_results["quantum_6q"]
    m = all_results["moe_6_9_12"]

    print(f"""
                     Classical    Quantum 6Q   MoE (6/9/12Q)
  Total params       {c['params_total']:>8,}      {q['params_total']:>8,}      {m['params_total']:>8,}
  Quantum params     {c['params_quantum']:>8}      {q['params_quantum']:>8}      {m['params_quantum']:>8}
  Final loss         {c['final_loss']:>8.4f}      {q['final_loss']:>8.4f}      {m['final_loss']:>8.4f}
  Avg ms/step        {c['avg_ms_step']:>8.0f}      {q['avg_ms_step']:>8.0f}      {m['avg_ms_step']:>8.0f}
  Total time (s)     {c['total_time_s']:>8.1f}      {q['total_time_s']:>8.1f}      {m['total_time_s']:>8.1f}
""")

    # Loss at key epochs
    print("  Loss by epoch:")
    print(f"  {'Epoch':>7}  {'Classical':>10}  {'Quantum':>10}  {'MoE':>10}")
    for i in [0, 4, 9, 14, 19]:
        if i < len(c["epochs"]):
            print(
                f"  {i+1:>7}  {c['epochs'][i]['loss']:>10.4f}"
                f"  {q['epochs'][i]['loss']:>10.4f}"
                f"  {m['epochs'][i]['loss']:>10.4f}"
            )

    # Generation comparison
    print("\n  Generation comparison:")
    for prompt in ["To be", "The fault", "Now is"]:
        print(f'\n  Prompt: "{prompt}"')
        print(f'    Classical: "{c["generations"][prompt]}"')
        print(f'    Quantum:   "{q["generations"][prompt]}"')
        print(f'    MoE:       "{m["generations"][prompt]}"')

    # Determine winner
    losses = {
        "Classical": c["final_loss"],
        "Quantum 6Q": q["final_loss"],
        "MoE": m["final_loss"],
    }
    winner = min(losses, key=losses.get)
    print(f"\n  Winner (lowest loss): {winner} ({losses[winner]:.4f})")

    # Efficiency: loss per second of training
    efficiency = {
        "Classical": (c["epochs"][0]["loss"] - c["final_loss"]) / c["total_time_s"],
        "Quantum 6Q": (q["epochs"][0]["loss"] - q["final_loss"]) / q["total_time_s"],
        "MoE": (m["epochs"][0]["loss"] - m["final_loss"]) / m["total_time_s"],
    }
    most_efficient = max(efficiency, key=efficiency.get)
    print(f"  Most efficient (loss reduction/sec): {most_efficient} ({efficiency[most_efficient]:.6f})")

    # Save results
    out_path = Path(__file__).parent / "experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
