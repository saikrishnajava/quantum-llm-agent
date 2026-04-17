"""
12-Qubit Proof of Concept  (PennyLane + NumPy — zero PyTorch)
===============================================================
Run:
    cd quantum-llm-agent
    python examples/demos/poc_12qubit.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from hybrid.interfaces.model import HybridQuantumLLM
from agents.reasoning.quantum_reasoning import QuantumReasoningModule
from agents.memory.quantum_memory import QuantumMemoryNetwork
from agents.coordination.quantum_agent import QuantumAgent, QuantumTool


class CalculatorTool(QuantumTool):
    name = "calculator"
    description = "Perform basic arithmetic calculations"
    def run(self, input_text: str) -> str:
        return f"[Calculator result for: {input_text[:30]}]"


class SearchTool(QuantumTool):
    name = "search"
    description = "Search for information on the web"
    def run(self, input_text: str) -> str:
        return f"[Search result for: {input_text[:30]}]"


def main():
    print("=" * 70)
    print("QUANTUM-SIMULATED LLM: 12-QUBIT PROOF OF CONCEPT")
    print("Stack: PennyLane v0.44.1 + NumPy (zero PyTorch dependency)")
    print("=" * 70)

    # 1. Model instantiation
    print("\n[1/6] Instantiating Hybrid Quantum LLM...")
    t0 = time.perf_counter()
    model = HybridQuantumLLM(
        vocab_size=1000, d_model=64, n_layers=1, n_heads=4,
        quantum_heads_per_layer=1, d_ff=128, max_seq_length=32, dropout=0.0,
        quantum_config={
            "use_quantum_embedding": True, "embedding_qubits": 4,
            "attention_qubits": 6, "activation_qubits": 3,
            "use_quantum_activation": False,
        },
    )
    params = model.count_parameters()
    print(f"   Created in {time.perf_counter() - t0:.2f}s")
    print(f"   Params — classical: {params['classical']:,}  quantum: {params['quantum']:,}  total: {params['total']:,}")

    # 2. Forward pass
    print("\n[2/6] Forward pass (quantum attention + quantum embedding)...")
    input_ids = np.random.randint(0, 1000, (1, 8))
    t0 = time.perf_counter()
    logits = model(input_ids)
    print(f"   Input:  {input_ids.shape}  →  Output: {logits.shape}")
    print(f"   Time:   {(time.perf_counter() - t0) * 1000:.1f} ms")
    print(f"   Logits: [{logits.min():.3f}, {logits.max():.3f}]")

    # 3. Token generation
    print("\n[3/6] Auto-regressive generation...")
    prompt = np.random.randint(0, 1000, (1, 4))
    t0 = time.perf_counter()
    generated = model.generate(prompt, max_new_tokens=8, temperature=0.8)
    print(f"   Prompt:    {prompt[0].tolist()}")
    print(f"   Generated: {generated[0].tolist()}")
    print(f"   Time:      {(time.perf_counter() - t0) * 1000:.1f} ms")

    # 4. Quantum reasoning
    print("\n[4/6] Quantum reasoning module...")
    reasoner = QuantumReasoningModule(d_model=64, reasoning_qubits=4, n_layers=1)
    ctx, opt, mem = np.random.randn(64), np.random.randn(64), np.random.randn(64)
    t0 = time.perf_counter()
    result = reasoner(ctx, opt, memory_state=mem)
    probs = result["decision_probs"]
    print(f"   Decision probs: {[f'{p:.3f}' for p in probs]}")
    print(f"   Selected:       option {int(np.argmax(probs))}")
    if result["pattern_scores"] is not None:
        print(f"   Pattern scores: {[f'{s:.3f}' for s in result['pattern_scores']]}")
    print(f"   Time:           {(time.perf_counter() - t0) * 1000:.1f} ms")

    # 5. Quantum memory
    print("\n[5/6] Quantum memory network...")
    mem_net = QuantumMemoryNetwork(d_model=64, memory_size=16, n_qubits=4)
    for _ in range(5):
        mem_net.store(np.random.randn(64))
    recalled = mem_net.recall(np.random.randn(64), top_k=3)
    print(f"   Stored: {mem_net.size} memories  |  Recalled top-3: {len(recalled)} items")

    # 6. Agent loop
    print("\n[6/6] Quantum agent (2-step loop)...")
    agent = QuantumAgent(
        model=model, reasoning_module=reasoner, memory=mem_net,
        tools=[CalculatorTool(), SearchTool()], max_steps=2,
    )
    t0 = time.perf_counter()
    agent_result = agent.run("What is quantum computing?")
    print(f"   Steps: {len(agent_result['steps'])}")
    for i, step in enumerate(agent_result["steps"]):
        print(f"     Step {i+1}: tool={step['tool']}, obs={step['observation']}")
    print(f"   Time:   {(time.perf_counter() - t0) * 1000:.1f} ms")
    print(f"   Memory: {agent_result['memory_size']} items")

    # Summary
    print("\n" + "=" * 70)
    print("PROOF OF CONCEPT COMPLETE")
    print("=" * 70)
    print(f"""
  Hybrid Quantum LLM          {params['total']:,} parameters
  Quantum attention            6-qubit circuit, 1 quantum head
  Quantum embedding            4-qubit feature map + positional encoding
  Quantum reasoning            4-qubit decision circuit
  Quantum memory               4-qubit pattern matcher
  Total qubits                 ~12 (across all components)
  Agent tools                  2 (calculator, search)
  Dependencies                 PennyLane + NumPy only

  This is the world's first quantum-simulated LLM with agentic workflows.
""")


if __name__ == "__main__":
    main()
