"""
Qubit Scaling Configuration
=============================
Pre-defined configurations for scaling from 12 → 20 → 30 qubits.

Run:
    cd quantum-llm-agent
    python examples/demos/scaling_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hybrid.interfaces.model import HybridQuantumLLM
from hybrid.interfaces.optimizations import Timer, profile_model

CONFIGS = {
    "12-qubit (Phase 1 PoC)": dict(
        vocab_size=500, d_model=64, n_layers=1, n_heads=4,
        quantum_heads_per_layer=1, d_ff=128, max_seq_length=32,
        quantum_config={
            "use_quantum_embedding": True, "embedding_qubits": 4,
            "attention_qubits": 6, "activation_qubits": 3,
            "use_quantum_activation": False,
        },
    ),
    "16-qubit (Phase 1+)": dict(
        vocab_size=500, d_model=128, n_layers=1, n_heads=4,
        quantum_heads_per_layer=1, d_ff=256, max_seq_length=32,
        quantum_config={
            "use_quantum_embedding": True, "embedding_qubits": 6,
            "attention_qubits": 6, "activation_qubits": 4,
            "use_quantum_activation": False,
        },
    ),
    "20-qubit (Phase 2 entry)": dict(
        vocab_size=500, d_model=128, n_layers=2, n_heads=4,
        quantum_heads_per_layer=1, d_ff=256, max_seq_length=32,
        quantum_config={
            "use_quantum_embedding": True, "embedding_qubits": 8,
            "attention_qubits": 9, "activation_qubits": 4,
            "use_quantum_activation": False,
        },
    ),
    "24-qubit (Phase 2 mid)": dict(
        vocab_size=500, d_model=256, n_layers=2, n_heads=8,
        quantum_heads_per_layer=2, d_ff=512, max_seq_length=32,
        quantum_config={
            "use_quantum_embedding": True, "embedding_qubits": 10,
            "attention_qubits": 9, "activation_qubits": 6,
            "use_quantum_activation": False,
        },
    ),
}


def main():
    print("=" * 70)
    print("QUANTUM LLM — QUBIT SCALING ANALYSIS")
    print("=" * 70)

    results = []
    for name, cfg in CONFIGS.items():
        print(f"\n--- {name} ---")
        t0 = time.perf_counter()

        try:
            model = HybridQuantumLLM(**cfg)
            build_time = (time.perf_counter() - t0) * 1000
            params = model.count_parameters()
            print(f"  Build:    {build_time:.0f} ms")
            print(f"  Params:   {params['total']:,} (quantum: {params['quantum']})")

            # Forward pass benchmark
            test_input = np.random.randint(0, cfg["vocab_size"], (1, 8))
            with Timer("forward") as t:
                model(test_input)
            print(f"  Forward:  {t.elapsed_ms:.1f} ms (1 batch, seq=8)")

            # Memory estimate
            max_qubits = max(
                cfg["quantum_config"].get("embedding_qubits", 0),
                cfg["quantum_config"].get("attention_qubits", 0),
                cfg["quantum_config"].get("activation_qubits", 0),
            )
            mem_mb = (2**max_qubits * 16) / (1024**2)
            print(f"  Peak mem: {mem_mb:.2f} MB (statevector for {max_qubits} qubits)")

            results.append({
                "name": name, "params": params["total"],
                "quantum_params": params["quantum"],
                "forward_ms": t.elapsed_ms, "peak_qubits": max_qubits,
                "mem_mb": mem_mb,
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"name": name, "error": str(e)})

    # Summary table
    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)
    print(f"{'Config':<28} {'Params':>10} {'Q-Params':>10} {'Forward':>10} {'Peak Q':>8} {'Mem':>10}")
    print("-" * 78)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<28} {'FAILED':>10}")
        else:
            print(f"{r['name']:<28} {r['params']:>10,} {r['quantum_params']:>10} "
                  f"{r['forward_ms']:>8.0f}ms {r['peak_qubits']:>6}q {r['mem_mb']:>8.2f}MB")


if __name__ == "__main__":
    main()
