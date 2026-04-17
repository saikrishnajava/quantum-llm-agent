"""
Quantum Resource Manager
=========================
Tracks quantum resource usage, estimates memory, and provides
system-wide performance reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ResourceSnapshot:
    """Point-in-time resource measurement."""
    timestamp: float = 0.0
    qubits_used: int = 0
    circuit_depth: int = 0
    memory_estimate_mb: float = 0.0
    execution_time_ms: float = 0.0


class QuantumResourceManager:
    """
    Lightweight resource tracker for the hybrid transformer.

    Call ``begin()`` / ``end()`` around quantum operations to record
    usage, then ``report()`` to get aggregate statistics.
    """

    def __init__(self, max_qubits: int = 20, max_circuit_depth: int = 100):
        self.max_qubits = max_qubits
        self.max_circuit_depth = max_circuit_depth
        self._snapshots: list[ResourceSnapshot] = []
        self._active: ResourceSnapshot | None = None

    def begin(self, qubits: int, circuit_depth: int = 0) -> None:
        self._active = ResourceSnapshot(
            timestamp=time.perf_counter(),
            qubits_used=qubits,
            circuit_depth=circuit_depth,
            memory_estimate_mb=(2**qubits * 16) / (1024**2),
        )

    def end(self) -> None:
        if self._active is not None:
            self._active.execution_time_ms = (
                (time.perf_counter() - self._active.timestamp) * 1000
            )
            self._snapshots.append(self._active)
            self._active = None

    def report(self) -> dict:
        if not self._snapshots:
            return {"operations": 0}

        total_time = sum(s.execution_time_ms for s in self._snapshots)
        max_qubits = max(s.qubits_used for s in self._snapshots)
        max_mem = max(s.memory_estimate_mb for s in self._snapshots)

        return {
            "operations": len(self._snapshots),
            "total_time_ms": round(total_time, 2),
            "avg_time_ms": round(total_time / len(self._snapshots), 2),
            "peak_qubits": max_qubits,
            "peak_memory_mb": round(max_mem, 2),
        }

    def reset(self) -> None:
        self._snapshots.clear()
        self._active = None
