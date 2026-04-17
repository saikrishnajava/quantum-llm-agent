"""
Quantum Backend Manager
=======================
Manages quantum device selection, initialization, and resource tracking.
Provides seamless switching between simulators and hardware backends.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pennylane as qml

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


@dataclass
class QuantumDeviceStats:
    """Tracks quantum device usage statistics."""
    quantum_operations: int = 0
    classical_fallbacks: int = 0
    total_quantum_time: float = 0.0
    total_classical_time: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.quantum_operations + self.classical_fallbacks
        return self.quantum_operations / total if total > 0 else 0.0

    @property
    def avg_quantum_time(self) -> float:
        return (
            self.total_quantum_time / self.quantum_operations
            if self.quantum_operations > 0
            else 0.0
        )


@dataclass
class BackendConfig:
    """Configuration for a single quantum backend."""
    device: str
    max_qubits: int
    supports_gradients: bool = True
    diff_method: str = "backprop"
    description: str = ""


class QuantumBackendManager:
    """
    Central manager for quantum backend lifecycle.

    Responsibilities:
      - Load backend configs from YAML
      - Create PennyLane devices on demand with caching
      - Enforce resource limits (qubit count, circuit depth, memory)
      - Track performance statistics
      - Provide classical fallback execution
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._configs: dict[str, BackendConfig] = {}
        self._active_backend: str = "default_qubit"
        self._device_cache: dict[str, qml.Device] = {}
        self._stats = QuantumDeviceStats()

        # Resource limits
        self._max_qubits: int = 20
        self._max_circuit_depth: int = 100
        self._max_memory_gb: float = 8.0
        self._cache_size: int = 500

        if config_path is None:
            config_path = CONFIG_DIR / "quantum_backends.yaml"
        if config_path.exists() and yaml is not None:
            self._load_config(config_path)
        else:
            self._register_defaults()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _load_config(self, path: Path) -> None:
        with open(path) as f:
            raw = yaml.safe_load(f)

        for name, cfg in raw.get("backends", {}).items():
            self._configs[name] = BackendConfig(
                device=cfg["device"],
                max_qubits=cfg.get("max_qubits", 20),
                supports_gradients=cfg.get("supports_gradients", True),
                diff_method=cfg.get("diff_method", "backprop"),
                description=cfg.get("description", ""),
            )

        self._active_backend = raw.get("active_backend", "default_qubit")

        limits = raw.get("resource_limits", {})
        self._max_qubits = limits.get("max_qubits", 20)
        self._max_circuit_depth = limits.get("max_circuit_depth", 100)
        self._max_memory_gb = limits.get("max_quantum_memory_gb", 8.0)
        self._cache_size = limits.get("circuit_cache_size", 500)

        logger.info(
            "Loaded quantum backends: %s (active: %s)",
            list(self._configs),
            self._active_backend,
        )

    def _register_defaults(self) -> None:
        self._configs["default_qubit"] = BackendConfig(
            device="default.qubit",
            max_qubits=23,
            supports_gradients=True,
            diff_method="backprop",
            description="PennyLane default statevector simulator",
        )
        self._active_backend = "default_qubit"

    # ------------------------------------------------------------------
    # Device Management
    # ------------------------------------------------------------------

    def get_device(self, n_wires: int, backend: Optional[str] = None) -> qml.Device:
        """
        Return a PennyLane device with *n_wires* qubits.

        Devices are cached by (backend_name, n_wires).  If the requested
        qubit count exceeds the backend or resource limits, a ValueError
        is raised.
        """
        backend = backend or self._active_backend
        cfg = self._configs.get(backend)
        if cfg is None:
            raise ValueError(f"Unknown backend: {backend!r}")

        if n_wires > cfg.max_qubits:
            raise ValueError(
                f"Backend {backend!r} supports at most {cfg.max_qubits} qubits, "
                f"but {n_wires} were requested."
            )
        if n_wires > self._max_qubits:
            raise ValueError(
                f"Resource limit allows at most {self._max_qubits} qubits, "
                f"but {n_wires} were requested."
            )

        memory_gb = self._estimate_memory_gb(n_wires)
        if memory_gb > self._max_memory_gb:
            raise ValueError(
                f"{n_wires} qubits require ~{memory_gb:.1f} GB, "
                f"exceeding the {self._max_memory_gb:.1f} GB limit."
            )

        cache_key = (backend, n_wires)
        if cache_key not in self._device_cache:
            self._device_cache[cache_key] = qml.device(cfg.device, wires=n_wires)
            logger.debug("Created device %s with %d wires", cfg.device, n_wires)
        return self._device_cache[cache_key]

    @property
    def active_diff_method(self) -> str:
        cfg = self._configs.get(self._active_backend)
        return cfg.diff_method if cfg else "backprop"

    # ------------------------------------------------------------------
    # Resource Estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_memory_gb(n_qubits: int) -> float:
        """Estimate memory for a full statevector (complex128)."""
        return (2**n_qubits * 16) / (1024**3)

    def can_execute_quantum(
        self, n_qubits: int, circuit_depth: int = 0
    ) -> bool:
        """Check whether the given quantum workload is feasible."""
        if n_qubits > self._max_qubits:
            return False
        if circuit_depth > self._max_circuit_depth:
            return False
        if self._estimate_memory_gb(n_qubits) > self._max_memory_gb:
            return False
        return True

    # ------------------------------------------------------------------
    # Execution with Fallback
    # ------------------------------------------------------------------

    def execute_with_fallback(self, quantum_fn, classical_fn, *args, **kwargs):
        """
        Try *quantum_fn*; on failure fall back to *classical_fn*.
        Both callables receive the same positional and keyword arguments.
        """
        try:
            t0 = time.perf_counter()
            result = quantum_fn(*args, **kwargs)
            self._stats.total_quantum_time += time.perf_counter() - t0
            self._stats.quantum_operations += 1
            return result
        except Exception as exc:
            logger.warning("Quantum execution failed (%s), using classical fallback.", exc)
            t0 = time.perf_counter()
            result = classical_fn(*args, **kwargs)
            self._stats.total_classical_time += time.perf_counter() - t0
            self._stats.classical_fallbacks += 1
            return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> QuantumDeviceStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = QuantumDeviceStats()

    def summary(self) -> dict:
        return {
            "active_backend": self._active_backend,
            "max_qubits": self._max_qubits,
            "max_circuit_depth": self._max_circuit_depth,
            "max_memory_gb": self._max_memory_gb,
            "stats": {
                "quantum_ops": self._stats.quantum_operations,
                "classical_fallbacks": self._stats.classical_fallbacks,
                "quantum_success_rate": f"{self._stats.success_rate:.2%}",
                "avg_quantum_time_ms": f"{self._stats.avg_quantum_time * 1000:.2f}",
            },
        }


# Module-level singleton
_global_manager: Optional[QuantumBackendManager] = None


def get_backend_manager() -> QuantumBackendManager:
    """Return (and lazily create) the global backend manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = QuantumBackendManager()
    return _global_manager
