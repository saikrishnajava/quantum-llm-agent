"""
Performance Optimizations
==========================
Circuit caching, positional pre-computation, and profiling utilities.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CircuitResultCache:
    """
    LRU cache for quantum circuit outputs.

    Keyed by a hash of the input + parameters, so identical calls
    return a cached result instead of re-simulating the circuit.
    """

    def __init__(self, max_size: int = 2048):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, *arrays: np.ndarray) -> str:
        h = hashlib.md5()
        for a in arrays:
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def get(self, *arrays: np.ndarray) -> Optional[np.ndarray]:
        key = self._key(*arrays)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        self.misses += 1
        return None

    def put(self, result: np.ndarray, *arrays: np.ndarray) -> None:
        key = self._key(*arrays)
        self._cache[key] = result.copy()
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0


class PositionalEncodingCache:
    """
    Pre-computes and caches quantum positional encodings for all
    positions up to *max_positions*, so the quantum circuit is only
    called once per position ever.
    """

    def __init__(self, positional_circuit, max_positions: int = 512):
        self._cache: dict[int, np.ndarray] = {}
        self._circuit = positional_circuit
        self._max_positions = max_positions

    def get(self, positions: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        positions : (batch, seq_len) integer array

        Returns
        -------
        (batch, seq_len, n_qubits)
        """
        batch_size, seq_len = positions.shape
        results = []
        for b in range(batch_size):
            row = []
            for s in range(seq_len):
                pos = int(positions[b, s])
                if pos not in self._cache:
                    bits = self._circuit._position_to_bits(pos)
                    self._cache[pos] = np.array(self._circuit._circuit(bits, self._circuit.params))
                row.append(self._cache[pos])
            results.append(np.array(row))
        return np.array(results)

    def precompute(self, n_positions: int) -> None:
        """Pre-fill cache for positions 0..n_positions-1."""
        for pos in range(min(n_positions, self._max_positions)):
            if pos not in self._cache:
                bits = self._circuit._position_to_bits(pos)
                self._cache[pos] = np.array(self._circuit._circuit(bits, self._circuit.params))
        logger.info("Pre-computed %d positional encodings", len(self._cache))

    def invalidate(self) -> None:
        """Clear cache (call after parameter update during training)."""
        self._cache.clear()


class Timer:
    """Simple context-manager timer for profiling."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000

    def __repr__(self):
        return f"{self.name}: {self.elapsed_ms:.1f} ms"


def profile_model(model, input_ids: np.ndarray, n_runs: int = 5) -> dict:
    """
    Profile the model's forward pass, returning timing breakdown.
    """
    # Warmup
    model(input_ids)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(input_ids)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "n_runs": n_runs,
        "input_shape": input_ids.shape,
    }
