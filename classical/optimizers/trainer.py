"""
Hybrid Quantum Trainer  (NumPy)
================================
"""

from __future__ import annotations

import logging
import time

import autograd
import numpy as np
import pennylane.numpy as pnp

from classical.nn import AdamW, CrossEntropyLoss
from hybrid.interfaces.model import HybridQuantumLLM

logger = logging.getLogger(__name__)


class HybridQuantumTrainer:
    def __init__(
        self,
        model: HybridQuantumLLM,
        learning_rate: float = 1e-4,
        quantum_lr_scale: float = 0.1,
        weight_decay: float = 0.01,
    ):
        self.model = model
        self._step = 0
        self.loss_fn = CrossEntropyLoss()

        self._locations = model._parameter_locations()
        self._all_params = [loc[3] for loc in self._locations]
        self._param_shapes = [p.shape for p in self._all_params]
        self._param_sizes = [p.size for p in self._all_params]

        classical_params, quantum_params = [], []
        classical_idx, quantum_idx = [], []
        for i, (name, p) in enumerate(model.named_parameters()):
            if "quantum" in name or name.endswith(".params"):
                quantum_params.append(p)
            else:
                classical_params.append(p)

        classical_ids = {id(p) for p in classical_params}
        quantum_ids = {id(p) for p in quantum_params}

        self._classical_map = []
        self._quantum_map = []
        for i, p in enumerate(self._all_params):
            pid = id(p)
            if pid in classical_ids:
                self._classical_map.append(i)
            elif pid in quantum_ids:
                self._quantum_map.append(i)

        self.classical_optimizer = AdamW(
            classical_params, lr=learning_rate, weight_decay=weight_decay,
        )
        self.quantum_optimizer = (
            AdamW(
                quantum_params,
                lr=learning_rate * quantum_lr_scale,
                weight_decay=0.001,
            )
            if quantum_params
            else None
        )

        logger.info(
            "Trainer — classical: %d, quantum: %d param tensors",
            len(classical_params),
            len(quantum_params),
        )

    def _flatten_params(self) -> np.ndarray:
        return np.concatenate([np.array(p).flatten() for p in self._all_params])

    def training_step(self, input_ids: np.ndarray, labels: np.ndarray) -> dict:
        self.model.train()
        t0 = time.perf_counter()
        locations = self._locations

        def loss_from_flat(flat_params):
            saved = []
            idx = 0
            for parent, attr_name, _, p in locations:
                size = p.size
                shape = p.shape
                saved.append((parent, attr_name, getattr(parent, attr_name)))
                setattr(parent, attr_name, flat_params[idx : idx + size].reshape(shape))
                idx += size

            logits = self.model(input_ids)
            loss = self.loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )

            # Collect MoE load-balance losses
            load_balance_loss = 0.0
            for layer in getattr(self.model, 'layers', []):
                attn = getattr(layer, 'attention', None)
                if attn and hasattr(attn, 'load_balance_losses'):
                    for lb in attn.load_balance_losses:
                        load_balance_loss = load_balance_loss + lb
                    attn.load_balance_losses = []
            if load_balance_loss != 0.0:
                loss = loss + 0.01 * load_balance_loss

            for parent, attr_name, original in saved:
                setattr(parent, attr_name, original)

            return loss

        flat_params = self._flatten_params()
        loss_and_grad = autograd.value_and_grad(loss_from_flat)
        loss_val, flat_grad = loss_and_grad(flat_params)

        total_norm = float(np.linalg.norm(np.array(flat_grad)))
        max_grad_norm = 1.0
        if total_norm > max_grad_norm and total_norm > 0:
            flat_grad = flat_grad * (max_grad_norm / total_norm)

        idx = 0
        all_grads = []
        for shape, size in zip(self._param_shapes, self._param_sizes):
            all_grads.append(np.array(flat_grad[idx : idx + size]).reshape(shape))
            idx += size

        classical_grads = [all_grads[i] for i in self._classical_map]
        self.classical_optimizer.step(classical_grads)

        if self.quantum_optimizer and self._quantum_map:
            quantum_grads = [all_grads[i] for i in self._quantum_map]
            self.quantum_optimizer.step(quantum_grads)

        elapsed = time.perf_counter() - t0
        self._step += 1
        return {
            "step": self._step,
            "loss": float(loss_val),
            "grad_norm": total_norm,
            "elapsed_ms": elapsed * 1000,
        }

    def evaluate(self, input_ids: np.ndarray, labels: np.ndarray) -> dict:
        self.model.eval()
        logits = self.model(input_ids)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        loss = self.loss_fn(flat_logits, flat_labels)
        preds = logits.argmax(axis=-1)
        accuracy = float((preds == labels).mean())
        return {"eval_loss": float(loss), "accuracy": accuracy}
