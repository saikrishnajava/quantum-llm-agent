"""
NumPy-based Neural Network Primitives
=======================================
Lightweight replacements for torch.nn building blocks so the entire
quantum-LLM codebase runs on PennyLane + NumPy with zero PyTorch
dependency.  All trainable parameters use ``pennylane.numpy`` so they
are automatically differentiable through PennyLane's autograd.

Provided classes:
  Parameter, Module, Linear, Embedding, LayerNorm, GELU, Dropout,
  MultiheadAttention, CrossEntropyLoss, AdamW
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import autograd.numpy as anp  # autograd-traced ops (sqrt, exp, tanh, log)
import pennylane.numpy as pnp  # autograd-aware numpy for Parameter creation


# =====================================================================
# Parameter
# =====================================================================

def Parameter(data: np.ndarray, requires_grad: bool = True) -> pnp.tensor:
    """Wrap an array as a PennyLane trainable tensor."""
    return pnp.array(data, requires_grad=requires_grad)


# =====================================================================
# Module
# =====================================================================

class Module:
    """Minimal base class for neural network modules."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self) -> list[pnp.tensor]:
        """Collect all trainable arrays from this module and children."""
        params: list[pnp.tensor] = []
        for v in self.__dict__.values():
            if isinstance(v, pnp.tensor) and getattr(v, "requires_grad", False):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> list[tuple[str, pnp.tensor]]:
        pairs: list[tuple[str, pnp.tensor]] = []
        for k, v in self.__dict__.items():
            name = f"{prefix}.{k}" if prefix else k
            if isinstance(v, pnp.tensor) and getattr(v, "requires_grad", False):
                pairs.append((name, v))
            elif isinstance(v, Module):
                pairs.extend(v.named_parameters(name))
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    if isinstance(item, Module):
                        pairs.extend(item.named_parameters(f"{name}.{i}"))
        return pairs

    def _parameter_locations(self):
        """Return (parent_module, attr_name, list_index_or_None, param) for each trainable param."""
        locations = []
        for k, v in self.__dict__.items():
            if isinstance(v, pnp.tensor) and getattr(v, "requires_grad", False):
                locations.append((self, k, None, v))
            elif isinstance(v, Module):
                locations.extend(v._parameter_locations())
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    if isinstance(item, Module):
                        locations.extend(item._parameter_locations())
        return locations

    def eval(self):
        self._training = False
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v.eval()
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        item.eval()
        return self

    def train(self, mode: bool = True):
        self._training = mode
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v.train(mode)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        item.train(mode)
        return self


# =====================================================================
# Linear
# =====================================================================

class Linear(Module):
    """y = x @ W^T + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        k = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(
            np.random.uniform(-k, k, (out_features, in_features)).astype(np.float64)
        )
        self.bias = (
            Parameter(np.random.uniform(-k, k, (out_features,)).astype(np.float64))
            if bias else None
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


# =====================================================================
# Embedding
# =====================================================================

class Embedding(Module):
    """Lookup table that maps integer indices to dense vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.weight = Parameter(
            np.random.normal(0, 0.02, (num_embeddings, embedding_dim)).astype(np.float64)
        )

    def forward(self, indices):
        idx = np.asarray(indices, dtype=int)
        one_hot = np.eye(self.num_embeddings, dtype=np.float64)[idx]
        return one_hot @ self.weight


# =====================================================================
# LayerNorm
# =====================================================================

class LayerNorm(Module):
    """Layer normalisation over the last dimension."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = Parameter(np.ones(normalized_shape, dtype=np.float64))
        self.beta = Parameter(np.zeros(normalized_shape, dtype=np.float64))

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        normed = (x - mean) / (var + self.eps) ** 0.5
        return self.gamma * normed + self.beta


# =====================================================================
# Activations
# =====================================================================

class GELU(Module):
    """Gaussian Error Linear Unit (approximate via exp-based tanh)."""

    def forward(self, x):
        z = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
        e2z = anp.exp(2.0 * z)
        tanh_z = (e2z - 1.0) / (e2z + 1.0)
        return 0.5 * x * (1.0 + tanh_z)


class Softmax:
    """Numerically stable softmax along an axis."""

    def __init__(self, axis: int = -1):
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=self.axis, keepdims=True)
        exps = anp.exp(shifted)
        return exps / exps.sum(axis=self.axis, keepdims=True)


# =====================================================================
# Dropout (no-op in numpy — applied stochastically only during training)
# =====================================================================

class Dropout(Module):
    def __init__(self, p: float = 0.1):
        self.p = p
        self._training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self._training and self.p > 0:
            mask = np.random.binomial(1, 1 - self.p, x.shape).astype(x.dtype)
            return x * mask / (1 - self.p)
        return x


# =====================================================================
# MultiheadAttention (classical, numpy)
# =====================================================================

class MultiheadAttention(Module):
    """
    Standard multi-head scaled-dot-product attention in pure numpy.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        self.softmax = Softmax(axis=-1)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Parameters
        ----------
        x : (batch, seq, embed_dim)
        mask : optional (batch, seq, seq) or broadcastable

        Returns
        -------
        (batch, seq, embed_dim)
        """
        B, S, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, S, H, D).transpose((0, 2, 1, 3))  # (B,H,S,D)
        k = self.k_proj(x).reshape(B, S, H, D).transpose((0, 2, 1, 3))
        v = self.v_proj(x).reshape(B, S, H, D).transpose((0, 2, 1, 3))

        scores = (q @ k.transpose((0, 1, 3, 2))) / self.scale  # (B,H,S,S)
        if mask is not None:
            scores = anp.where(mask, scores, -1e9)
        weights = self.softmax(scores)
        context = (weights @ v).transpose((0, 2, 1, 3)).reshape(B, S, self.embed_dim)
        return self.out_proj(context)


# =====================================================================
# Loss
# =====================================================================

class CrossEntropyLoss:
    """Cross-entropy loss for language modelling."""

    def __call__(self, logits, targets):
        """
        Parameters
        ----------
        logits  : (N, C) raw scores
        targets : (N,) integer class labels
        """
        targets = np.asarray(targets, dtype=int)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_sum_exp = anp.log(anp.exp(shifted).sum(axis=-1))
        log_probs = shifted - log_sum_exp[..., None]
        N = targets.shape[0]
        loss = -log_probs[np.arange(N), targets].mean()
        return loss


# =====================================================================
# Optimiser
# =====================================================================

class AdamW:
    """Minimal AdamW optimiser operating on a list of numpy arrays."""

    def __init__(
        self,
        params: list[pnp.tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue
            # Weight decay
            p_data = np.array(p)
            p_data -= self.lr * self.weight_decay * p_data

            # Moment updates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            p_data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p[...] = p_data

    def zero_grad(self) -> None:
        pass  # numpy arrays don't accumulate gradients like torch
