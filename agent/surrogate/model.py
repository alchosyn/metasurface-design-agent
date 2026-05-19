"""
Lightweight MLP surrogate model for the agent.

Architecture matches ml/surrogate_optimization.py ForwardSurrogate but with
smaller default hidden size (64 vs 128) and mandatory dropout for MC
uncertainty estimation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ForwardSurrogate(nn.Module):
    """MLP: (L, W, h) -> (sin dphi, cos dphi, T_TE, T_TM).

    Dropout is enabled by default (0.1) to support MC Dropout uncertainty.
    """

    def __init__(self, hidden: int = 64, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(hidden, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StandardScaler:
    """Zero-mean, unit-variance normaliser (matches ml/surrogate_optimization.py)."""

    def __init__(self, data: np.ndarray | None = None):
        if data is not None:
            self.mean = data.mean(axis=0).astype(np.float32)
            self.std = data.std(axis=0).astype(np.float32)
            self.std[self.std < 1e-8] = 1.0
        else:
            self.mean = np.zeros(1, dtype=np.float32)
            self.std = np.ones(1, dtype=np.float32)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> StandardScaler:
        obj = cls.__new__(cls)
        obj.mean = np.asarray(state["mean"], dtype=np.float32)
        obj.std = np.asarray(state["std"], dtype=np.float32)
        return obj
