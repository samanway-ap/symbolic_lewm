"""Base class for the five training arms (SYMBOLIC_ABSTRACTION_PLAN.md,
Phase H.3): "Each arm is a subclass of a single MaskedTrainer with one
overridden compute_loss(batch, signals). Signals arrive via a precomputed
per-sample side table ..., not recomputed in the inner loop."

This is a minimal, model-agnostic scaffold -- it does not depend on the real
LeWM/JEPA model. That dependency belongs to the real training run (Phase H);
what's model-agnostic and worth testing NOW, before any of that exists, is
the isolation property in §15: "With masking disabled, every arm reproduces
control bit-for-bit under a fixed seed." A dummy nn.Module standing in for
JEPA is sufficient to exercise that property.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def per_sample_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(pred - target)**2, mean over the feature dim -> (B,). Shared by every
    arm so that "reduces to control when masking is disabled" is a
    structural guarantee (same function, same call) rather than a hope that
    two independently-written expressions happen to round identically."""
    return (pred - target).pow(2).mean(dim=-1)


class MaskedTrainer:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def compute_loss(self, batch: dict[str, torch.Tensor], signals: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError
