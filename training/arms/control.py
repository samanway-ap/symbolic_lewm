"""Control arm (Phase H.2): unmodified objective, no reweighting, no mask
bias. Every other arm's `compute_loss` must reduce to this exact expression
when its intervention is disabled -- that equivalence is what
tests/test_arm_isolation.py checks."""
from __future__ import annotations

from training.masked_trainer import MaskedTrainer, per_sample_squared_error


class ControlArm(MaskedTrainer):
    def compute_loss(self, batch, signals):
        pred = self.model(batch["x"])
        return per_sample_squared_error(pred, batch["y"]).mean()
