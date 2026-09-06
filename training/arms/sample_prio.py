"""sample_prio arm (Phase H.2): reweight the batch loss by
edge_rarity(q_t, a_t) * (1 + kappa * abstraction_failure_flag_t).

With signals["edge_rarity"] == 1 everywhere and kappa == 0 (or
abstraction_failure_flag == 0 everywhere), the weight is exactly 1 for every
sample and this must reduce, bit-for-bit, to ControlArm's loss.
"""
from __future__ import annotations

from training.masked_trainer import MaskedTrainer, per_sample_squared_error


class SamplePrioArm(MaskedTrainer):
    def __init__(self, model, kappa: float = 1.0) -> None:
        super().__init__(model)
        self.kappa = kappa

    def compute_loss(self, batch, signals):
        pred = self.model(batch["x"])
        per_sample = per_sample_squared_error(pred, batch["y"])
        weight = signals["edge_rarity"] * (1.0 + self.kappa * signals["abstraction_failure_flag"])
        return (per_sample * weight).mean()
