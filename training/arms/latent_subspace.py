"""latent_subspace arm (Phase H.2):
L_pred = ||P_V(zhat - z)||^2 + beta * ||P_V_perp(zhat - z)||^2

signals["P_V"] == None means "masking disabled" -- no subspace restriction
-- and must reduce, bit-for-bit, to the plain per-sample squared error used
by ControlArm regardless of beta. This is deliberately a separate code path
(not `P_V = identity_matrix` run through matmul) since multiplying by a
literal identity matrix is not guaranteed bit-exact through an arbitrary
BLAS backend, whereas skipping the projection entirely is exact by
construction.
"""
from __future__ import annotations

from training.masked_trainer import MaskedTrainer, per_sample_squared_error


class LatentSubspaceArm(MaskedTrainer):
    def __init__(self, model, beta: float = 0.1) -> None:
        super().__init__(model)
        self.beta = beta

    def compute_loss(self, batch, signals):
        pred = self.model(batch["x"])
        P_V = signals.get("P_V")
        if P_V is None:
            return per_sample_squared_error(pred, batch["y"]).mean()

        err = pred - batch["y"]
        proj = err @ P_V
        orth = err - proj
        per_sample = proj.pow(2).mean(dim=-1) + self.beta * orth.pow(2).mean(dim=-1)
        return per_sample.mean()
