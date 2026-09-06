"""§15 'Arm isolation': with masking disabled, every arm reproduces control
bit-for-bit under a fixed seed.

Uses a tiny dummy nn.Module standing in for the real JEPA/LeWM model -- the
isolation property is about the LOSS/MASK plumbing (Phase H.3), not about
the model architecture, so it does not need the real trained model.
"""
from __future__ import annotations

import torch
from torch import nn

from training.arms.control import ControlArm
from training.arms.latent_subspace import LatentSubspaceArm
from training.arms.sample_prio import SamplePrioArm


def _make_model_and_batch(seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
    batch = {"x": torch.randn(32, 8), "y": torch.randn(32, 8)}
    return model, batch


def test_sample_prio_matches_control_when_disabled():
    model, batch = _make_model_and_batch(seed=1)
    control = ControlArm(model)
    variant = SamplePrioArm(model, kappa=1.0)

    disabled_signals = {
        "edge_rarity": torch.ones(batch["x"].shape[0]),
        "abstraction_failure_flag": torch.zeros(batch["x"].shape[0]),
    }

    control_loss = control.compute_loss(batch, {})
    variant_loss = variant.compute_loss(batch, disabled_signals)

    assert torch.equal(control_loss, variant_loss), (
        f"control={control_loss.item()!r} vs sample_prio(disabled)={variant_loss.item()!r}"
    )


def test_sample_prio_with_kappa_zero_also_matches_regardless_of_flags():
    """kappa=0 must disable the abstraction-failure term entirely, even if
    the flags themselves are nonzero -- a second, independent way to reach
    the disabled state."""
    model, batch = _make_model_and_batch(seed=2)
    control = ControlArm(model)
    variant = SamplePrioArm(model, kappa=0.0)

    signals = {
        "edge_rarity": torch.ones(batch["x"].shape[0]),
        "abstraction_failure_flag": torch.ones(batch["x"].shape[0]),  # nonzero, but kappa=0
    }

    assert torch.equal(control.compute_loss(batch, {}), variant.compute_loss(batch, signals))


def test_latent_subspace_matches_control_when_disabled():
    model, batch = _make_model_and_batch(seed=3)
    control = ControlArm(model)
    variant = LatentSubspaceArm(model, beta=0.37)  # beta value must not matter when P_V is None

    control_loss = control.compute_loss(batch, {})
    variant_loss = variant.compute_loss(batch, {"P_V": None})

    assert torch.equal(control_loss, variant_loss)


def test_arms_do_diverge_from_control_when_masking_is_actually_enabled():
    """Guard against a vacuous isolation test: with a genuinely
    non-trivial signal, the variants MUST differ from control -- otherwise
    the 'disabled' tests above would be trivially true for the wrong reason
    (e.g. a no-op compute_loss)."""
    model, batch = _make_model_and_batch(seed=4)
    control = ControlArm(model)
    control_loss = control.compute_loss(batch, {})

    prio = SamplePrioArm(model, kappa=1.0)
    n = batch["x"].shape[0]
    nontrivial_signals = {
        "edge_rarity": torch.linspace(0.2, 5.0, n),
        "abstraction_failure_flag": (torch.arange(n) % 2).float(),
    }
    prio_loss = prio.compute_loss(batch, nontrivial_signals)
    assert not torch.equal(control_loss, prio_loss)

    subspace = LatentSubspaceArm(model, beta=0.1)
    d = batch["y"].shape[-1]
    P_V = torch.eye(d)[:, : d // 2] @ torch.eye(d)[:, : d // 2].T  # rank-(d//2) projector
    subspace_loss = subspace.compute_loss(batch, {"P_V": P_V})
    assert not torch.equal(control_loss, subspace_loss)


def test_reproducible_under_fixed_seed():
    """Same seed, same model init, same batch -> identical loss across two
    independently constructed arm instances (no hidden nondeterminism)."""
    model_a, batch_a = _make_model_and_batch(seed=42)
    model_b, batch_b = _make_model_and_batch(seed=42)

    loss_a = ControlArm(model_a).compute_loss(batch_a, {})
    loss_b = ControlArm(model_b).compute_loss(batch_b, {})
    assert torch.equal(loss_a, loss_b)
