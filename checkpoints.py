"""Load a specific JEPA checkpoint, or a random-init floor model, bypassing
`oracle.lewm_g.get_model()`'s memoized singleton (which always loads
`lewm_droid_s0/weights_epoch_20.pt`). E2 needs three DISTINCT models
(theta_p, theta_full, random-init) evaluated under the identical capability
ladder -- get_model()'s caching would make that impossible within one
process, and re-pointing its global would risk stale state leaking into
other modules that call get_model() expecting theta_0.

theta_p / theta_full existing on disk (every epoch 1-20 was checkpointed
during the original training run) means NO retraining is required for E2 --
see preregistration.md's 2026-08-24 entry.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle.lewm_g import DEVICE, _LEWM_DIR  # noqa: E402

RUN_NAME = "lewm_droid_s0"
FULL_EPOCH = 20
PARTIAL_EPOCH = 7   # 35% of 20 -- inside the doc's 30-40% band


def load_epoch(epoch: int):
    warnings.filterwarnings("ignore")
    from stable_worldmodel.wm.utils import load_pretrained
    model = load_pretrained(f"{RUN_NAME}/weights_epoch_{epoch}.pt").to(DEVICE).eval()
    model.requires_grad_(False)
    return model


def load_random_init():
    """Same architecture (from the checkpoint's own config.json), random
    weights -- the floor baseline E2 needs to prove the instrument can
    separate a trained model from an untrained one."""
    warnings.filterwarnings("ignore")
    import json
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from stable_worldmodel.data import get_cache_dir

    ckpt_dir = get_cache_dir(None, sub_folder="checkpoints") / RUN_NAME
    config = json.loads((ckpt_dir / "config.json").read_text())
    model = instantiate(OmegaConf.create(config)).to(DEVICE).eval()
    model.requires_grad_(False)
    return model


def load_named(name: str):
    """name in {'partial', 'full', 'random'} -- the three E2 models."""
    if name == "partial":
        return load_epoch(PARTIAL_EPOCH)
    if name == "full":
        return load_epoch(FULL_EPOCH)
    if name == "random":
        return load_random_init()
    raise ValueError(f"unknown checkpoint name {name!r}")
