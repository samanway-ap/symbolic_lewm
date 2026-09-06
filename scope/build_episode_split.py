"""Canonical 80/20 episode-level split of drawer_open_close, frozen once.
Every "held-out"/"train" reference from here onward (Phase B refit, Phase
C predicate fidelity, Phase F acceptance, Phase I eval) uses this exact
file -- not an ad hoc resample.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 3072
TRAIN_FRAC = 0.8
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    ids = json.loads((OUT_DIR / "scope_episode_ids.json").read_text())
    drawer_ids = sorted(ids["primary"]["episode_ids"])

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(drawer_ids)
    n_train = int(round(len(shuffled) * TRAIN_FRAC))
    train_ids = sorted(int(x) for x in shuffled[:n_train])
    test_ids = sorted(int(x) for x in shuffled[n_train:])

    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(drawer_ids)

    split = {
        "seed": SEED, "train_frac": TRAIN_FRAC, "n_total": len(drawer_ids),
        "n_train": len(train_ids), "n_test": len(test_ids),
        "train_episode_ids": train_ids, "test_episode_ids": test_ids,
    }
    (OUT_DIR / "episode_split.json").write_text(json.dumps(split, indent=2))
    print(f"train={len(train_ids)} test={len(test_ids)} -> episode_split.json")


if __name__ == "__main__":
    main()
